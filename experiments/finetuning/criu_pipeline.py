"""Experiment orchestration; workers recover using files, never Trial objects."""

import json
from pathlib import Path
import subprocess
import sys
import time

import control
from pipeline import ROOT, event, finish, launch, session
from state import compare


def collect(trial, attempt, generation):
    """Copy worker events into the report while retaining their original clocks."""
    for line in (attempt / "events.jsonl").read_text().splitlines():
        row = {**json.loads(line), "generation": generation}
        if row["event"] == "restore_returned":
            row["restore_log"] = str(attempt / "restore.log")
        trial.events.append(row)
        if row["event"] == "local_snapshot_published":
            trial.events.append({**row, "event": "filesystem_synced"})


def run(trial):
    args, output = trial.args, trial.args.run_dir
    # Independent restoration always inspects untouched state, including timing
    # repetitions. Its cost is deliberately part of the reported recovery time.
    launch(trial, output, extra=["--external-control"])
    event(trial, "launched", pid=trial.pid)
    for generation in args.capture:
        snapshot = output / "snapshots" / str(generation)
        event(trial, "capture_worker_launched", generation=generation)
        with (output / f"checkpoint-{generation}.log").open("wb") as log:
            worker = subprocess.Popen([
                sys.executable, str(ROOT / "experiments/finetuning/checkpoint.py"),
                "--run-dir", str(output), "--snapshot", str(snapshot),
                "--tools", str(args.tools), "--at", str(generation)],
                env=trial.env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            try:
                # Only this launch/adoption parent collects the trainer's status.
                # The unrelated worker observes removal before publishing success.
                session.reap(trial.pid, trial.process, timeout=310)
                trial.pid, trial.process = None, None
                if worker.wait(timeout=310) != 0:
                    raise RuntimeError(f"Capture worker failed: see checkpoint-{generation}.log")
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
        event(trial, "capture_worker_exited", generation=generation)
        manifest = control.read(snapshot / "manifest.json")
        if manifest["update"] != generation:
            raise ValueError("Automated capture missed its requested boundary")
        collect(trial, output / "attempts" / manifest["capture_id"], generation)
        compare(control.read(args.reference / f"state-{generation}.json"), control.read(snapshot / "before.json"))
        event(trial, "job_b_requested", generation=generation)
        session.command([sys.executable, ROOT / "experiments/finetuning/job_b.py"],
                        output / f"job-b-{generation}.log", trial.env, timeout=60)
        event(trial, "job_b_completed", generation=generation)
        event(trial, "restore_worker_launched", generation=generation)
        session.command([sys.executable, ROOT / "experiments/finetuning/restore.py",
                         "--snapshot", snapshot, "--tools", args.tools],
                        output / f"restore-{generation}.log", trial.env, timeout=310)
        event(trial, "restore_worker_exited", generation=generation)
        phase = control.read(output / "control/phase.json")
        attempt = output / "attempts" / phase["attempt_id"]
        collect(trial, attempt, generation)
        restored = control.read(attempt / "result.json")["identity"]
        trial.pid = restored["pid"]
        # The worker is gone. The harness is only the reaping/verification parent;
        # no recovery state was passed to the worker and no supervisor is added.
        if not session.matches(restored):
            raise RuntimeError("Restored trainer did not survive worker exit")
        event(trial, "survived_worker_exit", generation=generation, pid=trial.pid)
    finish(trial)
    for generation in args.capture:
        snapshot = output / "snapshots" / str(generation)
        if control.inventory(snapshot) != control.read(snapshot / "manifest.json")["payload"]:
            raise ValueError("Restore mutated the immutable snapshot payload")
