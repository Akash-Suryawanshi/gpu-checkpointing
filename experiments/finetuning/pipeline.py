"""Shared steps; each dedicated pipeline keeps its save/dump/restore sequence.

The controller supervises the trainer process (running program). Empty marker
files communicate readiness or permission between them, never model state.
"""

from argparse import Namespace
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import time

import state

ROOT = Path(__file__).resolve().parents[2]
# Keep Linux process/CRIU command details in their dedicated session module.
sys.path.insert(0, str(ROOT / "experiments/criu"))
import session


@dataclass
class Trial:
    """Track the active trainer for sampling and cleanup, including after errors.

    PID is Linux's process number; Popen is Python's handle for a child it started.
    CRIU restores a PID without Popen. ``env`` holds child environment variables;
    ``directory`` identifies its files. Model state stays in the trainer.
    """

    args: Namespace
    tools: dict
    env: dict
    events: list = field(default_factory=list)
    pid: int | None = None
    process: subprocess.Popen | None = None
    directory: Path | None = None


def event(trial, name, **values):
    """Record an event using the controller's elapsed-time clock, in nanoseconds."""
    row = {"event": name, "monotonic_ns": time.monotonic_ns(), **values}
    trial.events.append(row)
    print(json.dumps(row), flush=True)


def launch(trial, directory, pause=(), extra=()):
    """Start a trainer and retain its identity before waiting on any markers."""
    args = trial.args
    arguments = ["--assets", args.assets, "--run-dir", directory,
                 "--dropout", args.dropout, "--until", args.until]
    if pause:
        arguments += ["--pause-at", ",".join(map(str, pause))]
    if args.timing:
        arguments += ["--timing"]
    # The directory also identifies the child during guarded failure cleanup.
    trial.directory = directory
    trial.process = session.launch(sys.executable, ROOT / "experiments/finetuning/train.py",
                                   [*arguments, *extra], directory, trial.env)
    trial.pid = trial.process.pid


def capture_boundary(trial, generation):
    """Check the completed update identified by generation, then request capture.

    Timing skips inspection only after run.py admits matching correctness evidence.
    """
    args = trial.args
    session.wait_marker(args.run_dir / f"ready-{generation}", trial.pid, 300)
    expected = None
    if not args.timing:
        expected = json.loads((args.reference / f"state-{generation}.json").read_text())
        state.compare(expected, json.loads((args.run_dir / f"state-{generation}.json").read_text()))
    event(trial, "capture_requested", generation=generation)
    return expected


def handoff(trial, generation, require_clean_exit=False):
    """Verify exit → GPU availability → local sync → job B → restore request.

    Reaping collects the child's exit status and removes its process record.
    Observe GPU availability afterward: release during staging can be temporary.
    """
    output = trial.args.run_dir
    code = session.reap(trial.pid, trial.process)
    # Application save must exit normally; CRIU can terminate its original.
    if require_clean_exit and code != 0:
        raise RuntimeError("Application save process failed")
    event(trial, "original_exit_verified", generation=generation, pid=trial.pid)
    # Keep the PID for partial-restore cleanup: CRIU recreates that same number.
    trial.process = None
    event(trial, "gpu_observation_started", generation=generation)
    gpu_log = output / f"gpu-after-exit-{generation}.csv"
    session.command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                    gpu_log, trial.env)
    if gpu_log.read_text().strip():
        raise RuntimeError("GPU compute processes remain after original exit")
    event(trial, "gpu_observed_after_exit", generation=generation)
    # Flush this filesystem's writes; this does not copy images to another host.
    session.command(["sync", "-f", output], output / f"sync-{generation}.log", trial.env)
    event(trial, "filesystem_synced", generation=generation)
    event(trial, "job_b_requested", generation=generation)
    # A separate allocation and checked sum prove another job can use CUDA.
    session.command([sys.executable, ROOT / "experiments/finetuning/job_b.py"],
                    output / f"job-b-{generation}.log", trial.env, timeout=60)
    event(trial, "job_b_completed", generation=generation)
    event(trial, "restore_requested", generation=generation)


def inspect_restore(trial, generation, expected, *, request_inspection=False):
    """Record restore return, save diagnostic maps, then optionally request inspection.

    CRIU needs an inspect marker; application loading starts inspection itself.
    Diagnostic continuation requires reference → before → after equality.
    Timing skips maps/comparison but still releases CRIU's saved wait.
    """
    output = trial.args.run_dir
    event(trial, "restore_returned", generation=generation, pid=trial.pid)
    if not trial.args.timing:
        # Linux lists this process's mapped memory ranges for later diagnosis.
        (output / f"restored-{generation}.maps").write_text(Path(f"/proc/{trial.pid}/maps").read_text())
    if request_inspection:
        (output / f"inspect-{generation}").touch(exist_ok=False)
    if not trial.args.timing:
        session.wait_marker(output / f"inspected-{generation}", trial.pid)
        session.release_verified(output, generation, expected)
        event(trial, "continuation_permitted", generation=generation)


def finish(trial, record_exit=True):
    """Require both the trainer's completion marker and a successful exit."""
    session.wait_marker(trial.directory / "done", trial.pid, 300)
    if session.reap(trial.pid, trial.process) != 0:
        raise RuntimeError("Trainer did not exit successfully")
    if record_exit:
        event(trial, "final_exit_verified", pid=trial.pid)
    # Failure cleanup must no longer target this exited trainer.
    trial.pid, trial.process = None, None


def compare_run(reference, actual, until, timing=False):
    """Compare every loss and boundary exactly; timing saves only the final boundary.

    Final timing inspection happens after the measured next-update endpoint.
    """
    for update in ([until] if timing else range(until + 1)):
        state.compare(json.loads((reference / f"state-{update}.json").read_text()),
                      json.loads((actual / f"state-{update}.json").read_text()))
    expected = [json.loads(s)["loss"] for s in (reference / "updates.jsonl").read_text().splitlines()]
    observed = [json.loads(s)["loss"] for s in (actual / "updates.jsonl").read_text().splitlines()]
    state.compare(expected, observed, "losses")
