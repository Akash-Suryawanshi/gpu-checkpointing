"""Probe CPU or GPU process restoration in the intended execution environment."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/inference"))
import lifecycle
from worker import control
from control import session


def probe(args):
    run, tools = args.output.resolve(), args.tools.resolve()
    run.mkdir(mode=0o700, parents=True)
    (run / "images").mkdir()
    (run / "empty-plugins").mkdir()
    deadline = time.monotonic() + args.timeout
    env = session.child_environment(tools / "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint")
    config = control.tools_config(tools)
    process, expected, saved_pid = None, None, None
    session.adopt_restored_children()
    def command(arguments, name, privileged=False):
        with lifecycle.helper(arguments, run / name, env, deadline, privileged) as child:
            lifecycle.wait_helper(child, deadline)
    def criu(action):
        extra = ["--tree", str(expected["pid"])] if action == "dump" else ["--restore-detached", "--pidfile", str(run / "restored.pid")]
        try:
            command(["sudo", "-n", "env", "-i", "PATH=" + env["PATH"], "LD_LIBRARY_PATH=" + config["libraries"],
                config["criu"], "--no-default-config", action, "--images-dir", run / "images",
                "--work-dir", run, "--libdir", run / "empty-plugins" if args.mode == "cpu" else config["plugin"],
                "--shell-job", "--log-file", run / (action + ".log"), "-v4", *extra], action + ".stdout", True)
        finally:
            # Root-created images and the PID record must remain inspectable by the caller.
            session.command(["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", run],
                            run / "ownership.log", env, timeout=10)
        log = (run / (action + ".log")).read_text()
        if " Error " in log or "unsupported" in log.lower():
            raise RuntimeError("CRIU probe reported an error")
        if args.mode == "gpu":
            required = "Checkpointing CUDA devices" if action == "dump" else "resuming devices on pid"
            if required not in log or "cuda_plugin: initialized:" not in log:
                raise ValueError("CUDA plugin did not perform the probe")
    try:
        target = ROOT / ("experiments/cpu/cpu_counter.py" if args.mode == "cpu" else "experiments/gpu/gpu_memory_probe.py")
        if args.mode == "cpu":
            (run / "input.txt").write_text("".join(f"{i:04d}\n" for i in range(1, 26)))
            command([sys.executable, target, "--mode", "baseline", "--run-dir", run], "baseline.jsonl")
        process = session.launch(sys.executable, target,
            ["--run-dir", run, *(["--mode", "pause"] if args.mode == "cpu" else [])], run, env)
        saved_pid = process.pid
        expected = session.identity(process.pid)
        session.wait_marker(run / "ready", process.pid, timeout=control.remaining(deadline))
        criu("dump")
        session.reap(process.pid, process, timeout=control.remaining(deadline))
        expected, process = None, None
        if args.mode == "gpu":
            command([sys.executable, ROOT / "experiments/finetuning/job_b.py"], "job-b.log")
        criu("restore")
        expected = session.identity(int((run / "restored.pid").read_text()))
        (run / "release").touch()
        session.wait_marker(run / "done", expected["pid"], timeout=control.remaining(deadline))
        if session.reap(expected["pid"], timeout=control.remaining(deadline)):
            raise RuntimeError("Restored probe exited nonzero")
        expected = None
        (run / "process.jsonl").write_bytes((run / "updates.jsonl").read_bytes())
        if args.mode == "cpu":
            command([sys.executable, ROOT / "experiments/cpu/verify_cpu.py", run], "comparison.json")
        else:
            before, after = [json.loads(line) for line in (run / "process.jsonl").read_text().splitlines()]
            if before["sha256"] != after["sha256"] or not after["next_gpu_operation_correct"]:
                raise ValueError("Restored GPU bytes or next operation differ")
        control.write(run / "result.json", {"status": "passed", "mode": args.mode,
                      "original_reaped": True, "restored_reaped": True})
    finally:
        if expected is not None:
            session.kill_identified(expected)
            session.reap(expected["pid"], process, timeout=10)
        elif saved_pid is not None:
            pidfile = run / "restored.pid"
            pid = int(pidfile.read_text()) if pidfile.exists() else saved_pid
            # Partial restoration can precede the PID file. Require this
            # supervisor's adopted child and the probe's exact run argument.
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and int(stat.read_text().rsplit(")", 1)[1].split()[1]) == os.getpid():
                session.cleanup(pid, run)
                if stat.exists():
                    raise lifecycle.CleanupError("Partial probe could not be identified and reaped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    probe(parser.parse_args())
