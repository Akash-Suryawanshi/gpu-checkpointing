"""Restore a published snapshot, verify untouched state, and leave training alive."""

import argparse
import os
from pathlib import Path
import time

import control
from control import session
from state import compare


def release(snapshot, attempt, restoration, signal):
    """Inspection JSON is comparison evidence, never a source of model state."""
    if control.read(attempt / "inspected.json") != {"attempt_id": restoration["attempt_id"]}:
        raise ValueError("Inspection identity mismatch")
    compare(control.read(snapshot / "before.json"), control.read(attempt / "after.json"))
    control.write(signal, restoration)


def restore(args):
    os.umask(0o077)
    snapshot = args.snapshot.resolve()
    deadline = time.monotonic() + args.timeout
    run = Path(control.read(snapshot / "manifest.json")["run"])
    with control.lock(run, deadline):
        manifest = control.validate(snapshot, run, args.tools, deadline)
        capture_id = manifest["capture_id"]
        attempt_id, attempt = control.attempt(run)
        restoration = {"capture_id": capture_id, "attempt_id": attempt_id}
        emit = lambda name, **values: control.event(attempt, name, **restoration, generation=manifest["update"], **values)
        env = session.child_environment(args.tools / "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint")
        expected = None
        released = False
        session.adopt_restored_children()
        try:
            control.phase(run, "restoring", **restoration, snapshot=str(snapshot))
            emit("restore_requested")
            pid = session.criu("restore", run, attempt_id, control.tools_config(args.tools), env,
                               images=snapshot / "images", attempt=attempt, timeout=control.remaining(deadline))
            expected = session.identity(pid)
            emit("restore_returned", pid=pid)
            job = {**manifest["job"], "identity": expected}
            control.write(run / "job.json", job)
            control.write(run / "attempts" / capture_id / "inspect.json", restoration)
            control.wait(attempt / "inspected.json", deadline, expected)
            control.remaining(deadline)
            # Persist the terminal phase before release. A failure never creates
            # continue; only this identified attempt is eligible for cleanup.
            compare(control.read(snapshot / "before.json"), control.read(attempt / "after.json"))
            control.phase(run, "verified", **restoration, snapshot=str(snapshot))
            release(snapshot, attempt, restoration, attempt / "continue.json")
            released = True
            control.phase(run, "restored", **restoration, snapshot=str(snapshot))
            emit("continuation_permitted", pid=pid)
            control.write(attempt / "result.json", {"identity": expected, **restoration})
            return expected
        except BaseException:
            if not released:
                if expected is None:
                    pidfile = attempt / f"restored-{attempt_id}.pid"
                    if pidfile.exists():
                        pid = int(pidfile.read_text())
                        if session.alive(pid):
                            candidate = session.identity(pid)
                            # Require the actual attempt's adopted child and job arguments.
                            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                            if int(fields[1]) == os.getpid() and str(run).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0"):
                                expected = candidate
                if expected is not None:
                    session.kill_identified(expected)
                    session.reap(expected["pid"])
                control.phase(run, "failed_restore", **restoration, snapshot=str(snapshot))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    restore(parser.parse_args())
