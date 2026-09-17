"""Capture/restore ownership and bounded cleanup for the inference supervisor."""

import os
from pathlib import Path
import subprocess
import sys
import time

from worker import control
from control import session

HERE = Path(__file__).resolve().parent


def capture(run, tools, env, process, deadline):
    with (run / "capture.log").open("wb") as log:
        command = subprocess.Popen([sys.executable, str(HERE.parent / "finetuning/checkpoint.py"),
            "--run-dir", str(run), "--snapshot", str(run / "snapshot"), "--tools", str(tools),
            "--at", "1", "--timeout", str(control.remaining(deadline))],
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                control.remaining(deadline)
                if command.poll() is not None:
                    raise RuntimeError("Capture exited before original; see capture.log")
                time.sleep(0.005)
            session.reap(process.pid, process, control.remaining(deadline))
            if command.wait(timeout=control.remaining(deadline)):
                raise RuntimeError("Capture failed; see capture.log")
        finally:
            if command.poll() is None:
                command.kill()
                command.wait(timeout=10)
    if control.read(run / "control/phase.json")["status"] != "published":
        raise ValueError("Snapshot publication incomplete")
    return {"original_reaped": True, "capture_exited": True}


def restore(run, tools, env, deadline):
    session.command([sys.executable, HERE.parent / "finetuning/restore.py", "--snapshot", run / "snapshot",
                     "--tools", tools, "--timeout", str(control.remaining(deadline))],
                    run / "restore.log", env, timeout=control.remaining(deadline))
    phase = control.read(run / "control/phase.json")
    identity = control.read(run / "attempts" / phase["attempt_id"] / "result.json")["identity"]
    if not session.matches(identity):
        raise RuntimeError("Restored worker did not survive restore-command exit")
    return identity


def restored_candidate(run):
    """Registration can exist after release even when the final result is missing."""
    if not (run / "snapshot/manifest.json").exists() or not (run / "control/phase.json").exists():
        return None
    manifest, phase = control.read(run / "snapshot/manifest.json"), control.read(run / "control/phase.json")
    if "attempt_id" not in phase:
        return None
    if (manifest["run"] != str(run) or phase.get("capture_id") != manifest["capture_id"]
            or phase["status"] not in ("restoring", "verified", "restored", "failed_restore")):
        raise ValueError("Ambiguous restored cleanup ownership")
    attempt = phase["attempt_id"]
    pidfile = run / "attempts" / attempt / f"restored-{attempt}.pid"
    if not pidfile.exists():
        return None
    pid = int(pidfile.read_text())
    if not Path(f"/proc/{pid}").exists():
        return None
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if int(fields[1]) != os.getpid():
        raise ValueError("Restored PID is not an adopted child")
    if fields[0] == "Z":
        session.reap(pid, timeout=10)
        return None
    identity = session.identity(pid)
    job = control.read(run / "job.json")
    if job["job_id"] != manifest["job"]["job_id"] or job["run"] != str(run):
        raise ValueError("Restored registration belongs to another job")
    registered = job["identity"] == identity
    # Before host registration, only the actual attempt's adopted child with
    # this exact run argument is eligible. A reused PID is never sufficient.
    early = phase["status"] in ("restoring", "failed_restore") and str(run).encode() in Path(
        f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if not (registered or early) or not session.matches(identity):
        raise ValueError("Restored identity mismatch")
    return identity


def cleanup(run, original=None, process=None):
    candidates = []
    if original is not None and Path(f"/proc/{original['pid']}").exists():
        candidates.append(original)
    restored = restored_candidate(run)
    if restored and restored not in candidates:
        candidates.append(restored)
    for identity in candidates:
        pid = identity["pid"]
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if int(fields[1]) != os.getpid():
            raise ValueError("Cleanup requires parent ownership")
        if fields[0] != "Z":
            if not session.matches(identity):
                raise ValueError("Cleanup identity changed")
            session.kill_identified(identity)
        session.reap(pid, process if process and process.pid == pid else None, timeout=10)
