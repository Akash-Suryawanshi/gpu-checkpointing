"""Capture/restore ownership and bounded cleanup for the inference supervisor."""

from contextlib import contextmanager
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from worker import control
from control import session

HERE = Path(__file__).resolve().parent


class CleanupError(RuntimeError):
    """A helper group could still be active; further GPU work must stop."""


def helper_status(process):
    # WNOWAIT keeps the leader's PID reserved until its whole group is stopped.
    # Popen.poll()/wait() would reap it early, opening a group-ID reuse race.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


@contextmanager
def helper(arguments, log, env, deadline, privileged=True):
    """Own one helper group, including sudo/CRIU, until cancellation cleanup ends."""
    control.remaining(deadline)
    with log.open("wb") as stream:
        process = subprocess.Popen(list(map(str, arguments)), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env={**env, "GPU_SNAPSHOT_HELPER_GROUP": "1"})
        try:
            yield process
        finally:
            try:
                # The unreaped leader pins this group ID. Restored workers retain
                # their saved, separate session and are handled by identity below.
                code = "import os,signal,sys\ntry: os.killpg(int(sys.argv[1]),signal.SIGKILL)\nexcept ProcessLookupError: pass"
                if privileged:
                    subprocess.run(["sudo", "-n", "python3", "-c", code, str(process.pid)],
                                   check=True, timeout=10, stdin=subprocess.DEVNULL)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=10)
                cleanup_deadline = time.monotonic() + 10
                while True:
                    try:
                        pid, _ = os.waitpid(-process.pid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if not pid:
                        control.remaining(cleanup_deadline)
                        time.sleep(0.005)
            except BaseException as error:
                raise CleanupError("Helper group cleanup failed") from error


def wait_helper(process, deadline):
    while (status := helper_status(process)) is None:
        time.sleep(min(0.005, control.remaining(deadline)))
    control.remaining(deadline)
    if status.si_code != os.CLD_EXITED or status.si_status:
        raise RuntimeError("Independent helper failed; see its log")


def capture(run, tools, env, process, deadline, contract=None, model_policy="strict-v1"):
    arguments = [sys.executable, HERE.parent / "finetuning/checkpoint.py", "--run-dir", run,
        "--snapshot", run / "snapshot", "--tools", tools, "--at", "1",
        "--timeout", str(control.remaining(deadline)), "--model-policy", model_policy,
        *(["--contract", contract] if contract else [])]
    with helper(arguments, run / "capture.log", env, deadline) as command:
        while process.poll() is None:
            control.remaining(deadline)
            if helper_status(command) is not None:
                raise RuntimeError("Capture exited before original; see capture.log")
            time.sleep(0.005)
        session.reap(process.pid, process, control.remaining(deadline))
        wait_helper(command, deadline)
    if control.read(run / "control/phase.json")["status"] != "published":
        raise ValueError("Snapshot publication incomplete")
    return {"original_reaped": True, "capture_exited": True}


def restore(run, tools, env, deadline):
    order = control.read(run / "run.json")["key"].get("validation_order", "payload-first")
    workers = control.read(run / "run.json")["key"].get("validation_workers", 1)
    arguments = [sys.executable, HERE.parent / "finetuning/restore.py", "--snapshot", run / "snapshot",
                 "--tools", tools, "--timeout", str(control.remaining(deadline)), "--validation-order", order,
                 "--validation-workers", str(workers)]
    with helper(arguments, run / "restore.log", env, deadline) as command:
        wait_helper(command, deadline)
    phase = control.current_attempt(run)
    identity = control.read(run / "attempts" / phase["attempt_id"] / "result.json")["identity"]
    if not session.matches(identity):
        raise RuntimeError("Restored worker did not survive restore-command exit")
    return identity


def restored_candidate(run):
    """Registration can exist after release even when the final result is missing."""
    if not (run / "snapshot/manifest.json").exists() or not (run / "control/phase.json").exists():
        return None
    manifest, phase = control.read(run / "snapshot/manifest.json"), control.current_attempt(run)
    if "attempt_id" not in phase:
        return None
    if phase["status"] == "released":
        return None  # A reusable image between activations owns no process.
    if (manifest["run"] != str(run) or phase.get("capture_id") != manifest["capture_id"]
            or phase["status"] not in ("restoring", "verified", "restored", "failed_restore")):
        raise ValueError("Ambiguous restored cleanup ownership")
    attempt = phase["attempt_id"]
    pidfile = run / "attempts" / attempt / f"restored-{attempt}.pid"
    if not pidfile.exists():
        original_pid = manifest["job"]["identity"]["pid"]
        stat = Path(f"/proc/{original_pid}/stat")
        if stat.exists() and int(stat.read_text().rsplit(")", 1)[1].split()[1]) == os.getpid():
            raise CleanupError("Partial restore exists without attempt PID evidence")
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
