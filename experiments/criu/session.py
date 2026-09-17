"""Manage one trainer process; CRIU's CUDA plugin owns NVIDIA save/restore.

The controller starts a child with its own memory and process identifier (PID).
See docs/01-cpu-checkpointing.md for Linux process and resource concepts.
"""

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def child_environment(cuda_checkpoint):
    """Limit inherited settings and sensitive variables in child tool logs."""
    # Select the pinned helper, isolated packages, one CPU thread, offline assets,
    # and deterministic cuBLAS operations through explicit environment variables.
    return {"PATH": f"{Path(cuda_checkpoint).parent}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(Path.home()), "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def adopt_restored_children():
    """Adopt orphaned descendants so this controller can collect restored exits.

    Call before CRIU: its detached trainer loses its original parent. A Linux
    subreaper becomes the adopting parent; it cannot adopt unrelated processes.
    """
    # C library's prctl system call: option 36 enables PR_SET_CHILD_SUBREAPER.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")


def alive(pid):
    """Check Linux's /proc view for a running, non-zombie process.

    State Z means exited but not yet collected by its parent (a zombie).
    """
    try:
        # The state field follows the parenthesized process name in /proc/PID/stat.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_marker(path, pid, timeout=120):
    """Wait for a trainer-created control file, failing on exit or timeout."""
    # Monotonic time ignores calendar-clock changes; check for crashes while waiting.
    deadline = time.monotonic() + timeout
    while not path.exists():
        if not alive(pid):
            raise RuntimeError(f"Process exited before {path.name}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Waiting for {path.name}")
        time.sleep(0.05)


def reap(pid, process=None, timeout=30):
    """Collect a child's exit status and verify its process record is gone.

    Linux retains an exited child's record until its parent waits (reaps it).
    Reaping must finish before CRIU can reuse the captured numeric PID.
    """
    if process is not None:
        # Popen owns the original child and already provides a bounded wait.
        code = process.wait(timeout=timeout)
    else:
        # Adopted children lack Popen handles; WNOHANG permits a bounded waitpid loop.
        deadline = time.monotonic() + timeout
        while True:
            child, status = os.waitpid(pid, os.WNOHANG)
            if child:
                code = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Waiting for exit of {pid}")
            time.sleep(0.05)
    if Path(f"/proc/{pid}").exists():
        raise RuntimeError("Original PID still present")
    return code


def cleanup(pid, run, process=None):
    """Kill and reap a leftover trainer whose arguments identify this run.

    PIDs can be reused. This check reduces mistaken targeting, but is not atomic.
    """
    try:
        # /proc separates arguments with zero bytes; zombies have no arguments.
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        owned = str(run).encode() in cmdline.split(b"\0")
        if owned:
            os.kill(pid, signal.SIGKILL)  # Force termination of the identified trainer.
        # Empty bytes split into the nonempty list [b'']. Check raw bytes AND exit:
        # a live child can briefly expose an empty command line during launch.
        if (not cmdline and not alive(pid)) or owned:
            reap(pid, process)
    except (FileNotFoundError, ProcessLookupError, ChildProcessError):
        pass


def launch(python, script, arguments, run, env):
    """Start a trainer with file-backed output and no connection to the terminal."""
    # Log files and /dev/null avoid saved terminal/pipe handles. A new session
    # separates terminal control; it grants no additional permissions.
    with (run / "updates.jsonl").open("ab") as out, (run / "trainer.stderr").open("ab") as err:
        return subprocess.Popen([str(python), str(script), *map(str, arguments)], env=env,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)


def command(arguments, log, env, timeout=120):
    """Run a bounded command, retain both output streams, and fail on nonzero exit."""
    # Pass an argument list directly instead of asking a shell to interpret it.
    with log.open("wb") as output:
        return subprocess.run(list(map(str, arguments)), env=env, stdin=subprocess.DEVNULL,
                              stdout=output, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def criu(action, run, generation, tools, env, pid=None):
    """Dump terminates the original; restore returns a PID waiting for inspection.

    The caller must reap the original and verify GPU handoff. The CUDA plugin
    restores/unlocks NVIDIA state; do not repeat those transitions manually.
    """
    # CRIU refuses existing PID files; every generation needs fresh paths.
    directory = run / f"images-{generation}"
    pidfile = run / f"restored-{generation}.pid"
    if action == "dump":
        directory.mkdir()
    extra = ["--tree", str(pid)] if action == "dump" else ["--restore-detached", "--pidfile", str(pidfile)]
    # sudo -n requests host privileges without prompting; env -i clears inherited
    # settings. Inner/outer timeouts bound CRIU/sudo. Select the plugin explicitly,
    # permit this session arrangement (--shell-job), and ignore host CRIU config.
    args = ["sudo", "-n", "env", "-i", "PATH=" + env["PATH"],
            "LD_LIBRARY_PATH=" + tools["libraries"], "timeout", "--kill-after=5", "120",
            tools["criu"], "--no-default-config", action,
            "--images-dir", str(directory), "--libdir", tools["plugin"], "--shell-job",
            "--log-file", action + ".log", "-v4", *extra]
    try:
        command(args, run / f"{action}-{generation}.stdout", env, timeout=130)
    finally:
        # Return root-owned output to the caller even on failure; the run stays private.
        subprocess.run(["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(directory)], check=True)
    if action == "dump":
        # Require completed output; this alone does not establish durable storage.
        if not all((directory / name).is_file() for name in ("inventory.img", "pstree.img")):
            raise RuntimeError("Successful dump lacks image inventory")
    else:
        subprocess.run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(pidfile)], check=True)
    log = (directory / (action + ".log")).read_text()
    # Require CUDA plugin evidence; retain remaining diagnostics through warnings().
    required = "Checkpointing CUDA devices" if action == "dump" else "resuming devices on pid"
    if required not in log or "cuda_plugin: initialized:" not in log or " Error " in log or "unsupported" in log.lower():
        if action == "restore":
            cleanup(int(pidfile.read_text()), run)
        raise RuntimeError(f"CUDA plugin did not perform {action}")
    if action == "restore":
        return int(pidfile.read_text())


def release_verified(run, generation, reference):
    """Permit the next update only when reference, captured, and restored state agree.

    Restored evidence is recorded before repair. Markers carry permission, not
    replacement training state; import comparison code only when needed.
    """
    from state import compare
    if not (run / f"inspected-{generation}").exists():
        raise ValueError("Restored inspection has not completed")
    before = json.loads((run / f"state-{generation}.json").read_text())
    after = json.loads((run / f"after-{generation}.json").read_text())
    compare(reference, before)
    compare(before, after)
    (run / f"continue-{generation}").touch(exist_ok=False)


def warnings(run):
    """Collect compatibility warnings and errors from every generation's CRIU logs."""
    return [line.strip() for path in sorted(run.glob("images-*/*.log"))
            for line in path.read_text().splitlines()
            if " Warn " in line or " Error " in line or "unsupported" in line.lower()]
