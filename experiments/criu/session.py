"""Launch, capture, reconstruct, and collect one Linux trainer process.

The controller is the Python process calling these functions. Its trainer is a
child process: a separate running program with its own memory and Linux process
identifier (PID). CRIU saves and reconstructs that process; its CUDA plugin owns
the corresponding NVIDIA save/restore operations. See docs/01-cpu-checkpointing.md
for the process and kernel concepts used here.
"""

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def child_environment(cuda_checkpoint):
    """Return only the environment variables needed by the isolated workload.

    A child inherits variables such as search paths unless given an explicit
    environment. Keeping this list small avoids accidental host configuration
    and sensitive variables appearing in tool logs.
    """
    # PATH lets the CRIU CUDA plugin find the pinned NVIDIA helper. Python ignores
    # user-installed packages, CPU libraries use one thread, and model access is
    # offline. The cuBLAS workspace setting supports deterministic CUDA operations.
    return {"PATH": f"{Path(cuda_checkpoint).parent}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(Path.home()), "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def adopt_restored_children():
    """Ask Linux to make this controller the parent of orphaned descendants.

    Detached restore leaves a trainer whose creating process has exited. As a
    subreaper, this controller adopts it and can later collect its exit status.
    Call this before launching CRIU, so ownership is established for its children.
    """
    # prctl is a Linux system call exposed by the C library. Option 36 means
    # PR_SET_CHILD_SUBREAPER; 1 enables it. errno describes a failed system call.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")


def alive(pid):
    """Report whether Linux still lists the PID as a non-zombie process.

    /proc is a kernel-provided view of running processes, not ordinary saved
    files. State Z means the process exited but its parent has not collected its
    status yet; that zombie cannot write another readiness marker.
    """
    try:
        # The state field follows the parenthesized process name in /proc/PID/stat.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def identity(pid):
    """Bind a PID to this boot, start tick, and owner, avoiding PID reuse."""
    directory = Path(f"/proc/{pid}")
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "uid": directory.stat().st_uid,
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def matches(expected):
    try:
        return identity(expected["pid"]) == expected and alive(expected["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return False


def observe_exit(expected, deadline, removed=False):
    """Observe an unrelated trainer; only its actual parent can collect status."""
    while True:
        try:
            current = identity(expected["pid"])
        except (FileNotFoundError, ProcessLookupError):
            return
        if current != expected:
            raise RuntimeError("PID reused while observing original exit")
        if not removed and not alive(expected["pid"]):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Original trainer has not exited/been reaped")
        time.sleep(0.05)


def kill_identified(expected):
    """Pin the process with a pidfd, then recheck identity before signaling."""
    try:
        fd = os.pidfd_open(expected["pid"])
    except ProcessLookupError:
        return
    try:
        if matches(expected):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
    finally:
        os.close(fd)


def wait_marker(path, pid, timeout=120):
    """Wait for a trainer-created control file, failing on exit or timeout."""
    # A monotonic clock measures elapsed time without calendar-clock corrections.
    # Polling also checks the child, so a crash does not look like a slow update.
    deadline = time.monotonic() + timeout
    while not path.exists():
        if not alive(pid):
            raise RuntimeError(f"Process exited before {path.name}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Waiting for {path.name}")
        time.sleep(0.05)


def reap(pid, process=None, timeout=30):
    """Collect a child's exit status, verify its process record is gone, return it.

    Exiting frees the running process's resources, but Linux retains a small
    zombie record until its parent calls wait. Removing that record is reaping;
    it must finish before CRIU attempts to reuse the captured numeric PID.
    """
    if process is not None:
        # Popen owns the original child and already provides a bounded wait.
        code = process.wait(timeout=timeout)
    else:
        # The restored child was adopted rather than created by Popen. WNOHANG
        # makes waitpid return immediately, allowing this loop to enforce a limit.
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
    """Force a leftover trainer to exit and collect it after a failed operation.

    A PID can be reused for an unrelated process. Check its command-line argument
    for this run directory before sending SIGKILL, Linux's forceful termination
    signal. This check reduces mistaken targeting; it is not an atomic PID guard.
    """
    try:
        # Linux separates /proc command-line arguments with zero bytes. A zombie
        # has an empty command line and needs only reaping, not another signal.
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        owned = str(run).encode() in cmdline.split(b"\0")
        if owned:
            os.kill(pid, signal.SIGKILL)
        # Check raw bytes: splitting an empty command line yields [b''], which
        # is a nonempty list. A live child can briefly have no command line
        # during launch too, so require an exited state before collecting it.
        if (not cmdline and not alive(pid)) or owned:
            reap(pid, process)
    except (FileNotFoundError, ProcessLookupError, ChildProcessError):
        pass


def launch(python, script, arguments, run, env):
    """Start a trainer with file-backed output and no connection to the terminal."""
    # File descriptors are handles to open resources. Regular log files and
    # /dev/null avoid terminal/pipe dependencies in the image. A new session
    # separates the child's terminal control; it does not grant extra permission.
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
    """Dump or restore one generation using CRIU and its native CUDA plugin.

    Dump leaves the original terminated; the caller still must reap it and verify
    the GPU handoff. Restore returns a PID whose trainer waits for inspection.
    No manual NVIDIA restore/unlock belongs here: the plugin performs both.
    """
    # Every capture needs fresh image and PID paths. This CRIU version creates
    # its PID file exclusively, so reusing one makes a later restore fail.
    directory = run / f"images-{generation}"
    pidfile = run / f"restored-{generation}.pid"
    if action == "dump":
        directory.mkdir()
    extra = ["--tree", str(pid)] if action == "dump" else ["--restore-detached", "--pidfile", str(pidfile)]
    # sudo requests the host privileges needed to inspect/reconstruct processes;
    # -n forbids an interactive password prompt. env -i removes inherited settings.
    # The inner timeout bounds CRIU itself; the outer timeout also covers sudo.
    # --shell-job permits the job's session/group arrangement. --libdir selects
    # the CUDA plugin, while --no-default-config prevents host config overrides.
    args = ["sudo", "-n", "env", "-i", "PATH=" + env["PATH"],
            "LD_LIBRARY_PATH=" + tools["libraries"], "timeout", "--kill-after=5", "120",
            tools["criu"], "--no-default-config", action,
            "--images-dir", str(directory), "--libdir", tools["plugin"], "--shell-job",
            "--log-file", action + ".log", "-v4", *extra]
    try:
        command(args, run / f"{action}-{generation}.stdout", env, timeout=130)
    finally:
        # CRIU's root-owned images/logs stay private beneath the mode-0700 run directory.
        # Return file ownership to the invoking user, including after failure.
        subprocess.run(["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(directory)], check=True)
    if action == "dump":
        # CRIU must report success and leave its image inventory and process tree.
        # This checks completed output, not storage survival after losing the host.
        if not all((directory / name).is_file() for name in ("inventory.img", "pstree.img")):
            raise RuntimeError("Successful dump lacks image inventory")
    else:
        subprocess.run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(pidfile)], check=True)
    log = (directory / (action + ".log")).read_text()
    # Command success alone cannot prove the selected CUDA plugin did the work.
    # Keep other warnings available through warnings(); do not silently erase them.
    required = "Checkpointing CUDA devices" if action == "dump" else "resuming devices on pid"
    if required not in log or "cuda_plugin: initialized:" not in log or " Error " in log or "unsupported" in log.lower():
        if action == "restore":
            cleanup(int(pidfile.read_text()), run)
        raise RuntimeError(f"CUDA plugin did not perform {action}")
    if action == "restore":
        return int(pidfile.read_text())


def release_verified(run, generation, reference):
    """Compare restored evidence before permitting the next training update.

    The trainer records its restored state before any repair. Both the captured
    state and that observation must match before creating the continue marker.
    """
    # Import only at verification; the controller never supplies replacement
    # training state. Marker files carry control requests, not model bytes.
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
