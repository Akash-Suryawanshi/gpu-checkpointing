"""Small CRIU lifecycle operations for this single-process experiment."""

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def child_environment(cuda_checkpoint):
    return {"PATH": f"{Path(cuda_checkpoint).parent}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(Path.home()), "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def adopt_restored_children():
    # CRIU detaches; adopt its restored child so this controller can reap it.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_marker(path, pid, timeout=120):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if not alive(pid):
            raise RuntimeError(f"Process exited before {path.name}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Waiting for {path.name}")
        time.sleep(0.05)


def reap(pid, process=None, timeout=30):
    if process is not None:
        code = process.wait(timeout=timeout)
    else:
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
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if str(run).encode() in cmdline:
            os.kill(pid, signal.SIGKILL)
        if not cmdline or str(run).encode() in cmdline:
            reap(pid, process)
    except (FileNotFoundError, ProcessLookupError, ChildProcessError):
        pass


def launch(python, script, arguments, run, env):
    with (run / "updates.jsonl").open("ab") as out, (run / "trainer.stderr").open("ab") as err:
        return subprocess.Popen([str(python), str(script), *map(str, arguments)], env=env,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)


def command(arguments, log, env, timeout=120):
    with log.open("wb") as output:
        return subprocess.run(list(map(str, arguments)), env=env, stdin=subprocess.DEVNULL,
                              stdout=output, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def criu(action, run, generation, tools, env, pid=None):
    directory = run / f"images-{generation}"
    pidfile = run / f"restored-{generation}.pid"
    if action == "dump":
        directory.mkdir()
    extra = ["--tree", str(pid)] if action == "dump" else ["--restore-detached", "--pidfile", str(pidfile)]
    args = ["sudo", "-n", "env", "-i", "PATH=" + env["PATH"],
            "LD_LIBRARY_PATH=" + tools["libraries"], "timeout", "--kill-after=5", "120",
            tools["criu"], "--no-default-config", action,
            "--images-dir", str(directory), "--libdir", tools["plugin"], "--shell-job",
            "--log-file", action + ".log", "-v4", *extra]
    try:
        command(args, run / f"{action}-{generation}.stdout", env, timeout=130)
    finally:
        # CRIU's root-owned images/logs stay private beneath the mode-0700 run directory.
        subprocess.run(["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(directory)], check=True)
    if action == "dump":
        if not all((directory / name).is_file() for name in ("inventory.img", "pstree.img")):
            raise RuntimeError("Successful dump lacks image inventory")
    else:
        subprocess.run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(pidfile)], check=True)
    log = (directory / (action + ".log")).read_text()
    required = "Checkpointing CUDA devices" if action == "dump" else "resuming devices on pid"
    if required not in log or "cuda_plugin: initialized:" not in log or " Error " in log or "unsupported" in log.lower():
        if action == "restore":
            cleanup(int(pidfile.read_text()), run)
        raise RuntimeError(f"CUDA plugin did not perform {action}")
    if action == "restore":
        return int(pidfile.read_text())


def release_verified(run, generation, reference):
    # Import only at verification; the controller never provides restore state to the trainer.
    from state import compare
    if not (run / f"inspected-{generation}").exists():
        raise ValueError("Restored inspection has not completed")
    before = json.loads((run / f"state-{generation}.json").read_text())
    after = json.loads((run / f"after-{generation}.json").read_text())
    compare(reference, before)
    compare(before, after)
    (run / f"continue-{generation}").touch(exist_ok=False)


def warnings(run):
    return [line.strip() for path in sorted(run.glob("images-*/*.log"))
            for line in path.read_text().splitlines()
            if " Warn " in line or " Error " in line or "unsupported" in line.lower()]
