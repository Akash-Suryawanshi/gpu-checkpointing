"""AC1: prove original-process exit and data continuation from a DMTCP image.

This bounded research probe uses a private coordinator and existing workloads.
It rejects unsupported-resource warnings, even when DMTCP returns success.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def command(args: list[str], log: Path, timeout: int = 90,
            check: bool = True) -> subprocess.CompletedProcess:
    """Run a bounded tool command, appending both output streams to its log."""
    with log.open("ab") as output:
        return subprocess.run(args, stdout=output, stderr=subprocess.STDOUT,
                              timeout=timeout, check=check)


def wait_marker(path: Path, process: subprocess.Popen, timeout: int = 90) -> None:
    """Wait for a control file while watching the child for exit or timeout."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"Process exited {process.returncode} before {path.name}")
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.1)


def main() -> None:
    """Run one historical DMTCP capture/restore and retain qualified evidence.

    DMTCP inserts checkpoint support into the launched program. Its coordinator
    is a separate control process; the CUDA plugin manages GPU transitions. This
    probe does not establish the compatibility of the later LoRA workload.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cpu", "gpu"])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dmtcp-root", type=Path, required=True,
                        help="DMTCP build containing bin/ and plugin/cuda/")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    tool = args.dmtcp_root.resolve()
    run = args.run_dir.resolve()
    os.umask(0o077)
    # A private, fresh directory protects logs/images and prevents stale markers
    # from accidentally releasing a new workload before restoration is complete.
    run.mkdir()
    for child in ("images", "tmp"):
        (run / child).mkdir()
    processes = []
    port_files = []
    # DMTCP logs the environment; only pass the variables this probe needs.
    env = {key: os.environ[key] for key in
           ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LD_LIBRARY_PATH")
           if key in os.environ}
    env["DMTCP_DISABLE_PRGNAME_PREFIX"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    controller = run / "controller.log"
    events = []

    def launch(binary: str, arguments: list[str], phase: str) -> tuple[subprocess.Popen, Path]:
        """Start a workload/restart with its own coordinator and file-backed logs."""
        # Port 0 asks the OS to choose a free communication port. The port file
        # tells dmtcp_command how to address this coordinator, not another run's.
        port = run / f"{phase}.port"
        port_files.append(port)
        common = [str(tool / "bin" / binary), "--new-coordinator", "--coord-port", "0",
                  "--port-file", str(port), "--ckptdir", str(run / "images"),
                  "--tmpdir", str(run / "tmp"), "--coord-logfile", str(run / f"{phase}.coordinator.log")]
        with (run / "process.jsonl").open("ab") as out, (run / f"{phase}.stderr").open("wb") as err:
            # No terminal or pipe must survive capture. Append mode lets the
            # restored process continue writing after the original's log records.
            process = subprocess.Popen(common + arguments, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, env=env, cwd=root,
                                       start_new_session=True)
        processes.append(process)
        return process, port

    try:
        # Reuse the small CPU/file-position or GPU/tensor workload. Here the
        # surrounding DMTCP lifecycle, not the workload, supplies process restore.
        if args.mode == "cpu":
            (run / "input.txt").write_text("".join(f"{i:04d}\n" for i in range(1, 26)))
            target = ["python3", str(root / "experiments/cpu/cpu_counter.py"), "--mode", "pause", "--run-dir", str(run)]
        else:
            target = ["--with-plugin", str(tool / "plugin/cuda/libdmtcp_cuda.so"),
                      "python3", str(root / "experiments/gpu/gpu_memory_probe.py"), "--run-dir", str(run)]
        original, port = launch("dmtcp_launch", ["--no-gzip"] + target, "original")
        wait_marker(run / "ready", original)
        wait_marker(port, original)
        events.append({"event": "ready", "real_pid": original.pid})
        # /proc/PID/maps lists virtual-address ranges and their backing objects.
        # Kernel-owned anonymous objects may need more than copying their bytes.
        # This guard does not prove that every other shared mapping is supported.
        maps = Path(f"/proc/{original.pid}/maps").read_text()
        (run / "original.maps").write_text(maps)
        if "anon_inode:" in maps:
            raise RuntimeError("Anonymous kernel-backed mapping needs explicit support; refusing acceptance")
        start = time.monotonic()
        command([str(tool / "bin/dmtcp_command"), "--coord-port", port.read_text().strip(), "--checkpoint"], controller)
        # This pinned revision can return from the request before image completion.
        # In its normal uncompressed path, a .temp image becomes .dmtcp after the
        # write barrier. The final name matters here, not a guessed file-size delay.
        deadline = time.monotonic() + 90
        images = []
        while not images:
            if original.poll() is not None:
                raise RuntimeError("Original exited before checkpoint image completed")
            if time.monotonic() > deadline:
                raise TimeoutError("Checkpoint image did not finalize")
            images = list((run / "images").rglob("*.dmtcp"))
            time.sleep(0.1)
        if len(images) != 1:
            raise RuntimeError(f"Expected exactly one process image, found {len(images)}")
        events.append({"event": "image_written", "bytes": images[0].stat().st_size,
                       "command_seconds": time.monotonic() - start})
        # Writing the image can resume the original and reacquire GPU resources.
        # End it explicitly, collect its exit status, and verify its /proc record
        # is gone before claiming a handoff to a different GPU job.
        command([str(tool / "bin/dmtcp_command"), "--coord-port", port.read_text().strip(), "--quit"], controller)
        original.wait(timeout=15)
        if Path(f"/proc/{original.pid}").exists():
            raise RuntimeError("Original process still exists")
        events.append({"event": "original_gone", "real_pid": original.pid})
        # Ask the filesystem to flush pending writes. This is local persistence,
        # not proof of survival after instance deletion or a move to another host.
        command(["sync", "-f", str(images[0])], controller)
        if args.mode == "gpu":
            # Inspect GPU ownership after exit and run independent GPU work. An
            # earlier drop in monitored memory could have been only temporary.
            command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], run / "gpu-after-original-exit.txt")
            command(["python3", "-c", "import torch; x=torch.ones(1024*1024,device='cuda'); value=x.sum().item(); assert value==1048576; print(value)"], run / "job-b.txt")
        restored, restored_port = launch("dmtcp_restart", [str(images[0])], "restored")
        wait_marker(restored_port, restored)
        # GPU state may still be rebuilding; releasing the application only lets
        # its next GPU call wait for the plugin to finish restoration.
        (run / "release").touch()
        wait_marker(run / "done", restored)
        code = restored.wait(timeout=15)
        if code != 0:
            raise RuntimeError(f"Restored process exited {code}")
        rows = [json.loads(line) for line in (run / "process.jsonl").read_text().splitlines() if line.startswith("{")]
        # Check continuation from the saved boundary, not merely exit code zero.
        if args.mode == "gpu":
            ready = [r for r in rows if r["event"] == "ready"]
            done = [r for r in rows if r["event"] == "done"]
            if len(ready) != 1 or len(done) != 1 or ready[0]["sha256"] != done[0]["sha256"] or not done[0]["next_gpu_operation_correct"]:
                raise RuntimeError("GPU evidence mismatch")
        else:
            before = next(r for r in rows if r["event"] == "before")
            after = next(r for r in rows if r["event"] == "after")
            if any(before[k] != after[k] for k in ("step", "token", "offset")):
                raise RuntimeError("CPU state mismatch")
            if [r["step"] for r in rows if r["event"] == "step"] != list(range(1,26)):
                raise RuntimeError("CPU continuation mismatch")
        for path in run.rglob("*"):
            # Successful numerical continuation does not excuse a warning that
            # an underlying resource was unsupported. Keep the raw logs as evidence.
            if path.is_file() and path.suffix in (".log", ".stderr"):
                content = path.read_text(errors="replace").lower()
                if "not supported" in content or "unsupported" in content:
                    raise RuntimeError(f"Unsupported-resource diagnostic in {path}")
        events.append({"event": "restored_verified", "real_pid": restored.pid})
        print(json.dumps({"passed": True, "mode": args.mode, "events": events}))
    finally:
        # Preserve events even after failure. Ask each private coordinator to
        # quit, then force any remaining child to exit and collect its status.
        (run / "controller-events.json").write_text(json.dumps(events, indent=2))
        for port in port_files:
            if port.exists():
                command([str(tool / "bin/dmtcp_command"), "--coord-port", port.read_text().strip(), "--quit"], controller, timeout=10, check=False)
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
