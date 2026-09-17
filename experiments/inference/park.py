"""Bounded RAM parking and resource admission; no additional supervisor."""

import json
import os
from pathlib import Path
import subprocess
import threading
import time

from worker import control
from control import session

MIB = 1024 ** 2
GIB = 1024 ** 3


def host_memory():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    limits = []
    relative = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                    if line.startswith("0::"))
    root = Path("/sys/fs/cgroup")
    current = root / relative.lstrip("/")
    # A parent slice can constrain a child whose own memory.max says 'max'.
    for directory in (current, *current.parents):
        if not directory.is_relative_to(root):
            break
        limit = directory / "memory.max"
        if limit.exists() and limit.read_text().strip() != "max":
            maximum = int(limit.read_text())
            used = int((directory / "memory.current").read_text())
            limits.append({"path": str(directory), "limit": maximum, "used": used})
            available = min(available, max(0, maximum - used))
    return {"available": available, "cgroup_limits": limits}


def resources(deadline, pid=None):
    def query(fields, subject="gpu"):
        return subprocess.check_output(["nvidia-smi", f"--query-{subject}={fields}",
            "--format=csv,noheader,nounits"], text=True, timeout=control.remaining(deadline)).strip()
    total, free, used = map(int, query("memory.total,memory.free,memory.used").split(","))
    processes = query("pid", "compute-apps")
    record = {**host_memory(), "gpu_total": total * MIB, "gpu_free": free * MIB,
              "gpu_used": used * MIB, "compute_pids": [int(p) for p in processes.splitlines() if p.strip()]}
    if pid and Path(f"/proc/{pid}/status").exists():
        fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
        record["rss_bytes"] = int(fields.get("VmRSS", "0").split()[0]) * 1024
    return record


def admission(observed, memory, action, pid=None):
    if any(p != pid for p in observed["compute_pids"]):
        raise ValueError("Foreign GPU compute process")
    reserved = memory["reserved_vram"]
    gpu_required = reserved + GIB if action == "restore" else GIB
    if observed["gpu_free"] < gpu_required:
        raise ValueError("Insufficient GPU reserve")
    required = reserved + 4 * GIB + (memory["rss_bytes"] if action == "restore" else 0)
    if action in ("park", "capture", "restore") and observed["available"] < required:
        raise ValueError(f"Insufficient host memory for {action}: need {required}, available {observed['available']}")
    return {"host_required": required, "gpu_free_required": gpu_required}


class Parking:
    def __init__(self, helper, identity, deadline, emit):
        self.helper, self.identity, self.deadline, self.emit = helper, identity, deadline, emit

    def command(self, *arguments):
        control.remaining(self.deadline)
        if not session.matches(self.identity) or self.identity["uid"] != os.getuid():
            raise ValueError("RAM worker identity changed")
        completed = subprocess.run([str(self.helper), *arguments, "--pid", str(self.identity["pid"])],
            capture_output=True, text=True, timeout=control.remaining(self.deadline))
        self.emit("cuda_helper", arguments=list(arguments), code=completed.returncode,
                  stdout=completed.stdout.strip(), stderr=completed.stderr.strip())
        completed.check_returncode()
        control.remaining(self.deadline)
        return completed.stdout.strip().lower()

    def expect(self, state):
        actual = self.command("--get-state")
        if actual != state:
            raise ValueError(f"CUDA state {actual!r}; expected {state!r}")

    def transition(self, action, before, after):
        self.expect(before)
        self.command("--action", action)
        self.expect(after)

    def park(self):
        self.transition("lock", "running", "locked")
        self.transition("checkpoint", "locked", "checkpointed")

    def wake(self):
        self.transition("restore", "checkpointed", "locked")
        self.transition("unlock", "locked", "running")


def job_size(before_free, after_free):
    large = min(12288, after_free // MIB - 2048)
    if large >= before_free // MIB + 1024:
        return large, "allocation_enabled_by_release"
    if after_free < (256 + 2048) * MIB:
        raise ValueError("Insufficient free GPU memory for availability probe")
    return 256, "availability_only"


class Sampler:
    """Separate observer clock; sample peaks are not continuous maxima."""
    def __init__(self, run, interval, deadline):
        self.run, self.interval, self.deadline = run, interval, deadline
        self.pid = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with (self.run / "resources.jsonl").open("a") as stream:
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    row = {"observer_ns": time.monotonic_ns(), **resources(self.deadline, self.pid)}
                except Exception as error:
                    row = {"observer_ns": time.monotonic_ns(), "error": str(error)}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                self.stop.wait(max(0, self.interval - (time.monotonic() - started)))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Resource sampler did not stop")
