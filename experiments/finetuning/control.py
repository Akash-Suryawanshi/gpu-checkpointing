"""File protocol shared by independently launched trainers and workers.

Records carry identity and permission, never replacement training state. See
../../docs/independent-lifecycle-details.md for the persistence ordering.
"""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

from prepare import file_hash

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/criu"))
import session

SCHEMA = 1


def read(path):
    return json.loads(Path(path).read_text())


def sync_directory(path):
    with directory_fd(path) as fd:
        os.fsync(fd)


@contextmanager
def directory_fd(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield fd
    finally:
        os.close(fd)


def write(path, value):
    """Persist a record and its directory entry before returning to the caller."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


@contextmanager
def lock(run, deadline=None):
    """The kernel releases flock when a worker dies; phase evidence still remains."""
    directory = Path(run) / "control"
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / "lock").open("a") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if deadline is None:
                    raise RuntimeError("Another operation owns this job") from error
                time.sleep(min(0.05, remaining(deadline)))
        yield


def monotonic_offset():
    row = next(line.split() for line in Path("/proc/self/timens_offsets").read_text().splitlines()
               if line.startswith("monotonic"))
    return int(row[1]) + int(row[2]) / 1e9


def request_expired(request):
    # Request deadlines use the underlying host clock. A reconstructed trainer
    # may have a different time-namespace offset from the next capture worker.
    return time.monotonic() - monotonic_offset() > request["deadline"]


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Operation deadline expired")
    return value


def wait(path, deadline, identity=None, poll=0.05):
    while not Path(path).exists():
        remaining(deadline)
        if identity is not None and not session.matches(identity):
            raise RuntimeError("Identified trainer exited while waiting")
        time.sleep(min(poll, remaining(deadline)))
    remaining(deadline)
    return read(path)


def phase(run, status, **values):
    write(Path(run) / "control/phase.json", {"status": status, **values})


def event(attempt, name, **values):
    row = {"event": name, "monotonic_ns": time.monotonic_ns(), **values}
    with (attempt / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
    return row


def attempt(run):
    identifier = uuid.uuid4().hex
    path = Path(run) / "attempts" / identifier
    path.mkdir(mode=0o700, parents=True)
    sync_directory(path.parent)
    sync_directory(run)
    return identifier, path


def register(run, assets, until):
    """Register once before initialization so a fast workload cannot miss a request."""
    run = Path(run)
    (run / "control").mkdir(mode=0o700, exist_ok=True)
    with lock(run):
        if (run / "job.json").exists():
            raise ValueError("Job directory already registered")
        job = {"schema": SCHEMA, "job_id": uuid.uuid4().hex,
               "run": str(run), "assets": str(Path(assets).resolve()), "until": until,
               "python": sys.executable, "identity": session.identity(os.getpid())}
        write(run / "job.json", job)
        phase(run, "idle")
        for directory in run.parents:
            sync_directory(directory)
    return job


def boundary(run, job, update, inspect):
    """Acknowledge at a complete update and wait without fetching data or RNG.

    Before dump starts, expiry/cancellation releases this request. Once armed,
    unknown CPU/CUDA state is never resumed automatically after worker failure.
    """
    request_path = run / "control/request.json"
    if not request_path.exists():
        return
    request = read(request_path)
    if request["job_id"] != job["job_id"] or update < request["at"]:
        return
    capture = request["capture_id"]
    folder = run / "attempts" / capture
    current = read(run / "control/phase.json")
    if current.get("capture_id") != capture or current["status"] != "requested":
        return
    if request_expired(request) or update >= job["until"]:
        write(folder / "rejected.json", {"reason": "expired or final boundary"})
        return
    before = inspect()
    write(folder / "before.json", before)
    write(folder / "ready.json", {"capture_id": capture, "job_id": job["job_id"],
                                 "update": update, "identity": job["identity"]})
    while True:
        current = read(run / "control/phase.json")
        if current.get("capture_id") != capture:
            raise RuntimeError("Pause ownership changed")
        if current["status"] == "cancelled":
            return
        if current["status"] == "requested" and request_expired(request):
            # Expiry must not race a live worker changing requested -> dumping.
            # Only an unowned, still-unarmed request can release itself safely.
            try:
                with lock(run):
                    if read(run / "control/phase.json") == current:
                        return
            except RuntimeError:  # A worker still owns the operation.
                pass
        signal = folder / "inspect.json"
        if signal.exists():
            restoration = read(signal)
            if restoration["capture_id"] != capture:
                raise ValueError("Stale inspection request")
            restored = run / "attempts" / restoration["attempt_id"]
            refreshed = read(run / "job.json")
            if refreshed["job_id"] != job["job_id"] or refreshed["identity"]["pid"] != os.getpid():
                raise ValueError("Restored registration belongs to another job")
            # Start ticks in /proc are relative to the observer's clock namespace.
            # The host worker owns identity; do not overwrite it from this restored namespace.
            job.update(refreshed)
            write(restored / "after.json", inspect())
            write(restored / "inspected.json", {"attempt_id": restoration["attempt_id"]})
            # No deadline from the old clock domain is used after reconstruction.
            while not (restored / "continue.json").exists():
                time.sleep(0.05)
            if read(restored / "continue.json") != restoration:
                raise ValueError("Wrong continuation identity")
            return
        time.sleep(0.05)


def tools_config(root):
    root = Path(root).resolve()
    return {"criu": str(root / "criu-4.2.1/criu/criu"),
            "plugin": str(root / "criu-4.2.1/plugins/cuda"),
            "libraries": str(root / "criu-deps/usr/lib/x86_64-linux-gnu")}


def runtime_libraries(directory):
    # Extracted development packages can contain dangling unversioned linker
    # symlinks; fingerprint the usable runtime files, not unavailable build inputs.
    return [path for path in directory.glob("*.so*") if path.is_file()]


def hash_files(paths, deadline, workers=1):
    if workers not in (1, 4):
        raise ValueError("Validation workers must be 1 or 4")
    def checked(path):
        remaining(deadline)
        digest = file_hash(path)
        remaining(deadline)
        return str(path.absolute()), digest
    # Each independent file retains its ordinary SHA-256. At most four
    # bounded read buffers exist; changing concurrency never changes identity.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(checked, paths))


def dependencies(job, tools, deadline, workers=1):
    """Fingerprint dependencies without importing torch or creating a CUDA context."""
    assets = Path(job["assets"])
    manifest = read(assets / "manifest.json")
    paths = list((ROOT / "experiments/finetuning").glob("*.py"))
    paths += list((ROOT / "experiments/inference").glob("*.py"))
    paths += [ROOT / "experiments/criu/session.py", assets / "manifest.json", assets / "tokens.json"]
    paths += [Path(manifest["model_path"]) / name for name in manifest["identity"]["files"]]
    paths += [Path(tools) / name for name in (
        "criu-4.2.1/criu/criu", "criu-4.2.1/plugins/cuda/cuda_plugin.so",
        "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint")]
    paths += runtime_libraries(Path(tools) / "criu-deps/usr/lib/x86_64-linux-gnu")
    hashes = hash_files(paths, deadline, workers)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                   "--format=csv,noheader"], text=True, timeout=remaining(deadline)).strip()
    return {"files": hashes, "python": platform.python_version(), "executable": job["python"],
            "packages": dict(sorted((d.metadata["Name"].lower(), d.version)
                                     for d in importlib.metadata.distributions())),
            "kernel": platform.release(), "gpu": gpu,
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def storage(path):
    """Record mount/backing device and reject volatile memory filesystems."""
    data = json.loads(subprocess.check_output(
        ["findmnt", "-J", "-T", str(path), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"], text=True))
    mount = data["filesystems"][0]
    if mount["fstype"] in ("tmpfs", "ramfs"):
        raise ValueError("Snapshot storage must not be tmpfs/ramfs")
    usage = shutil.disk_usage(path)
    return {**mount, "total": usage.total, "free": usage.free}


def inventory(snapshot):
    """Only regular payload files are admitted; logs belong in attempts instead."""
    paths = [snapshot / "before.json", *sorted((snapshot / "images").rglob("*"))]
    result = {}
    for path in paths:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Nonregular snapshot payload")
        if path.is_file():
            result[str(path.relative_to(snapshot))] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    if not all(name in result for name in ("before.json", "images/inventory.img", "images/pstree.img")):
        raise ValueError("Incomplete payload inventory")
    return result


def publish(snapshot, manifest, run, deadline, emit):
    """Hash, sync payload, persist manifest/ancestors, then completion and phase."""
    try:
        manifest["payload"] = inventory(snapshot)
        emit("payload_hash_completed")
        for name in manifest["payload"]:
            remaining(deadline)
            with (snapshot / name).open("rb") as stream:
                os.fsync(stream.fileno())
        for directory in sorted((p for p in (snapshot / "images").rglob("*") if p.is_dir()),
                                key=lambda p: len(p.parts), reverse=True):
            sync_directory(directory)
        sync_directory(snapshot / "images")
        emit("payload_synced")
        write(snapshot / "manifest.json", manifest)
        for directory in (snapshot, *snapshot.parents):
            remaining(deadline)
            sync_directory(directory)
        write(snapshot / "COMPLETE", {"capture_id": manifest["capture_id"],
                                      "manifest_sha256": file_hash(snapshot / "manifest.json")})
        remaining(deadline)
        phase(run, "published", capture_id=manifest["capture_id"], snapshot=str(snapshot))
        remaining(deadline)
        emit("local_snapshot_published")
    except BaseException:
        # Visible COMPLETE without a durable terminal phase is never eligible.
        try:
            (snapshot / "COMPLETE").unlink(missing_ok=True)
            sync_directory(snapshot)
            phase(run, "failed", capture_id=manifest["capture_id"], snapshot=str(snapshot))
        except OSError:
            pass  # A failing disk may also prevent recording the failure.
        raise


def validate(snapshot, run, tools, deadline, order="payload-first", emit=lambda name: None, workers=1):
    if order not in ("payload-first", "dependencies-first"):
        raise ValueError("Unknown validation order")
    manifest = read(snapshot / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest["run"] != str(run):
        raise ValueError("Wrong snapshot schema or job path")
    complete = read(snapshot / "COMPLETE")
    if complete != {"capture_id": manifest["capture_id"], "manifest_sha256": file_hash(snapshot / "manifest.json")}:
        raise ValueError("Completion digest mismatch")
    expected_phase = {"status": "published", "capture_id": manifest["capture_id"], "snapshot": str(snapshot)}
    if read(run / "control/phase.json") != expected_phase:
        raise ValueError("Snapshot is ambiguous, consumed, or no longer current")
    def payload():
        emit("payload_validation_started")
        if inventory(snapshot) != manifest["payload"]:
            raise ValueError("Snapshot payload mismatch")
        remaining(deadline)
        emit("payload_validation_completed")
    if order == "payload-first":
        payload()
    job = read(run / "job.json")
    if job != manifest["job"] or job["identity"]["uid"] != os.getuid():
        raise ValueError("Job identity mismatch")
    if Path(f'/proc/{job["identity"]["pid"]}').exists():
        raise ValueError("Original PID is still present")
    request = read(run / "control/request.json")
    if request != manifest["request"] or (run / "attempts" / manifest["capture_id"] / "inspect.json").exists():
        raise ValueError("Stale control markers")
    emit("dependency_validation_started")
    if dependencies(job, tools, deadline, workers) != manifest["dependencies"]:
        raise ValueError("Environment, tools, source, or asset mismatch")
    emit("dependency_validation_completed")
    for name in ("updates.jsonl", "trainer.stderr"):
        if not (run / name).is_file() or file_hash(run / name) != manifest["external_files"][name]:
            raise ValueError("Required external log missing or changed")
    # Dependencies can exceed the page cache: read the image last so CRIU can
    # reuse verified pages. Both orders retain every original integrity check.
    if order == "dependencies-first":
        payload()
    return manifest
