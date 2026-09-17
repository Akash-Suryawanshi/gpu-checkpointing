"""File protocol shared by independently launched trainers and workers.

Records carry identity and permission, never replacement training state. See
../../docs/independent-lifecycle-details.md for the persistence ordering.
"""

from contextlib import contextmanager
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


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Operation deadline expired")
    return value


def wait(path, deadline, identity=None):
    while not Path(path).exists():
        remaining(deadline)
        if identity is not None and not session.matches(identity):
            raise RuntimeError("Identified trainer exited while waiting")
        time.sleep(min(0.05, remaining(deadline)))
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


