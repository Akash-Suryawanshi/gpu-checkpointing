"""Kernel-accounted device reads around an activation; hashing bytes are logical."""

import json
from pathlib import Path
import subprocess
import time

SECTOR = 512  # /proc/diskstats always counts 512-byte sectors, whatever the device reports.


def device(path):
    """Resolve the block device (major:minor) backing a mounted path."""
    data = json.loads(subprocess.check_output(
        ["findmnt", "-J", "-T", str(path), "-o", "SOURCE,MAJ:MIN"], text=True))
    mount = data["filesystems"][0]
    return {"source": mount["source"], "maj_min": mount["maj:min"]}


def read_bytes(maj_min, text=None):
    text = Path("/proc/diskstats").read_text() if text is None else text
    major, minor = maj_min.split(":")
    for line in text.splitlines():
        fields = line.split()
        if fields[:2] == [major, minor]:
            return int(fields[5]) * SECTOR
    raise ValueError(f"Device {maj_min} absent from diskstats")


def sample(paths):
    """One row per distinct device; other readers of the device are not separated."""
    rows, seen = [], set()
    for path in paths:
        found = device(path)
        if found["maj_min"] in seen:
            continue
        seen.add(found["maj_min"])
        rows.append({**found, "read_bytes": read_bytes(found["maj_min"]), "monotonic_ns": time.monotonic_ns()})
    return rows


def delta(before, after):
    if [row["maj_min"] for row in before] != [row["maj_min"] for row in after]:
        raise ValueError("Device set changed between samples")
    result = []
    for start, end in zip(before, after):
        change = end["read_bytes"] - start["read_bytes"]
        if change < 0 or end["monotonic_ns"] < start["monotonic_ns"]:
            raise ValueError("Device counters moved backwards")
        result.append({"source": start["source"], "maj_min": start["maj_min"], "read_bytes": change,
                       "seconds": (end["monotonic_ns"] - start["monotonic_ns"]) / 1e9})
    return result
