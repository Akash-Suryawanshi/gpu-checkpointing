"""Evict only this trial's disk data and verify residency without reading it."""

import ctypes
import mmap
import os
from pathlib import Path
import time


def resident_pages(fd, size):
    if not size:
        return 0
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vector = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    # ACCESS_COPY permits obtaining an address without touching the mapping;
    # mincore reports file-cache residency, not Python's virtual mappings.
    with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vector):
            raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(value & 1 for value in vector)


def drop(path):
    """Advise one file out of the page cache and report residency either side."""
    with path.open("rb") as stream:
        fd, size = stream.fileno(), path.stat().st_size
        before = resident_pages(fd, size)
        # Dirty pages cannot be evicted. Do not drop global caches or touch
        # other workloads: sync and advise only explicitly selected files.
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return {"path": str(path), "bytes": size,
                "resident_before": before, "resident_after": resident_pages(fd, size)}


def evict(paths, attempts=5, delay=0.5):
    """Leave every selected file with no resident pages, or say exactly which kept them.

    A concurrent reader or writeback can repopulate a file between the advice and
    the check. Retrying makes the guarantee stricter, never weaker: the returned
    records still have to show zero resident pages everywhere.
    """
    selected = sorted(set(map(Path, paths)))
    for attempt in range(attempts):
        if attempt:
            time.sleep(delay)
        records = [drop(path) for path in selected]
        remaining = [row for row in records if row["resident_after"]]
        if not remaining:
            return records
    detail = ", ".join(f"{Path(row['path']).name} kept {row['resident_after']} of "
                       f"{(row['bytes'] + mmap.PAGESIZE - 1) // mmap.PAGESIZE} pages"
                       for row in remaining)
    raise RuntimeError(f"Selected files are still cached after {attempts} attempts; "
                       f"cannot claim a cold start: {detail}")
