"""Evict only this trial's disk data and verify residency without reading it."""

import ctypes
import mmap
import os
from pathlib import Path


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


def evict(paths):
    records = []
    for path in sorted(set(map(Path, paths))):
        with path.open("rb") as stream:
            fd, size = stream.fileno(), path.stat().st_size
            before = resident_pages(fd, size)
            # Dirty pages cannot be evicted. Do not drop global caches or touch
            # other workloads: sync and advise only explicitly selected files.
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            after = resident_pages(fd, size)
            records.append({"path": str(path), "bytes": size,
                            "resident_before": before, "resident_after": after})
    if any(row["resident_after"] for row in records):
        raise RuntimeError("Selected files are still cached; cannot claim a cold start")
    return records
