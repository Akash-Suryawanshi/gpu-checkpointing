"""Small tensor workload for GPU suspension and DMTCP process-restore probes."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

ELEMENTS = 16 * 1024 * 1024


def digest(tensor: torch.Tensor) -> str:
    """Copy tensor bytes to CPU RAM and hash them for before/after comparison."""
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def run(run_dir: Path) -> None:
    """Check one tensor survives NVIDIA suspension and accepts a later operation.

    This workload is used both by the GPU-only shell probe (CPU process stays
    alive) and by the DMTCP probe (original exits). The controller determines
    which lifecycle was exercised; matching tensor data alone cannot do that.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("A working NVIDIA CUDA device is required")
    tensor = torch.arange(ELEMENTS, dtype=torch.float32, device="cuda")
    # CUDA normally queues work and lets Python proceed. Wait for initialization
    # before advertising readiness to an external snapshot controller.
    torch.cuda.synchronize()
    before = digest(tensor)
    print(json.dumps({"event": "ready", "pid": os.getpid(), "bytes": tensor.numel() * tensor.element_size(), "sha256": before}), flush=True)
    (run_dir / "ready").touch(exist_ok=False)
    # Marker files are control messages, not saved tensor contents. Wait on the
    # CPU without issuing more GPU operations until the controller permits them.
    while not (run_dir / "release").exists():
        time.sleep(0.05)
    after = digest(tensor)
    if after != before:
        raise RuntimeError("GPU memory changed across suspension")
    # Equal bytes alone are insufficient: a fresh calculation checks that the
    # restored CUDA resources can execute work as well as expose old data.
    tensor.add_(1)
    torch.cuda.synchronize()
    if tensor[0].item() != 1 or tensor[-1].item() != ELEMENTS:
        raise RuntimeError("Post-restore GPU computation returned the wrong result")
    print(json.dumps({"event": "done", "pid": os.getpid(), "sha256": after, "memory_matches": True, "next_gpu_operation_correct": True}), flush=True)
    (run_dir / "done").touch(exist_ok=False)


def main() -> None:
    """Use an absolute run path for the workload's readiness and release markers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.run_dir.resolve())


if __name__ == "__main__":
    main()
