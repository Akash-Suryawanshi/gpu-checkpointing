"""GPU-only suspension probe; the original CPU process remains alive throughout."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

ELEMENTS = 16 * 1024 * 1024


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def run(run_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A working NVIDIA CUDA device is required")
    tensor = torch.arange(ELEMENTS, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    before = digest(tensor)
    print(json.dumps({"event": "ready", "pid": os.getpid(), "bytes": tensor.numel() * tensor.element_size(), "sha256": before}), flush=True)
    (run_dir / "ready").touch(exist_ok=False)
    while not (run_dir / "release").exists():
        time.sleep(0.05)
    after = digest(tensor)
    if after != before:
        raise RuntimeError("GPU memory changed across suspension")
    tensor.add_(1)
    torch.cuda.synchronize()
    if tensor[0].item() != 1 or tensor[-1].item() != ELEMENTS:
        raise RuntimeError("Post-restore GPU computation returned the wrong result")
    print(json.dumps({"event": "done", "pid": os.getpid(), "sha256": after, "memory_matches": True, "next_gpu_operation_correct": True}), flush=True)
    (run_dir / "done").touch(exist_ok=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.run_dir.resolve())


if __name__ == "__main__":
    main()
