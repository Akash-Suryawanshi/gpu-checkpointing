"""Prove that a separate job can use the GPU while the trainer is absent."""

import torch


if __name__ == "__main__":
    # 1,048,576 float32 values occupy 4 MiB of device memory. This is a small
    # availability check, not a claim that every larger workload will fit.
    x = torch.ones(1024 * 1024, device="cuda")
    # sum() computes on the GPU; item() waits for and copies the scalar to the
    # CPU. Checking it proves useful work completed after the allocation.
    s = x.sum().item()
    assert s == 1048576
    print(s)
