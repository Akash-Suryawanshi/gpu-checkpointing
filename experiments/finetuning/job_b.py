"""Prove a separate process can allocate, touch, and compute on the released GPU."""

import argparse
import json
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, default=4)
    args = parser.parse_args()
    if args.mib <= 0:
        parser.error("--mib must be positive")
    # Touch every byte. Integer reduction avoids float32 summation rounding for
    # the larger probe; slicing into chunks avoids an equally large int64 copy.
    x = torch.ones(args.mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
    total = sum(int(chunk.sum().item()) for chunk in x.split(4 * 1024 * 1024))
    if total != x.numel():
        raise RuntimeError("Job B calculation mismatch")
    print(json.dumps({"mib": args.mib, "sum": total, "passed": True}))
