"""Bounded ServerlessLLM-inspired bulk loading, not a process snapshot.

One flat BF16 allocation receives verified chunks through two reusable pinned
buffers. CUDA events prevent overwriting a buffer while its DMA is in flight.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

CHUNK = 16 * 1024 * 1024


def metadata(folder):
    record = json.loads((folder / "manifest.json").read_text())
    if record["schema"] != 1 or record["chunk_bytes"] != CHUNK:
        raise ValueError("Unsupported packed weights")
    offset = 0
    for item in sorted(record["tensors"].values(), key=lambda item: item["offset"]):
        if (item["dtype"] != "BF16" or any(type(n) is not int or n <= 0 for n in item["shape"])
                or item["offset"] != offset or item["bytes"] != math.prod(item["shape"]) * 2):
            raise ValueError("Invalid tensor layout")
        offset += item["bytes"]
    if (offset != record["bytes"] or (folder / "weights.bin").stat().st_size != offset
            or len(record["chunks"]) != math.ceil(offset / CHUNK)):
        raise ValueError("Incomplete packed weights")
    return record


def verified_read(stream, target, expected):
    position = 0
    while position < len(target):
        count = stream.readinto(target[position:])
        if not count:
            raise ValueError("Truncated weight chunk")
        position += count
    if hashlib.sha256(target).hexdigest() != expected:
        raise ValueError("Weight chunk digest mismatch")


def pack(assets, output):
    from prepare_assets import validate_assets
    from worker import control
    manifest = validate_assets(assets)
    model = Path(manifest["model_path"])
    output.mkdir(parents=True)
    tensors, offset = {}, 0
    with (output / "weights.bin").open("xb") as destination:
        for shard in sorted(model.glob("*.safetensors")):
            with shard.open("rb") as source:
                header = json.loads(source.read(int.from_bytes(source.read(8), "little")))
                entries = sorted(((name, value) for name, value in header.items() if name != "__metadata__"),
                                 key=lambda item: item[1]["data_offsets"][0])
                previous = 0
                for name, value in entries:
                    start, end = value["data_offsets"]
                    if name in tensors or start != previous or value["dtype"] != "BF16":
                        raise ValueError("Requires dense, distinct BF16 source tensors")
                    tensors[name] = {"dtype": "BF16", "shape": value["shape"],
                                     "offset": offset + start, "bytes": end - start}
                    previous = end
                copied = 0
                while chunk := source.read(CHUNK):
                    destination.write(chunk)
                    copied += len(chunk)
                if copied != previous:
                    raise ValueError("Source shard length mismatch")
                offset += copied
        destination.flush()
        os.fsync(destination.fileno())
    chunks = []
    with (output / "weights.bin").open("rb") as stream:
        while chunk := stream.read(CHUNK):
            chunks.append(hashlib.sha256(chunk).hexdigest())
    # Dict insertion order is the flat layout; sorted JSON would destroy it.
    # Store offsets and re-order explicitly when reading the manifest instead.
    control.write(output / "manifest.json", {"schema": 1, "bytes": offset, "chunk_bytes": CHUNK,
        "tensors": tensors, "chunks": chunks, "assets_sha256": control.file_hash(assets / "manifest.json")})


def load(model_path, folder, pipelined=True):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    record = metadata(folder)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    # Model construction defines shapes without randomizing billions of weights.
    # Nonpersistent buffers (e.g. rotary frequencies) are initialized normally.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    if set(model.state_dict()) != set(record["tensors"]):
        raise ValueError("Packed tensors do not match model architecture")
    flat = torch.empty(record["bytes"], dtype=torch.uint8, device="cuda:0")
    buffers = [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=pipelined) for _ in range(2 if pipelined else 1)]
    events = [torch.cuda.Event() for _ in buffers]
    used = [False] * len(buffers)
    with (folder / "weights.bin").open("rb", buffering=0) as stream:
        for index, expected in enumerate(record["chunks"]):
            slot = index % len(buffers)
            if used[slot]:
                events[slot].synchronize()
            offset = index * CHUNK
            size = min(CHUNK, record["bytes"] - offset)
            verified_read(stream, memoryview(buffers[slot].numpy())[:size], expected)
            flat[offset:offset + size].copy_(buffers[slot][:size], non_blocking=pipelined)
            events[slot].record()
            used[slot] = True
    torch.cuda.synchronize()
    state = {name: flat[item["offset"]:item["offset"] + item["bytes"]].view(torch.bfloat16).view(item["shape"])
             for name, item in record["tensors"].items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.to("cuda:0").eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack(args.assets.resolve(), args.output.resolve())
