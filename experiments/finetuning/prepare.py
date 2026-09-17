"""Prepare pinned inputs outside measured training trials.

Model + examples -> tokens + manifest (file identities and execution environment).
Comparing manifests prevents changed inputs from appearing as restore failures.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
# Download only required model/tokenizer files at the pinned revision.
MODEL_FILES = (
    "config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
)


def file_hash(path):
    """Identify file contents with SHA-256 using bounded 4 MiB reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Write sorted evidence JSON; durable checkpoints use state.save_application."""
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def environment():
    """Require isolated Python and CUDA; identify the host and installed packages.

    CUDA runtime and driver versions can differ; record both for compatibility.
    """
    import torch

    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the isolated training interpreter")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this execution context")
    return {
        "python": platform.python_version(),
        "packages": dict(sorted((d.metadata["Name"].lower(), d.version)
                                for d in importlib.metadata.distributions())),
        "kernel": platform.release(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "allocator": torch.cuda.memory.get_allocator_backend(),
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ], text=True).strip(),
    }


def main():
    """Acquire pinned assets, build answer-only labels, then publish their manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, help="Reuse previously downloaded pinned model files")
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True)  # Reject existing inputs rather than overwrite them.
    config = json.loads((HERE / "config.json").read_text())
    if args.model_dir:
        # Check the cache's declared revision, then hash its actual files below.
        model = args.model_dir.resolve()
        source = json.loads((model / "asset-manifest.json").read_text())
        if source != {"repo": config["model"], "revision": config["revision"]}:
            raise ValueError("Cached model revision differs from config")
    else:
        from huggingface_hub import snapshot_download

        model = args.output.resolve() / "model"
        snapshot_download(
            config["model"], revision=config["revision"], token=False,
            local_dir=model, allow_patterns=list(MODEL_FILES),
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokens = []
    for line in (HERE / "examples.jsonl").read_text().splitlines():
        example = json.loads(line)
        # Append an end-of-sequence token. The model shifts labels to predict the next token.
        prompt = tokenizer.encode(example["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(example["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        # -100 excludes prompt tokens from the loss. Keep input/label lengths equal
        # and require an answer target to survive truncation.
        labels = ([-100] * len(prompt) + answer)[:config["max_tokens"]]
        if not any(label != -100 for label in labels):
            raise ValueError("Truncation removed every answer label")
        tokens.append({"input_ids": (prompt + answer)[:config["max_tokens"]], "labels": labels})
    write_json(args.output / "tokens.json", tokens)

    # Hash both source text and tokenized sequences; training uses the latter.
    manifest = {
        "model_path": str(model),
        "config": config,
        "environment": environment(),
        "identity": {
            "model": config["model"],
            "revision": config["revision"],
            "files": {name: file_hash(model / name) for name in MODEL_FILES},
            "examples_sha256": file_hash(HERE / "examples.jsonl"),
            "tokens_sha256": file_hash(args.output / "tokens.json"),
        },
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({
        "prepared": str(args.output),
        "examples": len(tokens),
        "max_tokens": max(len(row["input_ids"]) for row in tokens),
    }))


if __name__ == "__main__":
    main()
