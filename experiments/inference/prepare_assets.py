"""Validate pinned local files, smoke-test inference, publish complete assets last."""

import argparse
from pathlib import Path
import sys

from worker import control, load, render, generate, inspect
from prepare import environment

PINS = {"Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
        "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987"}


def validate_assets(assets):
    manifest = control.read(assets / "manifest.json")
    for name, digest in manifest["identity"]["files"].items():
        if control.file_hash(Path(manifest["model_path"]) / name) != digest:
            raise ValueError("Model file changed: " + name)
    for name, digest in manifest["artifacts"].items():
        if control.file_hash(assets / name) != digest:
            raise ValueError("Prepared evidence changed: " + name)
    return manifest


def main(args):
    model, output = args.model.resolve(), args.output.resolve()
    source = control.read(args.source)
    if PINS.get(source["repo"]) != source["revision"]:
        raise ValueError("Unapproved model revision")
    if output.exists():
        saved = validate_assets(output)  # Partial directories fail; never fill them in.
        if saved["model_path"] != str(model) or saved["identity"]["revision"] != source["revision"]:
            raise ValueError("Prepared assets belong to another model")
        return
    output.mkdir(mode=0o700, parents=True)
    try:
        required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        index = model / "model.safetensors.index.json"
        required += sorted(set(control.read(index)["weight_map"].values())) if index.exists() else ["model.safetensors"]
        if any(not (model / name).is_file() for name in required):
            raise ValueError("Incomplete model shards or tokenizer files")
        files = sorted(p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".txt"))
        hashes = {p.name: control.file_hash(p) for p in files}
        env = environment()
        network, tokenizer = load(model)
        inputs = {"benchmark": render(tokenizer, "Explain in one sentence why the sky looks blue."),
                  "warmup": render(tokenizer, "Reply with a short greeting."), "max_new_tokens": 16,
                  "decoder": "greedy", "thinking": False, "eos_token_id": network.generation_config.eos_token_id}
        control.write(output / "tokens.json", inputs)
        first = []
        response = generate(network, tokenizer, inputs["benchmark"], 16, first.append)
        control.write(output / "reference.json", {**response, "first_token": first[0]})
        control.write(output / "audit.json", inspect(network, "diagnostic"))
        # Force an EOS after token one to exercise the actual decoder's early stop.
        old = network.generation_config.eos_token_id
        network.generation_config.eos_token_id = first[0]
        stopped = generate(network, tokenizer, inputs["benchmark"], 16)
        network.generation_config.eos_token_id = old
        if stopped["tokens"] != first:
            raise ValueError("EOS early-stop smoke check failed")
        control.write(output / "smoke.json", {"forced_eos_tokens": stopped["tokens"], "passed": True})
        control.write(output / "manifest.json", {"model_path": str(model), "environment": env,
            "config": {"dtype": "bfloat16", "attention": "sdpa", "batch_size": 1},
            "identity": {"model": source["repo"], "revision": source["revision"], "files": hashes},
            "artifacts": {name: control.file_hash(output / name) for name in
                          ("tokens.json", "reference.json", "audit.json", "smoke.json")}})
    except BaseException as error:
        control.write(output / "failure.json", {"reason": str(error), "python": sys.executable})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
