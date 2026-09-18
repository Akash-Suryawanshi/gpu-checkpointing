"""Pin the revision, render one prompt, and record the reference vLLM itself produces.

Named `assets` rather than `prepare`: `prepare.py` and `prepare_assets.py` already
exist in the other experiment directories, and a bare import would silently select
the wrong one. Asset validation is reused from the inference preparation.
"""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Ahead of this script's own directory, so shared module names resolve to the
# modules being reused rather than to same-named files in this directory.
sys.path[:0] = [str(ROOT / "finetuning"), str(ROOT / "inference")]
# Last, so this directory's own modules are importable when a test loads these
# files by location, without ever shadowing a shared module name.
sys.path.append(str(HERE))

from prepare import environment  # noqa: E402
from prepare_assets import PINS, validate_assets  # noqa: E402
from worker import control, render  # noqa: E402

import engine  # noqa: E402


def eos_token_ids(model, tokenizer):
    """Read the shipped stop tokens; a chat model usually stops on more than one."""
    configured = model / "generation_config.json"
    return control.read(configured)["eos_token_id"] if configured.exists() else tokenizer.eos_token_id


def main(args):
    model, output = args.model.resolve(), args.output.resolve()
    source = control.read(args.source)
    if PINS.get(source["repo"]) != source["revision"]:
        raise ValueError("Unapproved model revision")
    if output.exists():
        saved = validate_assets(output)  # Partial directories fail; never fill them in.
        if saved["model_path"] != str(model) or saved["identity"]["revision"] != source["revision"]:
            raise ValueError("Prepared assets belong to another model")
        if saved["engine"]["gpu_fraction"] != args.gpu_fraction or saved["engine"]["max_model_len"] != args.max_model_len:
            raise ValueError("Prepared assets belong to another engine configuration")
        return
    output.mkdir(mode=0o700, parents=True)
    try:
        files = sorted(p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".txt"))
        hashes = {p.name: control.file_hash(p) for p in files}
        host = environment()
        # The reference is produced by the engine under test, at the settings the
        # campaign will use: another engine's tokens are not a contract for this one.
        served, loading = engine.load(model, args.gpu_fraction, False, args.max_model_len)
        tokenizer = served.get_tokenizer()
        inputs = {"benchmark": render(tokenizer, "Explain in one sentence why the sky looks blue."),
                  "warmup": render(tokenizer, "Reply with a short greeting."), "max_new_tokens": 16,
                  "decoder": "greedy", "thinking": False, "eos_token_id": eos_token_ids(model, tokenizer)}
        control.write(output / "tokens.json", inputs)
        response = engine.generate(served, inputs["benchmark"], 16, request_id="reference")
        if len(response["tokens"]) != 16 or response["finish_reason"] != "length":
            raise ValueError("Reference did not produce the full 16-token contract")
        control.write(output / "reference.json", {**response, "first_token": response["tokens"][0]})
        control.write(output / "audit.json", engine.inspect(served, "diagnostic"))
        # Repeat the first token alone: greedy decoding must not depend on how many
        # tokens were requested, and the length stop must end exactly one token in.
        stopped = engine.generate(served, inputs["benchmark"], 1, request_id="smoke")
        if stopped["tokens"] != response["tokens"][:1] or stopped["finish_reason"] != "length":
            raise ValueError("Single-token repeat smoke check failed")
        control.write(output / "smoke.json", {"single_token": stopped["tokens"], "passed": True})
        control.write(output / "manifest.json", {"model_path": str(model), "environment": host,
            "engine": {"name": "vllm", "version": engine.version(), "dtype": "bfloat16",
                       "gpu_fraction": args.gpu_fraction, "max_model_len": args.max_model_len,
                       "load_call_seconds": loading["load_call_seconds"]},
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
    parser.add_argument("--gpu-fraction", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    main(parser.parse_args())
