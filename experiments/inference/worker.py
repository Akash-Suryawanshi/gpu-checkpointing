"""Shared greedy decoder and an independently supervised inference worker."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))
import control


def load(model_path, loader="transformers", bundle=None, stats=None, compile_mode=None):
    # Direct preparation and isolated children must use the same cuBLAS policy.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if loader != "transformers":
        from stream_weights import load as stream_load
        return stream_load(model_path, Path(bundle), pipelined=loader in ("pipelined", "direct"),
                           direct=loader == "direct", stats=stats)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
    if compile_mode:
        # Compiled kernels live in this process. A fresh start must produce them;
        # a snapshot restores them, which is the state worth capturing.
        model = torch.compile(model, mode=compile_mode)
    torch.cuda.synchronize()
    return model, tokenizer


def render(tokenizer, prompt):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return {"prompt": prompt, "rendered": text, "input_ids": tokenizer(text).input_ids}


def generate(model, tokenizer, inputs, count, first=None, on_token=None):
    import torch
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    generated, cache = [], None
    with torch.inference_mode():
        ids = torch.tensor([inputs["input_ids"]], device="cuda")
        # Cache is local: neither warmup nor a prior request enters an idle image.
        for index in range(count):
            output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            token = int(output.logits[0, -1].argmax().item())
            generated.append(token)
            if index == 0 and first is not None:
                first(token)
            if on_token is not None:
                on_token(index, token)
            if token in eos:
                break
            cache = output.past_key_values
            ids = torch.tensor([[token]], device="cuda")
        torch.cuda.synchronize()
    return {"tokens": generated, "text": tokenizer.decode(generated, skip_special_tokens=True)}


def inspect(model, kind):
    import state
    # A compiled module wraps the original; its parameters are the same tensors.
    model = getattr(model, "_orig_mod", model)
    tensors = model.state_dict()
    if any(m.training for m in model.modules()) or any(t.device.type != "cuda" for t in tensors.values()):
        raise ValueError("Model must be entirely on GPU in evaluation mode")
    record = {"training": False, "kv_cache": None,
              "tensors": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "device": str(v.device)}
                          for k, v in tensors.items()}}
    if kind == "diagnostic":
        record["fingerprints"] = state.fingerprints(tensors)
    return record


def memory():
    import torch
    fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    return {"VmRSS": fields["VmRSS"].strip(), "rss_bytes": int(fields["VmRSS"].split()[0]) * 1024,
            "reserved_vram": torch.cuda.memory_reserved(), "allocated_vram": torch.cuda.memory_allocated()}


def attempt_paths(run, record, restoration):
    """Bind a restored worker to the attempt that released it; reject any other path."""
    if record is None or restoration is None:
        raise ValueError("Restored worker has no activation record")
    if any(record[key] != restoration[key] for key in ("attempt_id", "capture_id")):
        raise ValueError("Activation record names another attempt")
    root = (Path(run) / "attempts" / record["attempt_id"]).resolve()
    requests = (Path(run) / record["requests"]).resolve()
    if not requests.is_relative_to(root) or not requests.is_dir():
        raise ValueError("Request path escapes the attempt")
    return root, requests


def redirect_output(path):
    """Move stdout/stderr off the captured log files, which stay at their baseline content."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


def next_request(requests, index, poll):
    """Return the next request, or None once the owner writes stop.json."""
    while True:
        path = requests / f"request-{index}.json"
        if path.exists():
            return control.read(path)
        if (requests / "stop.json").exists():
            return None
        time.sleep(poll)


def serve(requests, run_id, model, tokenizer, inputs, poll):
    control.write(requests / "ready.json", {"run_id": run_id})
    index = 0
    while (request := next_request(requests, index := index + 1, poll)) is not None:
        if request["run_id"] != run_id:
            raise ValueError("Request identity mismatch")
        identity = {"run_id": run_id, "request_id": request["request_id"]}
        prompt = render(tokenizer, request["prompt"]) if "prompt" in request else {"input_ids": request["input_ids"]}
        count = request.get("max_new_tokens", inputs["max_new_tokens"])
        with (requests / f"tokens-{index}.jsonl").open("a") as stream:
            def on_token(position, token):
                # Flush each token to the page cache immediately; readers tail this file.
                stream.write(json.dumps({**identity, "index": position, "token_id": token,
                                         "text": tokenizer.decode([token])}) + "\n")
                stream.flush()
                if position == 0:
                    control.write(requests / f"first-{index}.json", {**identity, "token": token})
            response = generate(model, tokenizer, prompt, count, on_token=on_token)
        control.write(requests / f"response-{index}.json", {**identity, **response})
    # Fingerprinting every weight takes seconds during which this worker reads no
    # request. Do it once the owner has stopped sending, never between responses.
    control.write(requests / "audit-after.json", inspect(model, "diagnostic"))
    control.write(requests / "completed.json", memory())


def main(args):
    import torch
    run = args.run_dir.resolve()
    os.umask(0o077)
    job = control.register(run, args.assets, 2) if args.route == "disk" else None
    manifest = control.read(args.assets / "manifest.json")
    inputs = control.read(args.assets / "tokens.json")
    settings = control.read(run / "run.json")["key"]
    bundle = settings.get("bundle")
    stats, load_started = {}, time.monotonic()
    model, tokenizer = load(manifest["model_path"], settings.get("loader", "transformers"),
                            bundle["path"] if bundle else None, stats, settings.get("compile_mode"))
    control.write(run / "loading.json", {**stats, "load_call_seconds": time.monotonic() - load_started})
    poll = settings["poll_ms"] / 1000
    wait = lambda path: control.wait(path, time.monotonic() + 7200, poll=poll)
    if args.route != "fresh":
        generate(model, tokenizer, inputs["warmup"], 4)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        control.write(run / "audit-before.json", inspect(model, "diagnostic"))
    control.write(run / "memory.json", memory())
    control.write(run / "idle.json", {"run_id": args.run_id, "kv_cache": None})
    requests = run
    if args.route == "disk":
        wait(run / "control/request.json")
        restoration = control.boundary(run, job, 1, lambda: inspect(model, args.kind))
        record = control.activation(run)
        if record is not None:
            # Reusable image: serve only the attempt that released this process.
            root, requests = attempt_paths(run, record, restoration)
            redirect_output(root / "worker.log")
            control.write(root / "acknowledged.json", {**restoration, "pid": os.getpid()})
        elif control.read(run / "control/phase.json")["status"] not in ("verified", "restored"):
            raise RuntimeError("Capture did not produce a verified restore")
    elif args.route == "ram":
        # Only CPU file polling is permitted between idle publication and wake.
        wait(run / "wake.json")
        control.write(run / "wake-inspection.json", inspect(model, args.kind))
    serve(requests, args.run_id, model, tokenizer, inputs, poll)
    time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route", choices=("fresh", "resident", "ram", "disk"), required=True)
    parser.add_argument("--kind", choices=("diagnostic", "timing"), required=True)
    parser.add_argument("--run-id", required=True)
    main(parser.parse_args())
