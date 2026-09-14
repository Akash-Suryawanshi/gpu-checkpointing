"""One explicit LoRA update loop, with observable completed-update waiting points."""

import argparse
import json
import os
from pathlib import Path
import random
import time

from prepare import environment, file_hash, write_json


def memory():
    status = Path("/proc/self/status").read_text().splitlines()
    return {row.split(":")[0]: row.split(":")[1].strip() for row in status
            if row.startswith(("VmRSS:", "VmHWM:"))}


def wait(path):
    # The external controller owns deadlines: saved wall-clock deadlines age while off-GPU.
    while not path.exists():
        time.sleep(0.05)


def main(args):
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)

    def mappings(stage):
        (run / f"{stage}.maps").write_text(Path("/proc/self/maps").read_text())

    mappings("python")
    import torch
    import state
    mappings("torch")
    manifest = json.loads((args.assets / "manifest.json").read_text())
    state.compare(manifest["environment"], environment(), "environment")
    torch.cuda.init()
    mappings("cuda")
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    config = {**manifest["config"], "dropout": args.dropout}
    identity = manifest["identity"]
    model_path = Path(manifest["model_path"])
    if not args.timing:  # The controller verifies assets before launching a timing trial.
        for name, digest in identity["files"].items():
            state.compare(digest, file_hash(model_path / name), f"asset.{name}")
    state.compare(identity["tokens_sha256"], file_hash(args.assets / "tokens.json"), "tokens")
    tokens = json.loads((args.assets / "tokens.json").read_text())
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                dtype=torch.float32, attn_implementation="eager")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**config["lora"], lora_dropout=args.dropout, task_type="CAUSAL_LM"))
    model.cuda().train()
    mappings("model")
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in parameters.values()) != 540672:
        raise ValueError("Unexpected trainable parameter count")
    optimizer = torch.optim.AdamW(parameters.values(), **config["optimizer"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: max(0, 1 - n / config["updates"]))
    progress = {"update": 0, "next_example_index": 0, "example_order": list(range(len(tokens))),
                "dropout_calls": {}, "last_loss": None}
    if args.load:
        progress = state.load_application(args.load, model, optimizer, schedule, identity, config)

    def observe(name):
        def hook(module, inputs, output):
            if not module.training or module.p != args.dropout or not inputs[0].is_cuda:
                raise ValueError(f"Inactive CUDA dropout: {name}")
            counts = progress["dropout_calls"]
            counts[name] = counts.get(name, 0) + 1
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and "lora_dropout" in name:
            module.register_forward_hook(observe(name))

    def evidence(name, full=False):
        if not args.timing or full:
            record = state.inspect(model, optimizer, schedule, progress, identity, config)
            write_json(run / name, record)
            del record
        write_json(run / "memory.json", {**memory(), "allocated_vram": torch.cuda.memory_allocated(),
                                         "reserved_vram": torch.cuda.memory_reserved(),
                                         "peak_allocated_vram": torch.cuda.max_memory_allocated(),
                                         "peak_reserved_vram": torch.cuda.max_memory_reserved()})

    def inspect_restored(update):
        torch.cuda.synchronize()
        if args.timing:
            return  # Full correctness was checked separately; measure the next real update.
        evidence(f"after-{update}.json")  # Observe first; never repair process-restored state.
        (run / f"inspected-{update}").touch(exist_ok=False)
        wait(run / f"continue-{update}")

    if args.load:
        inspect_restored(progress["update"])
    else:
        evidence("state-0.json")
    initial = {n: state.tensor_record(p) for n, p in parameters.items()}
    while progress["update"] < args.until:
        start = time.monotonic_ns()
        index = progress["example_order"][progress["next_example_index"]]
        batch = {k: torch.tensor([v], device="cuda") for k, v in tokens[index].items()}
        rng_before = torch.cuda.get_rng_state()
        calls_before = sum(progress["dropout_calls"].values())
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
            raise ValueError("Missing or nonfinite adapter gradient")
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        value = loss.item()
        rng_changed = not torch.equal(rng_before, torch.cuda.get_rng_state())
        if args.dropout and (not rng_changed or sum(progress["dropout_calls"].values()) == calls_before):
            raise ValueError("Update did not exercise CUDA dropout")
        del batch, loss, output, rng_before
        progress.update(update=progress["update"] + 1, next_example_index=progress["next_example_index"] + 1,
                        last_loss=value)
        torch.cuda.synchronize()
        update = progress["update"]
        print(json.dumps({"update": update, "loss": value, "cuda_rng_changed": rng_changed,
                          "completed_ns": time.monotonic_ns(), "update_ns": time.monotonic_ns() - start}), flush=True)
        if update == 1:
            mappings("adam")
            if not all(set(optimizer.state[p]) == {"step", "exp_avg", "exp_avg_sq"} for p in parameters.values()):
                raise ValueError("Adam state missing")
        evidence(f"state-{update}.json")
        if update in args.pause_at:
            (run / f"ready-{update}").touch(exist_ok=False)
            if args.save:
                wait(run / f"save-{update}")
                state.save_application(args.save, model, optimizer, schedule, progress, identity, config)
                (run / f"saved-{update}").touch(exist_ok=False)
                return
            wait(run / f"inspect-{update}")
            inspect_restored(update)
    if not any(initial[n] != state.tensor_record(p) for n, p in parameters.items()):
        raise ValueError("Adapters did not change")
    if args.timing:
        evidence(f"state-{args.until}.json", full=True)  # After the measured next-update endpoint.
    (run / "done").touch(exist_ok=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dropout", type=float, choices=(0.0, 0.1), default=0.0)
    parser.add_argument("--until", type=int, choices=(2, 4), default=4)
    parser.add_argument("--pause-at", type=lambda s: [int(n) for n in s.split(",")], default=[])
    parser.add_argument("--save", type=Path)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--timing", action="store_true")
    main(parser.parse_args())
