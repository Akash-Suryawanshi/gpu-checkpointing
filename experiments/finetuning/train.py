"""Run a small LoRA workload with observable, completed-update capture boundaries.

The trainer and controller are separate processes: each is a running program with
its own memory. This process owns training; the controller owns capture, restore,
and permission to continue. An application restore rebuilds objects from a file;
a CRIU restore recreates saved process memory and resumes at its waiting point.
"""

import argparse
import json
import os
from pathlib import Path
import random
import time

from prepare import environment, file_hash, write_json


def memory():
    """Read current resident RAM (VmRSS) and its lifetime high-water mark (VmHWM).

    Linux exposes live process information through /proc; self means this process.
    Resident RAM is physical CPU memory currently backing it, not a saved image's
    disk size. Keep the kernel's units. CUDA statistics separately describe VRAM.
    """
    status = Path("/proc/self/status").read_text().splitlines()
    return {
        row.split(":")[0]: row.split(":")[1].strip()
        for row in status
        if row.startswith(("VmRSS:", "VmHWM:"))
    }


def wait(path):
    """Wait for a file whose existence signals permission from the controller.

    Their ordinary Python variables are separate, but they can see the same files.
    Markers therefore coordinate them before capture and after process recreation.
    """
    # The controller enforces timeouts: a saved deadline could expire while this
    # process is absent from the GPU, even though its restored state is healthy.
    while not path.exists():
        time.sleep(0.05)


def main(args):
    """Build the pinned trainer, then update, inspect, and pause at safe boundaries."""
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)  # Restrict newly created evidence files to their owning user.

    def mappings(stage):
        """Record Linux's list of this process's address ranges at each milestone.

        /proc/self/maps describes ranges of virtual addresses, their access flags,
        and any backing files. It does not contain the bytes stored in those ranges.
        These snapshots help investigate shared-memory compatibility warnings.
        """
        (run / f"{stage}.maps").write_text(Path("/proc/self/maps").read_text())

    # Import in stages so the mapping evidence distinguishes Python, PyTorch,
    # CUDA initialization, model allocation, and the first Adam allocation.
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

    # Reuse prepared files for every run. Dropout is the one per-trial setting;
    # all model, tokenizer-output, and environment identities must still match.
    config = {**manifest["config"], "dropout": args.dropout}
    identity = manifest["identity"]
    model_path = Path(manifest["model_path"])
    if not args.timing:
        # Timing trials move these large file hashes to the controller's preflight.
        for name, digest in identity["files"].items():
            state.compare(digest, file_hash(model_path / name), f"asset.{name}")
    state.compare(identity["tokens_sha256"], file_hash(args.assets / "tokens.json"), "tokens")
    tokens = json.loads((args.assets / "tokens.json").read_text())

    # Seed before adapter construction. Deterministic kernels and disabled TF32
    # keep the comparison strict; dropout may still consume the seeded RNG stream.
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False  # Training does not need a generation KV cache.
    model = get_peft_model(
        model,
        LoraConfig(**config["lora"], lora_dropout=args.dropout, task_type="CAUSAL_LM"),
    )
    # Pretrained loading uses evaluation mode. Explicit training mode is required
    # for the dropout experiment; successful gradients alone cannot prove it ran.
    model.cuda().train()
    mappings("model")
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    # Guard the expected Qwen model and q/v LoRA configuration against drift.
    if sum(p.numel() for p in parameters.values()) != 540672:
        raise ValueError("Unexpected trainable parameter count")

    # Only adapters train. Adam allocates its moment tensors on the first update;
    # the linear schedule is part of the continuation state, even in this tiny run.
    optimizer = torch.optim.AdamW(parameters.values(), **config["optimizer"])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda n: max(0, 1 - n / config["updates"])
    )
    progress = {
        "update": 0,
        "next_example_index": 0,
        "example_order": list(range(len(tokens))),
        "dropout_calls": {},
        "last_loss": None,
    }
    if args.load:
        # Application restoration reconstructs objects and loads their saved state.
        # CRIU restoration bypasses setup: it resumes at the saved wait below.
        progress = state.load_application(args.load, model, optimizer, schedule, identity, config)

    def observe(name):
        """Bind each dropout module's name into a hook that proves it was executed."""
        def hook(module, inputs, output):
            """Reject inactive/wrong-device dropout and count this module's call."""
            if not module.training or module.p != args.dropout or not inputs[0].is_cuda:
                raise ValueError(f"Inactive CUDA dropout: {name}")
            counts = progress["dropout_calls"]
            counts[name] = counts.get(name, 0) + 1

        return hook

    # A configured probability is insufficient evidence: these hooks see actual
    # LoRA forward calls, and each update separately checks CUDA RNG consumption.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and "lora_dropout" in name:
            module.register_forward_hook(observe(name))

    def evidence(name, full=False):
        """Write exact state evidence in diagnostics and allocator memory in all runs.

        Full inspection hashes model and optimizer tensors, which is substantial
        work. Timing runs defer it until after the measured next-update endpoint.
        """
        if not args.timing or full:
            write_json(run / name, state.inspect(model, optimizer, schedule, progress, identity, config))
        # Allocated memory belongs to live PyTorch tensors; reserved memory also
        # includes its allocator cache. These figures are not total driver VRAM.
        write_json(run / "memory.json", {
            **memory(),
            "allocated_vram": torch.cuda.memory_allocated(),
            "reserved_vram": torch.cuda.memory_reserved(),
            "peak_allocated_vram": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram": torch.cuda.max_memory_reserved(),
        })

    def inspect_restored(update):
        """Expose restored state before allowing a diagnostic run's next update."""
        torch.cuda.synchronize()
        if args.timing:
            return  # Correctness was checked separately; time the next real update.
        # Observe modes, adapters, flags, RNG, and tensors before changing anything:
        # calling train() or reseeding here could conceal a bad process restore.
        evidence(f"after-{update}.json")
        (run / f"inspected-{update}").touch(exist_ok=False)
        wait(run / f"continue-{update}")

    if args.load:
        inspect_restored(progress["update"])
    else:
        evidence("state-0.json")
    # Retain only adapter fingerprints to prove that the remaining updates train.
    initial = {n: state.tensor_record(p) for n, p in parameters.items()}
    while progress["update"] < args.until:
        start = time.monotonic_ns()
        index = progress["example_order"][progress["next_example_index"]]
        # One fixed example per update avoids data-loader workers and hidden cursors.
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

        # The capture boundary follows the whole update: parameters, optimizer,
        # schedule, cleared gradients, and data position must all agree.
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        value = loss.item()
        rng_changed = not torch.equal(rng_before, torch.cuda.get_rng_state())
        if args.dropout and (
            not rng_changed or sum(progress["dropout_calls"].values()) == calls_before
        ):
            raise ValueError("Update did not exercise CUDA dropout")
        del batch, loss, output, rng_before  # Do not retain per-update tensor references at capture.
        progress.update(
            update=progress["update"] + 1,
            next_example_index=progress["next_example_index"] + 1,
            last_loss=value,
        )
        torch.cuda.synchronize()  # Finish asynchronous CUDA work before reporting completion.
        update = progress["update"]
        # A monotonic clock measures elapsed time. Linux can give a restored
        # process its own clock offset (a time namespace), so the controller uses
        # recorded offsets to convert completed_ns into its clock. update_ns is a
        # difference within this process, so the offset cancels from that duration.
        print(json.dumps({
            "update": update,
            "loss": value,
            "cuda_rng_changed": rng_changed,
            "completed_ns": time.monotonic_ns(),
            "update_ns": time.monotonic_ns() - start,
        }), flush=True)
        if update == 1:
            mappings("adam")
            # A one-update capture must include initialized Adam moments, not an
            # empty optimizer that has yet to exercise its allocation path.
            if not all(
                set(optimizer.state[p]) == {"step", "exp_avg", "exp_avg_sq"}
                for p in parameters.values()
            ):
                raise ValueError("Adam state missing")
        evidence(f"state-{update}.json")
        if update in args.pause_at:
            # Generation-specific markers prevent a repeated restore from taking
            # an earlier update's signal as permission to proceed.
            (run / f"ready-{update}").touch(exist_ok=False)
            if args.save:
                wait(run / f"save-{update}")
                state.save_application(args.save, model, optimizer, schedule, progress, identity, config)
                (run / f"saved-{update}").touch(exist_ok=False)
                return  # The controller verifies this original trainer has exited.
            # CRIU captures while this wait is active. Only after restoring the
            # process does the controller create inspect-N, then continue-N after
            # comparing its evidence with the uninterrupted reference.
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
