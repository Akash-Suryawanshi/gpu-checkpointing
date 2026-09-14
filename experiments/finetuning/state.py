"""Named training evidence and complete application checkpoints for ordinary LoRA."""

import hashlib
import json
import os
from pathlib import Path
import random

import torch


def tensor_record(value):
    value = value.detach().contiguous()
    raw = value.reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), 4 * 1024 * 1024):
        digest.update(raw[start:start + 4 * 1024 * 1024].cpu().numpy().tobytes())
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": digest.hexdigest()}


def fingerprints(value):
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, dict):
        return {str(k): fingerprints(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprints(v) for v in value]
    return value


def rng_state():
    return {"python": random.getstate(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(saved):
    random.setstate(saved["python"])
    torch.set_rng_state(saved["cpu"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


def behavior(model):
    modules = dict(model.named_modules())
    return {"modes": {n: m.training for n, m in modules.items()},
            "trainable": {n: p.requires_grad for n, p in model.named_parameters()},
            "active_adapters": getattr(model, "active_adapters", []),
            "adapter_layers": {n: {"active": m.active_adapters, "disabled": m.disable_adapters,
                                   "merged": m.merged} for n, m in modules.items() if hasattr(m, "lora_A")},
            "dropout": {n: m.p for n, m in modules.items() if isinstance(m, torch.nn.Dropout)}}


def inspect(model, optimizer, schedule, progress, identity, config):
    parameters = dict(model.named_parameters())
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("Boundary contains uncleared gradients")
    names = {id(p): n for n, p in parameters.items()}
    groups = [{k: [names[id(p)] for p in v] if k == "params" else v for k, v in g.items()}
              for g in optimizer.param_groups]
    return fingerprints({"schema_version": 1, "progress": progress, "model_identity": identity,
                         "config": config, "training_behavior": behavior(model),
                         "adapters": {n: p for n, p in parameters.items() if p.requires_grad},
                         "base": {n: p for n, p in parameters.items() if not p.requires_grad},
                         "buffers": dict(model.named_buffers()),
                         "optimizer": {"states": {names[id(p)]: s for p, s in optimizer.state.items()},
                                       "groups": groups}, "schedule": schedule.state_dict(), "rng": rng_state()})


def compare(expected, actual, path="state"):
    """Stop at the first differing field, including missing fields and sequence length."""
    if type(expected) is not type(actual):
        raise ValueError(f"{path}: type differs")
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"{path}: keys differ")
        for key in expected:
            compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"{path}: length differs")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{i}]")
    elif expected != actual:
        raise ValueError(f"{path}: values differ")


def save_application(path, model, optimizer, schedule, progress, identity, config):
    path = Path(path)
    candidate = path.with_suffix(".temp")
    payload = {"schema_version": 1, "identity": identity, "config": config, "progress": progress,
               "adapters": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
               "buffers": {n: p.detach().cpu() for n, p in model.named_buffers()},
               "parameter_names": [n for n, p in model.named_parameters() if p.requires_grad],
               "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
               "behavior": behavior(model), "rng": rng_state()}
    try:
        with candidate.open("xb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(candidate, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        candidate.unlink(missing_ok=True)


def load_application(path, model, optimizer, schedule, identity, config):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    compare(saved["schema_version"], 1, "schema_version")
    compare(saved["identity"], identity, "identity")
    compare(saved["config"], config, "config")
    compare(saved["parameter_names"], [n for n, p in model.named_parameters() if p.requires_grad], "parameter_names")
    with torch.no_grad():
        for name, value in saved["adapters"].items():
            model.get_parameter(name).copy_(value)
        for name, value in saved["buffers"].items():
            model.get_buffer(name).copy_(value)
    # The scheduler must exist before loading optimizer state (including the current LR).
    schedule.load_state_dict(saved["schedule"])
    optimizer.load_state_dict(saved["optimizer"])
    for name, mode in saved["behavior"]["modes"].items():
        model.get_submodule(name).training = mode
    compare(fingerprints(saved["behavior"]), fingerprints(behavior(model)), "training_behavior")
    restore_rng(saved["rng"])  # Last: setup and loading must not advance the saved stream.
    return saved["progress"]
