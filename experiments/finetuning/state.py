"""Fingerprint complete state; save adapters and continuation state separately.

Application restore rebuilds the frozen base from its pinned identity, then
loads saved state. CRIU restores process memory instead; see criu_pipeline.py.
"""

import hashlib
import os
from pathlib import Path
import random

import torch


def tensor_record(value):
    """Record dtype, shape, and a SHA-256 hash of the exact tensor bytes.

    Detach from gradient tracking, arrange values contiguously in logical order,
    then view their bytes without numerical conversion or rounding.
    """
    value = value.detach().contiguous()
    raw = value.reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    # Transfer at most 4 MiB per chunk to avoid a full host copy of a GPU tensor.
    for start in range(0, raw.numel(), 4 * 1024 * 1024):
        digest.update(raw[start:start + 4 * 1024 * 1024].cpu().numpy().tobytes())
    # Equal byte strings alone would not distinguish different shapes or dtypes.
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": digest.hexdigest()}


def fingerprints(value):
    """Replace tensors with fingerprints; normalize keys and sequences for JSON."""
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, dict):
        return {str(key): fingerprints(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprints(item) for item in value]
    return value


def rng_state():
    """Capture Python, CPU PyTorch, and all already-initialized CUDA RNG streams."""
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state(),
        # Inspection must not create a CUDA context in a CPU-only process.
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(saved):
    """Restore the streams after reconstruction so the next random draws match."""
    random.setstate(saved["python"])
    torch.set_rng_state(saved["cpu"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


def behavior(model):
    """Read non-tensor settings without changing the model.

    Modes, adapter selection, gradient flags, and dropout affect continuation.
    Dropout needs both a nonzero probability and training mode.
    """
    modules = dict(model.named_modules())
    return {
        "modes": {name: module.training for name, module in modules.items()},
        "trainable": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
        "active_adapters": getattr(model, "active_adapters", []),
        "adapter_layers": {
            name: {
                "active": module.active_adapters,
                "disabled": module.disable_adapters,
                "merged": module.merged,
            }
            for name, module in modules.items() if hasattr(module, "lora_A")
        },
        "dropout": {
            name: module.p for name, module in modules.items()
            if isinstance(module, torch.nn.Dropout)
        },
    }


def inspect(model, optimizer, schedule, progress, identity, config):
    """Fingerprint continuation state after an update with cleared gradients.

    Identify optimizer tensors by parameter names: Python object IDs change
    when application restore reconstructs the model.
    """
    parameters = dict(model.named_parameters())
    if any(parameter.grad is not None for parameter in parameters.values()):
        raise ValueError("Boundary contains uncleared gradients")
    names = {id(parameter): name for name, parameter in parameters.items()}
    groups = []
    for group in optimizer.param_groups:
        # Replace only parameter references; keep learning rates and other settings.
        groups.append({
            key: [names[id(parameter)] for parameter in value] if key == "params" else value
            for key, value in group.items()
        })
    return fingerprints({
        "schema_version": 1,
        "progress": progress,
        "model_identity": identity,
        "config": config,
        "training_behavior": behavior(model),
        # In this ordinary LoRA workload, only adapters require gradients.
        "adapters": {name: parameter for name, parameter in parameters.items() if parameter.requires_grad},
        "base": {name: parameter for name, parameter in parameters.items() if not parameter.requires_grad},
        "buffers": dict(model.named_buffers()),
        "optimizer": {
            "states": {names[id(parameter)]: value for parameter, value in optimizer.state.items()},
            "groups": groups,
        },
        "schedule": schedule.state_dict(),
        "rng": rng_state(),
    })


def compare(expected, actual, path="state"):
    """Report the first mismatch, including types, missing fields, and lengths.

    Compare scalar values exactly, without numerical tolerance.
    """
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
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{index}]")
    elif expected != actual:
        raise ValueError(f"{path}: values differ")


def save_application(path, model, optimizer, schedule, progress, identity, config):
    """Save adapters, buffers, and continuation state, then persist locally.

    Identity and parameter order guard the reconstructed model on reload.
    """
    path = Path(path)
    candidate = path.with_suffix(".temp")
    payload = {
        "schema_version": 1,
        "identity": identity,
        "config": config,
        "progress": progress,
        "adapters": {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        },
        "buffers": {name: buffer.detach().cpu() for name, buffer in model.named_buffers()},
        "parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "behavior": behavior(model),
        "rng": rng_state(),
    }
    try:
        # "xb" rejects an existing file. flush() sends Python's buffered bytes
        # to Linux; fsync() persists them via the file's integer handle (fileno).
        with candidate.open("xb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        # Rename publishes only the finished file under its final name.
        os.replace(candidate, path)
        # Persist the directory's rename too; this cannot survive disk deletion.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        candidate.unlink(missing_ok=True)


def load_application(path, model, optimizer, schedule, identity, config):
    """Load into reconstructed objects; restore random-generator streams last.

    Check identity/configuration before copying. Restore module modes; require
    the other behavior flags to match. CRIU inspection never calls this repair.
    """
    # Stage serialized tensors on the CPU. copy_ keeps the model's existing
    # devices; the optimizer loader applies its own state-placement rules.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    compare(saved["schema_version"], 1, "schema_version")
    compare(saved["identity"], identity, "identity")
    compare(saved["config"], config, "config")
    compare(
        saved["parameter_names"],
        [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "parameter_names",
    )
    # Copy without gradient tracking; keep the optimizer's parameter references.
    with torch.no_grad():
        for name, value in saved["adapters"].items():
            model.get_parameter(name).copy_(value)
        for name, value in saved["buffers"].items():
            model.get_buffer(name).copy_(value)
    # Construct the scheduler before loading the optimizer's current learning
    # rate: scheduler construction can overwrite it.
    schedule.load_state_dict(saved["schedule"])
    optimizer.load_state_dict(saved["optimizer"])
    for name, mode in saved["behavior"]["modes"].items():
        # train(mode) recurses into children; direct flags preserve mixed modes.
        model.get_submodule(name).training = mode
    compare(fingerprints(saved["behavior"]), fingerprints(behavior(model)), "training_behavior")
    # Last: setup and loading must not consume the saved random stream.
    restore_rng(saved["rng"])
    return saved["progress"]
