"""Inspect complete training state and save the smaller application checkpoint.

Inspection hashes every model tensor, including the frozen base, so comparisons
can detect changed bytes without writing another full model. Application saves
contain the trainable adapters and continuation state; the caller reconstructs
the frozen model from the pinned identity before loading them. These functions
do not perform CRIU process capture.
"""

import hashlib
import os
from pathlib import Path
import random

import torch


def tensor_record(value):
    """Describe a tensor by its dtype, shape, and SHA-256 of its exact bytes.

    Detaching avoids building an autograd graph during inspection. Contiguous
    storage puts values in logical order; viewing as bytes preserves their exact
    representation rather than rounding values through a text conversion.
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
    """Replace tensors in nested state with records suitable for JSON comparison.

    Dictionary keys become strings and tuples become lists, matching the JSON
    evidence format. Scalar metadata stays unchanged so it is checked as well.
    """
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
    """Read the execution settings that tensor values alone cannot establish.

    Dropout needs both a nonzero probability and training mode. Adapter
    selection, disabled/merged adapter layers, and gradient flags also affect
    the next update, so inspection records them without changing the model.
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
    """Fingerprint a completed update, including state needed by the next one.

    This boundary requires gradients to have been cleared. Parameter names link
    optimizer moments and parameter groups to model tensors across processes;
    Python object IDs themselves would differ after application reconstruction.
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
    """Fail at the first mismatch and report its location in the nested state.

    Types, dictionary keys, and list lengths must match before values are
    compared. A missing field or an extra sequence entry cannot silently pass;
    scalar comparisons are exact, with no numerical tolerance.
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
    """Write, atomically publish, and locally sync an application checkpoint.

    Save adapters, buffers, and continuation state. Frozen weights come from
    the pinned base model on reload; parameter order and model identity guard
    against loading optimizer state into a different reconstruction.
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
        # Exclusive creation rejects an existing temporary file. flush() moves
        # Python's buffered bytes to Linux; fsync() asks Linux to persist them.
        # fileno() is the integer file descriptor Linux uses for this open file.
        with candidate.open("xb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        # A rename publishes the completed file in one operation, so readers do
        # not see a half-written checkpoint under the final filename.
        os.replace(candidate, path)
        # Sync the rename itself. This is local filesystem durability, not an
        # upload to storage that would survive losing the instance or its disk.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        candidate.unlink(missing_ok=True)


def load_application(path, model, optimizer, schedule, identity, config):
    """Load saved state into a reconstructed model, then restore RNG streams last.

    Validate identity, configuration, and parameter order before copying state.
    Restore module modes, but require adapter settings and trainable flags to
    already match the reconstruction. This is application-checkpoint loading;
    CRIU restores process memory and is inspected separately without repairs.
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
    # Copy tensor contents in place so optimizer references still point to the
    # same parameters; checkpoint loading must not create an autograd graph.
    with torch.no_grad():
        for name, value in saved["adapters"].items():
            model.get_parameter(name).copy_(value)
        for name, value in saved["buffers"].items():
            model.get_buffer(name).copy_(value)
    # The scheduler must exist before loading optimizer state (including the
    # current learning rate), because constructing a scheduler can change it.
    schedule.load_state_dict(saved["schedule"])
    optimizer.load_state_dict(saved["optimizer"])
    for name, mode in saved["behavior"]["modes"].items():
        # Set each flag directly: train(mode) would recursively overwrite child
        # modes, potentially losing a deliberate mix of training/evaluation.
        model.get_submodule(name).training = mode
    compare(fingerprints(saved["behavior"]), fingerprints(behavior(model)), "training_behavior")
    # Last: setup and loading must not consume the saved random stream.
    restore_rng(saved["rng"])
    return saved["progress"]
