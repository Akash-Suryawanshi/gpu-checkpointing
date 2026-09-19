"""A vLLM worker that serves file requests under the existing control protocol.

The engine runs inside this one process, so a capture sees the whole serving
state: weights, compiled kernels, and captured CUDA graphs. Startup work here is
computation rather than file reading, which is the difference this experiment
tests; see the [plan](../../docs/vllm-snapshot-plan.md).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Ahead of this script's own directory: a bare "prepare"/"control" import must
# reach the finetuning modules, never the same-named files next to this one.
sys.path[:0] = [str(ROOT / "finetuning"), str(ROOT / "inference")]

from worker import control, memory, next_request, redirect_output, attempt_paths  # noqa: E402

# vLLM starts its engine core in a separate process by default. One process is
# what CRIU dumps and what GPU monitoring attributes, so that split is disabled.
# USE_LIBUV=0 selects PyTorch's older distributed store: the libuv backend opens an
# io_uring ring, and CRIU refuses to dump a process holding one, failing with
# "Unknown shit 600 (anon_inode:[io_uring])" before it reaches any GPU mapping.
# Usage reporting is off because a measured run makes no network calls.
ENGINE_ENVIRONMENT = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "USE_LIBUV": "0",
                      "VLLM_NO_USAGE_STATS": "1", "VLLM_DO_NOT_TRACK": "1",
                      "VLLM_LOGGING_LEVEL": "WARNING"}


def eager_route(route):
    """Only the eager control disables CUDA-graph capture; the routes differ here alone."""
    return route == "eager"


def version():
    import vllm
    return vllm.__version__


def load(model_path, gpu_fraction, enforce_eager, max_model_len, seed=2026):
    """Start one engine and leave it holding the GPU state a snapshot would carry."""
    # Applied here rather than by each caller: preparation, the worker and any
    # probe must all get the same engine, and a forked engine core cannot
    # initialise CUDA once this process has touched it. These values are part of
    # the measured configuration and are recorded in the comparison key.
    os.environ.update(ENGINE_ENVIRONMENT)
    import torch
    from vllm import LLM
    torch.manual_seed(seed)
    started = time.monotonic()
    # Prefix caching is off so no request leaves key-value blocks behind for the
    # next one: each timed request recomputes its own prompt, as the reference did.
    engine = LLM(model=str(model_path), dtype="bfloat16", seed=seed,
                 gpu_memory_utilization=gpu_fraction, enforce_eager=enforce_eager,
                 max_model_len=max_model_len, enable_prefix_caching=False,
                 disable_log_stats=True, tensor_parallel_size=1)
    torch.cuda.synchronize()
    cache = cache_config(engine)
    return engine, {"load_call_seconds": time.monotonic() - started,
                    "enforce_eager": enforce_eager, "gpu_fraction": gpu_fraction,
                    "gpu_blocks": cache.num_gpu_blocks, "block_size": cache.block_size}


def core(engine):
    """The synchronous engine behind the offline wrapper, used to stream tokens."""
    return engine.llm_engine


def cache_config(engine):
    """Key-value cache geometry. The engine core fills in the block count during
    its startup profiling, so this is only meaningful once loading has finished."""
    return core(engine).vllm_config.cache_config


def model_of(engine):
    """Reach the loaded module. vLLM keeps it behind its executor's single worker."""
    holder = core(engine).model_executor
    # Each level is optional: a worker may or may not sit behind a wrapper.
    for name in ("driver_worker", "worker", "model_runner", "model"):
        holder = getattr(holder, name, holder)
    if not hasattr(holder, "state_dict"):
        raise AttributeError("vLLM executor layout changed; cannot reach the model")
    return holder


def generate(engine, inputs, count, on_token=None, request_id="request"):
    """Stream one greedy completion, reporting each token as the engine emits it."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    running = core(engine)
    if running.has_unfinished_requests():
        raise RuntimeError("Engine still has work; requests are served one at a time")
    parameters = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=count, detokenize=True,
                                output_kind=RequestOutputKind.DELTA)
    running.add_request(request_id, {"prompt_token_ids": list(inputs["input_ids"])}, parameters)
    tokens, text, reason = [], "", None
    while running.has_unfinished_requests():
        # step() returns only what is new for this request, so position is len(tokens).
        for output in running.step():
            if output.request_id != request_id:
                raise ValueError("Engine returned another request's output")
            completion = output.outputs[0]
            for token in completion.token_ids:
                if on_token is not None:
                    on_token(len(tokens), int(token), completion.text)
                tokens.append(int(token))
            text += completion.text
            reason = completion.finish_reason or reason
    return {"tokens": tokens, "text": text, "finish_reason": reason}


def release_kv_cache(engine):
    """Drop cached blocks before an image is taken, and say what the image still holds.

    Only cached content goes: the key-value tensors themselves are preallocated at
    startup and stay in the image. Their size is set by gpu_memory_utilization,
    which is recorded in the comparison key rather than trimmed here.
    """
    import torch
    running = core(engine)
    if running.has_unfinished_requests():
        raise RuntimeError("Cannot release the cache while a request is running")
    engine.reset_prefix_cache()
    torch.cuda.synchronize()
    cache = cache_config(engine)
    return {"released_ns": time.monotonic_ns(), "prefix_cache_reset": True,
            "gpu_blocks": cache.num_gpu_blocks, "block_size": cache.block_size,
            "reserved_vram": torch.cuda.memory_reserved()}


def inspect(engine, kind):
    """The immutable state two audits compare: weights, placement, cache geometry."""
    import state
    model = model_of(engine)
    tensors = model.state_dict()
    if any(module.training for module in model.modules()):
        raise ValueError("Engine model must be in evaluation mode")
    if any(tensor.device.type != "cuda" for tensor in tensors.values()):
        raise ValueError("Engine model must be entirely on GPU")
    # The profiled block count belongs in loading.json, not here. It is derived from
    # free GPU memory at startup and differs between two processes loading the same
    # weights, so an audit holding it would report identical models as changed.
    record = {"training": False, "kv_cache": None,
              "tensors": {name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                                 "device": str(tensor.device)} for name, tensor in tensors.items()}}
    if kind == "diagnostic":
        record["fingerprints"] = state.fingerprints(tensors)
    return record


def serve(requests, run_id, engine, inputs, poll):
    """Answer file requests until the owner stops; audit the weights only afterwards."""
    control.write(requests / "ready.json", {"run_id": run_id})
    index = 0
    while (request := next_request(requests, index := index + 1, poll)) is not None:
        if request["run_id"] != run_id:
            raise ValueError("Request identity mismatch")
        identity = {"run_id": run_id, "request_id": request["request_id"]}
        count = request.get("max_new_tokens", inputs["max_new_tokens"])
        with (requests / f"tokens-{index}.jsonl").open("a") as stream:
            def on_token(position, token, text, index=index, identity=identity, stream=stream):
                stream.write(json.dumps({**identity, "index": position, "token_id": token,
                                         "text": text}) + "\n")
                stream.flush()
                if position == 0:
                    control.write(requests / f"first-{index}.json", {**identity, "token": token})
            response = generate(engine, {"input_ids": request["input_ids"]}, count,
                                on_token, request_id=identity["request_id"])
        control.write(requests / f"response-{index}.json", {**identity, **response})
    # A full weight audit reads no request for its duration; it belongs after the
    # owner has stopped sending, never between two timed responses.
    control.write(requests / "audit-after.json", inspect(engine, "diagnostic"))
    control.write(requests / "completed.json", memory())


def main(args):
    run = args.run_dir.resolve()
    os.umask(0o077)
    job = control.register(run, args.assets, 2) if args.route == "snapshot" else None
    manifest = control.read(args.assets / "manifest.json")
    inputs = control.read(args.assets / "tokens.json")
    settings = control.read(run / "run.json")["key"]
    # Graph capture follows the route, and is deliberately not a key setting: the
    # three routes must share one comparison key to be aggregated against each other.
    engine, loading = load(manifest["model_path"], settings["gpu_fraction"],
                           eager_route(args.route), settings["max_model_len"])
    control.write(run / "loading.json", loading)
    poll = settings["poll_ms"] / 1000
    if args.route == "snapshot":
        # Only a warmed engine is worth capturing: graphs are captured at startup,
        # but the first real request still allocates and touches its own blocks.
        generate(engine, inputs["warmup"], 4, request_id="warmup")
        control.write(run / "kv-release.json", release_kv_cache(engine))
        control.write(run / "audit-before.json", inspect(engine, "diagnostic"))
    control.write(run / "memory.json", memory())
    control.write(run / "idle.json", {"run_id": args.run_id, "kv_cache": None})
    requests = run
    if args.route == "snapshot":
        control.wait(run / "control/request.json", time.monotonic() + 7200, poll=poll)
        restoration = control.boundary(run, job, 1, lambda: inspect(engine, args.kind))
        record = control.activation(run)
        if record is not None:
            root, requests = attempt_paths(run, record, restoration)
            redirect_output(root / "worker.log")
            control.write(root / "acknowledged.json", {**restoration, "pid": os.getpid()})
        elif control.read(run / "control/phase.json")["status"] not in ("verified", "restored"):
            raise RuntimeError("Capture did not produce a verified restore")
    serve(requests, args.run_id, engine, inputs, poll)
    time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route", choices=("cold", "eager", "snapshot"), required=True)
    parser.add_argument("--kind", choices=("diagnostic", "timing"), required=True)
    parser.add_argument("--run-id", required=True)
    main(parser.parse_args())
