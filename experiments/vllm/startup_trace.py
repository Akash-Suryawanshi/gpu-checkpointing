"""Diagnostic wall-clock spans; these are not CUDA kernel execution times."""

from contextlib import contextmanager, ExitStack
from functools import wraps
import json
import os
import time


WORKER_METHODS = (
    "init_device", "load_model", "determine_available_memory",
    "initialize_from_config", "compile_or_warm_up_model",
)


@contextmanager
def span(stream, name):
    """Flush starts too, so a killed worker leaves its unfinished phase visible."""
    if stream is None:
        yield
        return
    started = time.monotonic_ns()
    stream.write(json.dumps({"phase": name, "event": "start", "pid": os.getpid(),
                             "monotonic_ns": started}) + "\n")
    stream.flush()
    error = None
    try:
        yield
    except BaseException as caught:
        error = type(caught).__name__
        raise
    finally:
        ended = time.monotonic_ns()
        stream.write(json.dumps({"phase": name, "event": "end", "pid": os.getpid(),
                                 "monotonic_ns": ended, "seconds": (ended - started) / 1e9,
                                 "status": "failed" if error else "passed",
                                 "error_type": error}) + "\n")
        stream.flush()


@contextmanager
def worker_spans(stream):
    """Trace the in-process V1 worker, restoring its methods even after failure.

    Method boundaries include CPU work and any waits already in vLLM; adding a
    CUDA synchronization here would change the initialization being diagnosed.
    Missing methods fail explicitly instead of silently losing a phase.
    """
    if stream is None:
        yield
        return
    with span(stream, "instrumentation_imports"):
        from vllm.v1.worker.gpu_worker import Worker
        originals = {name: getattr(Worker, name) for name in WORKER_METHODS}

    def traced(name, original):
        @wraps(original)
        def call(*args, **kwargs):
            with span(stream, "worker." + name):
                return original(*args, **kwargs)
        return call

    with ExitStack() as stack:
        for name, original in originals.items():
            stack.callback(setattr, Worker, name, original)
            setattr(Worker, name, traced(name, original))
        yield
