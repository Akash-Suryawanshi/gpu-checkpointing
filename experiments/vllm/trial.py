"""Supervise one vLLM route — cold, eager, or snapshot — under one comparison key.

Capture, restore, ownership, eviction and device counters are reused from the
inference experiment; only the engine being measured is new. See the
[plan](../../docs/vllm-snapshot-plan.md) and the [runbook](README.md).
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Ahead of this script's own directory: shared module names must resolve to the
# modules being reused, not to same-named files here.
sys.path[:0] = [str(ROOT / "finetuning"), str(ROOT / "inference")]
# Last, so this directory's own modules are importable when a test loads these
# files by location, without ever shadowing a shared module name.
sys.path.append(str(HERE))

import cold_cache  # noqa: E402
import io_stats  # noqa: E402
import lifecycle  # noqa: E402
import measure  # noqa: E402
import park  # noqa: E402
from prepare_assets import validate_assets  # noqa: E402
from runtime import CACHE_KEYS, admit, dispatch, stop_worker  # noqa: E402
from worker import control  # noqa: E402
from control import session  # noqa: E402

import engine  # noqa: E402


def comparison_key(deps, manifest, env, **settings):
    """Everything that must match between a diagnostic and the runs it admits.

    The route is not in here. Cold, eager and snapshot differ only by route, so
    they share one key and can be compared; a setting that varies between them
    would split the campaign into keys that cannot be aggregated.
    """
    return {"dependencies": deps, "assets": manifest, "engine": "vllm",
            "engine_version": manifest["engine"]["version"],
            "gpu_fraction": manifest["engine"]["gpu_fraction"],
            "max_model_len": manifest["engine"]["max_model_len"],
            "boundary_schema": 2, "integrity_policy": "strict-v1",
            "inspection_policy": "full-diagnostic-post-response-timing-v1",
            "engine_environment": engine.ENGINE_ENVIRONMENT,
            "cache_paths": {k: env[k] for k in (*CACHE_KEYS, "VLLM_CACHE_ROOT") if k in env},
            **settings}


def compile_cache(policy, root, output):
    """Apply the compiled-kernel policy before a launch, and record what was there.

    `cold` deletes the cache so the route pays for its own compilation. `warm`
    requires one to exist already: a campaign whose first block had to compile
    would charge that block for work the other two inherit.
    """
    entries = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    record = {"policy": policy, "root": str(root), "files": len(entries),
              "bytes": sum(path.stat().st_size for path in entries)}
    if policy == "cold":
        if root.exists():
            shutil.rmtree(root)
        record["removed"] = True
    elif not entries:
        raise ValueError(f"Warm compiled-kernel cache required at {root}; run one throwaway "
                         "activation first, or choose --compile-cache cold")
    control.write(output / "compile-cache.json", record)
    return record


def trial(args):
    output, assets, tools = args.output.resolve(), args.assets.resolve(), args.tools.resolve()
    output.mkdir(mode=0o700, parents=True)
    deadline = time.monotonic() + args.timeout
    helper = tools / "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint"
    env = {**session.child_environment(helper), **engine.ENGINE_ENVIRONMENT}
    env["VLLM_CACHE_ROOT"] = str(Path(os.environ.get("VLLM_CACHE_ROOT") or Path(env["HOME"]) / ".cache/vllm"))
    process, original, sampler = None, None, None
    details, phase, failure = {}, "admission", None
    emit = lambda name, **values: control.event(output, name, **values)
    session.adopt_restored_children()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        manifest = validate_assets(assets)
        if manifest.get("engine", {}).get("name") != "vllm":
            raise ValueError("Assets were not prepared with vLLM")
        deps = control.dependencies({"assets": str(assets), "python": sys.executable}, tools, deadline)
        key = comparison_key(deps, manifest, env, poll_ms=args.poll_ms, sample_ms=args.sample_ms,
                             data_cache=args.data_cache, compile_cache=args.compile_cache)
        diagnostic = None
        if args.kind == "timing":
            diagnostic = measure.diagnostic_record(args.validated_run, key, args.route,
                                                   validate=measure.vllm_run)
        run = {"run_id": uuid.uuid4().hex, "route": args.route, "kind": args.kind, "block": args.block,
               "key": key, "diagnostic": diagnostic, "cache_condition": args.data_cache}
        control.write(output / "run.json", run)
        for source, target in (("reference.json", "reference.json"), ("tokens.json", "tokens.json"),
                               ("audit.json", "prepared-audit.json")):
            shutil.copyfile(assets / source, output / target)
        sampler = park.Sampler(output, args.sample_ms / 1000, deadline)
        sampler.thread.start()
        observed = park.resources(deadline)
        control.write(output / "initial-resources.json", {**observed, "root": control.storage(Path("/")),
                                                          "data": control.storage(output)})
        if observed["compute_pids"]:
            raise ValueError("GPU must start without foreign compute processes")
        model_files = [Path(manifest["model_path"]) / name for name in manifest["identity"]["files"]
                       if name.endswith(".safetensors")]
        cold_cache_rows = None

        def evict(extra=()):
            nonlocal cold_cache_rows
            if args.data_cache == "cold":
                cold_cache_rows = cold_cache.evict([*model_files, *extra])
                control.write(output / "cold-cache.json", cold_cache_rows)
                emit("data_cache_evicted")

        phase = "launch"
        compile_cache(args.compile_cache, Path(env["VLLM_CACHE_ROOT"]), output)
        if args.route != "snapshot":
            evict()
        io_before = io_stats.sample([manifest["model_path"], output])
        started = time.monotonic_ns()
        emit("activation_started", start_ns=started, route=args.route)
        process = session.launch(sys.executable, HERE / "engine.py", ["--assets", assets,
            "--run-dir", output, "--route", args.route, "--kind", args.kind,
            "--run-id", run["run_id"]], output, env)
        original = session.identity(process.pid)
        sampler.pid = process.pid
        emit("worker_launched", identity=original)
        print(f"{args.route}: worker identity {json.dumps(original)}, waiting for idle", flush=True)
        control.wait(output / "idle.json", deadline, original, args.poll_ms / 1000)
        memory = control.read(output / "memory.json")
        admit(output, "loaded", memory, deadline, original)
        details["preparation_seconds"] = (time.monotonic_ns() - started) / 1e9
        if args.route == "snapshot":
            if control.read(output / "audit-before.json") != control.read(output / "prepared-audit.json"):
                raise ValueError("Warmup changed immutable model state")
            phase = "capture"
            admit(output, "capture", memory, deadline, original)
            details["kv_release_ns"] = control.read(output / "kv-release.json")["released_ns"]
            details["capture_start_ns"] = time.monotonic_ns()
            details.update(lifecycle.capture(output, tools, env, process, deadline))
            process, original, sampler.pid = None, None, None
            details["save_seconds"] = (time.monotonic_ns() - details["capture_start_ns"]) / 1e9
            print("snapshot: published; original reaped; capture command exited", flush=True)
            phase = "restore"
            admit(output, "restore", control.read(output / "snapshot/manifest.json")["pre_staging_memory"], deadline)
            evict(path for path in (output / "snapshot/images").rglob("*") if path.is_file())
            io_before = io_stats.sample([manifest["model_path"], output])
            started = time.monotonic_ns()
            emit("activation_started", start_ns=started, route=args.route)
            original = lifecycle.restore(output, tools, env, deadline)
            sampler.pid = original["pid"]
            details["restore_exited"] = True
            print(f"snapshot: restore exited; worker {original['pid']} survived", flush=True)
        else:
            # These routes never warm up, so their first request is the audited one.
            shutil.copyfile(output / "prepared-audit.json", output / "audit-before.json")
        phase = "responses"
        control.wait(output / "ready.json", deadline, original, args.poll_ms / 1000)
        emit("worker_ready_observed")
        observations = []
        for index in (1, 2):
            request = {"run_id": run["run_id"], "request_id": uuid.uuid4().hex, "index": index,
                       "input_ids": control.read(output / "tokens.json")["benchmark"]["input_ids"]}
            if index == 2:
                started = time.monotonic_ns()
            first_seen = []

            def on_token(event, index=index, request=request):
                if first_seen:
                    return
                first_seen.append(event)
                emit("first_token_received", request_id=request["request_id"], received_ns=time.monotonic_ns())
                if index == 1:
                    control.write(output / "io.json", {"window": "activation_start_to_first_token",
                        "devices": io_stats.delta(io_before, io_stats.sample([manifest["model_path"], output])),
                        "logical_bytes": logical_bytes(manifest, output, args.route)})

            observation = dispatch(output, request, original, deadline, args.poll_ms / 1000, on_token)
            print(f"request {index}: first token {observation['first']['token']}", flush=True)
            observations.append({**observation, "start_ns": started})
        control.write(output / "observations.json", observations)
        if stop_worker(output, run["run_id"], original, process, deadline, args.poll_ms / 1000):
            raise RuntimeError("Worker exited nonzero")
        original, process, sampler.pid = None, None, None
        if control.read(output / "audit-after.json") != control.read(output / "prepared-audit.json"):
            raise ValueError("Immutable model state changed")
        if args.route == "snapshot":
            snapshot = control.read(output / "snapshot/manifest.json")
            if control.inventory(output / "snapshot") != snapshot["payload"]:
                raise ValueError("Restore changed immutable image payload")
            details.update(image_hashes_match=True, warnings=session.warnings(output),
                           image_bytes=sum(item["bytes"] for item in snapshot["payload"].values()))
        sampler.close()
        sampler = None
        names = ["reference.json", "tokens.json", "prepared-audit.json", "audit-before.json",
                 "audit-after.json", "observations.json", "memory.json", "resources.jsonl",
                 "loading.json", "io.json", "compile-cache.json"]
        if args.route == "snapshot":
            names.append("kv-release.json")
        if args.data_cache == "cold":
            names.append("cold-cache.json")
        record = {"status": "passed", "cleanup": "complete", "route": args.route, "kind": args.kind,
                  "run_id": run["run_id"], "block": args.block, "data_cache": args.data_cache,
                  "compile_cache": args.compile_cache, "cold_cache": cold_cache_rows,
                  "activation": {"start_ns": observations[0]["start_ns"]},
                  "kv_release_ns": details.get("kv_release_ns"),
                  "capture_start_ns": details.get("capture_start_ns"),
                  "lifecycle": {name: details.get(name) for name in
                                ("original_reaped", "capture_exited", "restore_exited", "image_hashes_match")},
                  "observations": observations, "details": details,
                  "run_sha256": measure.file_hash(output / "run.json"),
                  "evidence": {name: measure.file_hash(output / name) for name in names}}
        record["durations"] = measure.validate_vllm(record, control.read(output / "reference.json"))
        control.write(output / "result.json", record)
        measure.vllm_run(output)
        print(json.dumps(record["durations"]), flush=True)
    except BaseException as error:
        failure = {"status": "failed", "phase": phase, "reason": f"{type(error).__name__}: {error}"}
        if isinstance(error, lifecycle.CleanupError):
            failure["cleanup"] = "failed"
        raise
    finally:
        try:
            lifecycle.cleanup(output, original, process)
            if sampler:
                sampler.close()
        except BaseException as error:
            failure = {**(failure or {"status": "failed", "phase": "cleanup"}),
                       "cleanup": "failed", "cleanup_reason": str(error)}
            control.write(output / "result.json", failure)
            raise
        if failure:
            control.write(output / "result.json", {**failure, "cleanup": failure.get("cleanup", "complete")})


def logical_bytes(manifest, output, route):
    """What the checks actually read, as opposed to what the device counters saw."""
    counted = {"model_files": sum((Path(manifest["model_path"]) / name).stat().st_size
                                  for name in manifest["identity"]["files"])}
    if route == "snapshot":
        payload = control.read(output / "snapshot/manifest.json")["payload"]
        counted["payload"] = sum(item["bytes"] for item in payload.values())
    return counted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("trial",))
    for name in ("assets", "output", "tools"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route", choices=measure.VLLM_ROUTES, required=True)
    parser.add_argument("--kind", choices=("diagnostic", "timing"), required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--validated-run", type=Path)
    parser.add_argument("--block", type=int, choices=(1, 2, 3))
    parser.add_argument("--poll-ms", type=float, default=5)
    parser.add_argument("--sample-ms", type=float, default=100)
    parser.add_argument("--data-cache", choices=("uncontrolled", "cold"), default="uncontrolled")
    parser.add_argument("--compile-cache", choices=("warm", "cold"), default="warm",
                        help="warm keeps vLLM's compiled kernels on disk, as a production restart would")
    args = parser.parse_args()
    if args.kind == "timing" and (not args.validated_run or not args.block):
        parser.error("Timing requires --validated-run and --block")
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.poll_ms, args.sample_ms)):
        parser.error("Deadlines and intervals must be positive")
    trial(args)
