"""Supervise one four-route inference trial using the runbook command contract."""

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

import cold_cache
import lifecycle
import measure
import park
from prepare_assets import validate_assets
from worker import control
from control import session

HERE = Path(__file__).resolve().parent


def trial(args):
    output, assets, tools = args.output.resolve(), args.assets.resolve(), args.tools.resolve()
    staged_restore = args.disk_action == "restore"
    if not staged_restore:
        output.mkdir(mode=0o700, parents=True)
    deadline = time.monotonic() + args.timeout
    helper = tools / "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint"
    env = session.child_environment(helper)
    process, original, sampler = None, None, None
    details, phase, failure = {}, "admission", None
    emit = lambda name, **values: control.event(output, name, **values)
    session.adopt_restored_children()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        manifest = validate_assets(assets)
        deps = control.dependencies({"assets": str(assets), "python": sys.executable}, tools, deadline)
        bundle = None
        if args.bundle:
            from stream_weights import metadata
            if metadata(args.bundle)["assets_sha256"] != control.file_hash(assets / "manifest.json"):
                raise ValueError("Packed weights belong to different prepared assets")
            bundle = {"path": str(args.bundle.resolve()), "sha256": control.file_hash(args.bundle / "manifest.json")}
        key = {"dependencies": deps, "assets": manifest, "poll_ms": args.poll_ms,
               "loader": args.loader, "bundle": bundle,
               "validation_order": args.validation_order, "validation_workers": args.validation_workers,
               "data_cache": args.data_cache,
               "sample_ms": args.sample_ms, "inspection_policy": "full-diagnostic-post-response-timing-v1",
               "cache_paths": {k: env[k] for k in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH") if k in env}}
        diagnostic = None
        if args.kind == "timing":
            diagnostic = measure.diagnostic_record(args.validated_run, key, args.route)
        if staged_restore:
            run = control.read(output / "run.json")
            if run["key"] != key or run["route"] != args.route or run["kind"] != args.kind:
                raise ValueError("Staged restore settings changed")
            captured = control.read(output / "capture-result.json")
            if captured["status"] != "captured" or captured["cleanup"] != "complete":
                raise ValueError("Incomplete capture")
            details.update(captured["details"])
        else:
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
        def wait(name, identity=None):
            return control.wait(output / name, deadline, identity, args.poll_ms / 1000)
        def admit(action, memory, identity=None):
            observed = park.resources(deadline, identity["pid"] if identity else None)
            record = {"observed": observed, "memory": memory, "action": action}
            try:
                record["requirements"] = park.admission(observed, memory, action, identity["pid"] if identity else None)
            except Exception as error:
                record["refusal"] = str(error)
                raise
            finally:
                control.write(output / f"admission-{action}.json", record)
            return observed
        def reuse(before):
            after = park.resources(deadline, original["pid"] if original and session.matches(original) else None)
            park.admission(after, {"reserved_vram": 0, "rss_bytes": 0}, "reuse",
                           original["pid"] if original else None)
            mib, claim = park.job_size(before["gpu_free"], after["gpu_free"])
            print(f"GPU memory released: {(before['gpu_used'] - after['gpu_used']) / park.GIB:.2f} GiB; "
                  f"host RSS retained: {after.get('rss_bytes', 0) / park.GIB:.2f} GiB", flush=True)
            control.write(output / "released-resources.json", after)
            emit("job_b_started", mib=mib, claim=claim)
            # Keep ownership until exit is collected, including cancellation.
            with lifecycle.helper([sys.executable, HERE.parent / "finetuning/job_b.py", "--mib", str(mib)],
                                  output / "job-b.log", env, deadline, privileged=False) as command:
                lifecycle.wait_helper(command, deadline)
            checked = json.loads((output / "job-b.log").read_text())
            if not checked["passed"] or checked["sum"] != mib * park.MIB:
                raise ValueError("Job B result differs")
            control.write(output / "reuse.json", {**checked, "claim": claim, "exited": True})
            print(f"job B: {mib} MiB, {claim}, calculation passed, process exited", flush=True)
        def cold_start(images=()):
            if args.data_cache == "cold":
                files = [Path(manifest["model_path"]) / name for name in manifest["identity"]["files"]
                         if name.endswith(".safetensors")]
                if args.bundle:
                    files.append(args.bundle / "weights.bin")
                control.write(output / "cold-cache.json", cold_cache.evict([*files, *images]))
                emit("data_cache_evicted")
        if not staged_restore:
            phase = "launch"
            if args.route == "fresh":
                cold_start()
            started = time.monotonic_ns()
            process = session.launch(sys.executable, HERE / "worker.py", ["--assets", assets,
                "--run-dir", output, "--route", args.route, "--kind", args.kind, "--run-id", run["run_id"]], output, env)
            original = session.identity(process.pid)
            sampler.pid = process.pid
            emit("worker_launched", identity=original)
            print(f"{args.route}: worker identity {json.dumps(original)}, waiting for idle", flush=True)
            wait("idle.json", original)
            memory = control.read(output / "memory.json")
            observed = admit("loaded", memory, original)
            details["preparation_seconds"] = (time.monotonic_ns() - started) / 1e9
            if args.route == "fresh":
                # Fresh's first full audit occurs after response one, never during load.
                shutil.copyfile(output / "prepared-audit.json", output / "audit-before.json")
            elif control.read(output / "audit-before.json") != control.read(output / "prepared-audit.json"):
                raise ValueError("Warmup changed immutable model state")
            if args.route in ("ram", "disk"):
                phase = "park" if args.route == "ram" else "capture"
                observed = admit(phase, memory, original)
                saved_at = time.monotonic_ns()
                if args.route == "ram":
                    parking = park.Parking(helper, original, deadline, emit)
                    parking.park()
                    print("RAM: checkpointed; GPU state retained in host RAM", flush=True)
                else:
                    details.update(lifecycle.capture(output, tools, env, process, deadline))
                    process, original = None, None
                    sampler.pid = None
                    print("disk: published; original process reaped; capture command exited", flush=True)
                details["save_seconds"] = (time.monotonic_ns() - saved_at) / 1e9
                reuse(observed)
                if args.disk_action == "capture":
                    control.write(output / "capture-result.json", {"status": "captured", "cleanup": "complete", "details": details})
                    return
                if args.route == "ram":
                    phase = "wake"
                    # Host RAM already contains staged bytes: only GPU admission here.
                    admit("wake", memory, original)
                    started = time.monotonic_ns()
                    parking.wake()
                    control.write(output / "wake.json", {"run_id": run["run_id"]})
                    restored = wait("wake-inspection.json", original)
                    expected = control.read(output / "audit-before.json")
                    if args.kind == "timing":
                        expected.pop("fingerprints")
                    if restored != expected:
                        raise ValueError("RAM inspection differs")
                    print("RAM: running; wake inspection matched", flush=True)
        if args.route == "disk":
            phase = "restore"
            memory = control.read(output / "snapshot/manifest.json")["pre_staging_memory"]
            admit("restore", memory)
            cold_start(p for p in (output / "snapshot/images").rglob("*") if p.is_file())
            started = time.monotonic_ns()
            emit("disk_activation_started", start_ns=started)
            original = lifecycle.restore(output, tools, env, deadline)
            sampler.pid = original["pid"]
            details["restore_exited"] = True
            print(f"disk: restore command exited; worker {original['pid']} survived", flush=True)
        phase = "responses"
        wait("ready.json", original)
        emit("worker_ready_observed")
        observations = []
        for index in (1, 2):
            request = {"run_id": run["run_id"], "request_id": uuid.uuid4().hex,
                       "input_ids": control.read(output / "tokens.json")["benchmark"]["input_ids"]}
            request_start = time.monotonic_ns()
            if index == 2 or args.route == "resident":
                started = request_start
            control.write(output / f"request-{index}.json", request)
            published = time.monotonic_ns()
            first = wait(f"first-{index}.json", original)
            if any(first[k] != request[k] for k in ("run_id", "request_id")):
                raise ValueError("First token identity mismatch")
            received = time.monotonic_ns()
            emit("first_token_received", request_id=request["request_id"], received_ns=received)
            print(f"request {index}: received first token {first['token']}", flush=True)
            response = wait(f"response-{index}.json", original)
            observations.append({"request": request, "first": first, "response": response, "start_ns": started,
                "request_start_ns": request_start, "published_ns": published,
                "first_ns": received, "completed_ns": time.monotonic_ns()})
            if index == 1:
                wait("audit-after.json", original)
        control.write(output / "observations.json", observations)
        wait("completed.json", original)
        if session.reap(original["pid"], process, timeout=control.remaining(deadline)):
            raise RuntimeError("Worker exited nonzero")
        original, process = None, None
        sampler.pid = None
        if args.route == "disk":
            snapshot = output / "snapshot"
            manifest = control.read(snapshot / "manifest.json")
            if control.inventory(snapshot) != manifest["payload"]:
                raise ValueError("Restore changed immutable image payload")
            details.update(image_hashes_match=True, image_bytes=sum(p["bytes"] for p in manifest["payload"].values()),
                           warnings=session.warnings(output))
        if park.resources(deadline)["compute_pids"]:
            raise RuntimeError("GPU compute process remains")
        sampler.close()
        sampler = None
        names = ["reference.json", "tokens.json", "prepared-audit.json", "audit-before.json", "audit-after.json",
                 "observations.json", "memory.json", "resources.jsonl"]
        if args.route in ("ram", "disk"):
            names += ["reuse.json", "released-resources.json"]
        if args.data_cache == "cold":
            names += ["cold-cache.json"]
        result = {"status": "passed", "cleanup": "complete", "run_sha256": measure.file_hash(output / "run.json"),
                  "evidence": {name: measure.file_hash(output / name) for name in names}, **details,
                  "durations": measure.durations(observations, run["run_id"], control.read(output / "reference.json"))}
        control.write(output / "result.json", result)
        measure.validate_run(output)
        print("Full audit and health response passed; worker reaped", flush=True)
        if args.discard_image_after_success:
            shutil.rmtree(output / "snapshot/images")
            control.write(output / "image-discarded.json", {"reason": "accepted disk timing after final hash check"})
        print(json.dumps(result["durations"]), flush=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("trial",))
    for name in ("assets", "output", "tools"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route", choices=measure.ROUTES, required=True)
    parser.add_argument("--kind", choices=("diagnostic", "timing"), required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--validated-run", type=Path)
    parser.add_argument("--block", type=int, choices=(1, 2, 3))
    parser.add_argument("--disk-action", choices=("full", "capture", "restore"), default="full")
    parser.add_argument("--discard-image-after-success", action="store_true")
    parser.add_argument("--poll-ms", type=float, default=5)
    parser.add_argument("--sample-ms", type=float, default=100)
    parser.add_argument("--validation-order", choices=("payload-first", "dependencies-first"), default="payload-first")
    parser.add_argument("--validation-workers", type=int, choices=(1, 4), default=1)
    parser.add_argument("--data-cache", choices=("uncontrolled", "cold"), default="uncontrolled")
    parser.add_argument("--loader", choices=("transformers", "packed", "pipelined"), default="transformers")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.kind == "timing" and (not args.validated_run or not args.block):
        parser.error("Timing requires --validated-run and --block")
    if args.disk_action != "full" and (args.route != "disk" or args.kind != "diagnostic"):
        parser.error("Staged actions require disk diagnostics")
    if args.discard_image_after_success and (args.route != "disk" or args.kind != "timing"):
        parser.error("Image discard requires accepted disk timing")
    if args.data_cache == "cold" and (args.route not in ("fresh", "disk") or args.disk_action == "capture"):
        parser.error("Cold data-cache measurement requires fresh or a disk restore")
    if (args.loader != "transformers") != bool(args.bundle) or (args.bundle and args.route != "fresh"):
        parser.error("Packed/pipelined loaders require --bundle and --route fresh")
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.poll_ms, args.sample_ms)):
        parser.error("Deadlines and intervals must be positive")
    trial(args)
