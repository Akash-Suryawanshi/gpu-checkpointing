"""Compare uninterrupted training, application restart, and CRIU process restoration."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare import environment, file_hash, write_json
import state
import metrics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/criu"))
import session


def reference_key(args):
    manifest = json.loads((args.assets / "manifest.json").read_text())
    state.compare(manifest["environment"], environment(), "environment")
    sources = list((ROOT / "experiments/finetuning").glob("*.py")) + [ROOT / "experiments/criu/session.py"]
    tool_commits = {"criu-4.2.1": "9539417f3e3cfa4eb84c319cd71f4d52f1f08645",
                    "cuda-checkpoint": "00d5cce84c628088d6caa203fc4af40c1538b6f7"}
    for name, commit in tool_commits.items():
        observed = subprocess.check_output(["git", "-C", args.tools / name, "rev-parse", "HEAD"], text=True).strip()
        state.compare(commit, observed, f"tool.{name}.commit")
    binaries = ("criu-4.2.1/criu/criu", "criu-4.2.1/plugins/cuda/cuda_plugin.so",
                "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint")
    return {"environment": manifest["environment"], "identity": manifest["identity"],
            "config": manifest["config"], "dropout": args.dropout, "until": args.until,
            "tool_commits": tool_commits, "tool_sha256": {p: file_hash(args.tools / p) for p in binaries},
            "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(sources)}}


def compare_run(reference, actual, until, timing=False):
    for update in ([until] if timing else range(until + 1)):
        state.compare(json.loads((reference / f"state-{update}.json").read_text()),
                      json.loads((actual / f"state-{update}.json").read_text()))
    expected = [json.loads(s)["loss"] for s in (reference / "updates.jsonl").read_text().splitlines()]
    observed = [json.loads(s)["loss"] for s in (actual / "updates.jsonl").read_text().splitlines()]
    state.compare(expected, observed, "losses")


def run(args):
    os.umask(0o077)
    args.run_dir = args.run_dir.resolve()
    args.assets = args.assets.resolve()
    args.tools = args.tools.resolve()
    args.run_dir.mkdir(parents=True)
    output = args.run_dir
    tools = {"criu": str(args.tools / "criu-4.2.1/criu/criu"),
             "plugin": str(args.tools / "criu-4.2.1/plugins/cuda"),
             "libraries": str(args.tools / "criu-deps/usr/lib/x86_64-linux-gnu")}
    env = session.child_environment(args.tools / "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint")
    key = reference_key(args)
    write_json(output / "key.json", key)
    if args.mode != "reference":
        if args.reference is None or not (args.reference / "verified").exists():
            raise ValueError("A verified pair of references is required")
        state.compare(json.loads((args.reference / "key.json").read_text()), key, "reference_key")
    if args.timing:
        if args.validated_run is None:
            raise ValueError("Timing requires --validated-run with matching full correctness evidence")
        validated = json.loads((args.validated_run / "result.json").read_text())
        state.compare(json.loads((args.validated_run / "key.json").read_text()), key, "validated_key")
        if (not validated["numerical_passed"] or not validated["lifecycle_passed"]
                or validated.get("timing") or validated["mode"] != args.mode
                or validated["capture"] != args.capture):
            raise ValueError("Matching full correctness run did not pass")
        manifest = json.loads((args.assets / "manifest.json").read_text())
        for name, digest in key["identity"]["files"].items():
            state.compare(digest, file_hash(Path(manifest["model_path"]) / name), f"asset.{name}")
    session.adopt_restored_children()
    events = []
    result = {"mode": args.mode, "timing": args.timing, "capture": args.capture,
              "verification_scope": "final_state_and_all_losses" if args.timing else "all_update_boundaries_and_losses",
              "lifecycle_passed": False, "numerical_passed": False,
              "unqualified_compatibility": False, "events": events}
    pid = None
    process = None
    stop_sampling = metrics.sample_memory(lambda: pid)

    def event(name, **values):
        row = {"event": name, "monotonic_ns": time.monotonic_ns(), **values}
        events.append(row)
        print(json.dumps(row), flush=True)

    def launch(directory, pause=(), extra=()):
        arguments = ["--assets", args.assets, "--run-dir", directory,
                     "--dropout", args.dropout, "--until", args.until]
        if pause:
            arguments += ["--pause-at", ",".join(map(str, pause))]
        if args.timing:
            arguments += ["--timing"]
        return session.launch(sys.executable, ROOT / "experiments/finetuning/train.py", [*arguments, *extra], directory, env)

    try:
        checkpoint = output / "application.pt"
        process = launch(output, args.capture if args.mode != "reference" else (),
                         ["--save", checkpoint] if args.mode == "application" else [])
        pid = process.pid
        event("launched", pid=pid)
        if args.mode != "reference":
            for generation in args.capture:
                session.wait_marker(output / f"ready-{generation}", pid, 300)
                expected = None
                if not args.timing:
                    expected = json.loads((args.reference / f"state-{generation}.json").read_text())
                    state.compare(expected, json.loads((output / f"state-{generation}.json").read_text()))
                event("capture_requested", generation=generation)
                if args.mode == "criu":
                    session.criu("dump", output, generation, tools, env, pid)
                    event("dump_completed", generation=generation)
                else:
                    (output / f"save-{generation}").touch(exist_ok=False)
                    session.wait_marker(output / f"saved-{generation}", pid)
                    event("application_save_completed", generation=generation)
                code = session.reap(pid, process)
                if args.mode == "application" and code != 0:
                    raise RuntimeError("Application save process failed")
                event("original_exit_verified", generation=generation, pid=pid)
                # Keep the old PID for failure cleanup: this CRIU route restores that numeric PID.
                process = None
                event("gpu_observation_started", generation=generation)
                session.command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                                output / f"gpu-after-exit-{generation}.csv", env)
                if (output / f"gpu-after-exit-{generation}.csv").read_text().strip():
                    raise RuntimeError("GPU compute processes remain after original exit")
                event("gpu_observed_after_exit", generation=generation)
                session.command(["sync", "-f", output], output / f"sync-{generation}.log", env)
                event("filesystem_synced", generation=generation)
                event("job_b_requested", generation=generation)
                session.command([sys.executable, "-c", "import torch; x=torch.ones(1024*1024,device='cuda'); s=x.sum().item(); assert s==1048576; print(s)"],
                                output / f"job-b-{generation}.log", env, timeout=60)
                event("job_b_completed", generation=generation)
                event("restore_requested", generation=generation)
                if args.mode == "criu":
                    pid = session.criu("restore", output, generation, tools, env)
                else:
                    process = launch(output, extra=["--load", checkpoint])
                    pid = process.pid
                event("restore_returned", generation=generation, pid=pid)
                if not args.timing:
                    (output / f"restored-{generation}.maps").write_text(Path(f"/proc/{pid}/maps").read_text())
                if args.mode == "criu":
                    (output / f"inspect-{generation}").touch(exist_ok=False)
                if not args.timing:
                    session.wait_marker(output / f"inspected-{generation}", pid)
                    session.release_verified(output, generation, expected)
                    event("continuation_permitted", generation=generation)
        session.wait_marker(output / "done", pid, 300)
        if session.reap(pid, process) != 0:
            raise RuntimeError("Trainer did not exit successfully")
        event("final_exit_verified", pid=pid)
        pid, process = None, None
        result["lifecycle_passed"] = True
        if args.mode == "reference":
            repeat = output / "repeat"
            repeat.mkdir()
            process = launch(repeat)
            pid = process.pid
            session.wait_marker(repeat / "done", pid, 300)
            if session.reap(pid, process) != 0:
                raise RuntimeError("Second reference failed")
            pid, process = None, None
            compare_run(output, repeat, args.until)
            (output / "verified").touch(exist_ok=False)
        else:
            compare_run(args.reference, output, args.until, args.timing)
        result["numerical_passed"] = True
        event("verification_passed")
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["memory_samples"] = stop_sampling()
        if pid:
            session.cleanup(pid, output if args.mode != "reference" or not (output / "repeat").exists() else output / "repeat", process)
        result["warnings"] = session.warnings(output)
        result["compatibility"] = "qualified: sharing ownership and any resource warnings require interpretation"
        result["image_bytes"] = sum(p.stat().st_size for p in output.glob("images-*/*.img"))
        result["application_bytes"] = (output / "application.pt").stat().st_size if (output / "application.pt").exists() else 0
        if events:
            result["trial_seconds"] = (time.monotonic_ns() - events[0]["monotonic_ns"]) / 1e9
        metrics.finish(result, output)
        write_json(output / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "application", "criu"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--assets", type=Path, default=ROOT / "runs/finetuning/assets")
    parser.add_argument("--tools", type=Path, default=ROOT / "runs/tools")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--validated-run", type=Path)
    parser.add_argument("--dropout", type=float, choices=(0.0, 0.1), default=0.0)
    parser.add_argument("--until", type=int, choices=(2, 4), default=4)
    parser.add_argument("--capture", type=lambda s: [int(n) for n in s.split(",")], default=[2])
    args = parser.parse_args()
    if args.mode != "reference" and (args.capture != sorted(set(args.capture)) or any(n < 1 or n >= args.until for n in args.capture)):
        parser.error("Capture updates must be increasing, unique, and before the final update")
    if args.mode == "application" and len(args.capture) != 1:
        parser.error("Application comparison uses one capture")
    if args.timing and args.mode == "reference":
        parser.error("Timing applies to the application and CRIU comparisons")
    run(args)
