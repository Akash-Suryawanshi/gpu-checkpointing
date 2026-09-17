"""Validate inputs, select a dedicated pipeline, and retain success/failure evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from prepare import environment, file_hash, write_json
from pipeline import ROOT, Trial, compare_run, event, session
import application_pipeline
import criu_pipeline
import reference_pipeline
import state
import metrics


def reference_key(args):
    """Fingerprint inputs, environment, all trainer/controller sources, and tools.

    Code changes invalidate references. Tool commits identify source revisions;
    binary hashes identify the programs/plugin actually used.
    """
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


def run(args):
    """Validate before launch; record results and clean up the owned child on failure.

    Timing requires matching passed diagnostics. Trial tracks the current trainer
    so cleanup can find it even if a pipeline raises.
    """
    # Images contain process memory: keep files private and reject stale run markers.
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
    # Admit only references whose two independent runs matched exactly.
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
        # Check large asset hashes here, outside the trainer's measured interval.
        manifest = json.loads((args.assets / "manifest.json").read_text())
        for name, digest in key["identity"]["files"].items():
            state.compare(digest, file_hash(Path(manifest["model_path"]) / name), f"asset.{name}")
    # Adopt CRIU's detached trainer so this controller can collect its exit status.
    session.adopt_restored_children()
    trial = Trial(args, tools, env)
    result = {"mode": args.mode, "timing": args.timing, "capture": args.capture,
              "verification_scope": "final_state_and_all_losses" if args.timing else "all_update_boundaries_and_losses",
              "lifecycle_passed": False, "numerical_passed": False,
              "unqualified_compatibility": False, "events": trial.events}
    # Sampling follows changes of PID across application relaunch or CRIU restore.
    stop_sampling = metrics.sample_memory(lambda: trial.pid)

    try:
        # Dedicated files keep each route readable in execution order.
        pipelines = {"reference": reference_pipeline.run,
                     "application": application_pipeline.run, "criu": criu_pipeline.run}
        pipelines[args.mode](trial)
        result["lifecycle_passed"] = True
        if args.mode == "reference":
            # Publish admission only after numerical agreement, not just clean exits.
            compare_run(output, output / "repeat", args.until)
            (output / "verified").touch(exist_ok=False)
        else:
            compare_run(args.reference, output, args.until, args.timing)
        result["numerical_passed"] = True
        event(trial, "verification_passed")
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["memory_samples"] = stop_sampling()
        if trial.pid:
            session.cleanup(trial.pid, trial.directory, trial.process)
        # Equal state does not resolve resource-sharing warnings or future-host support.
        result["warnings"] = session.warnings(output)
        result["compatibility"] = "qualified: sharing ownership and any resource warnings require interpretation"
        result["image_bytes"] = sum(p.stat().st_size for p in output.glob("images-*/*.img"))
        result["application_bytes"] = (output / "application.pt").stat().st_size if (output / "application.pt").exists() else 0
        if trial.events:
            result["trial_seconds"] = (time.monotonic_ns() - trial.events[0]["monotonic_ns"]) / 1e9
        # Convert restored clock offsets before comparing timestamps; retain raw events.
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
    # Reject ambiguous or impossible boundaries before creating any run files.
    if args.mode != "reference" and (args.capture != sorted(set(args.capture)) or any(n < 1 or n >= args.until for n in args.capture)):
        parser.error("Capture updates must be increasing, unique, and before the final update")
    if args.mode == "application" and len(args.capture) != 1:
        parser.error("Application comparison uses one capture")
    if args.timing and args.mode == "reference":
        parser.error("Timing applies to the application and CRIU comparisons")
    run(args)
