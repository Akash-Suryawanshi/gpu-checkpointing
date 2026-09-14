"""Compare uninterrupted training, application restart, and CRIU process restoration."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from prepare import environment, file_hash, write_json
import state

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/criu"))
import session


def reference_key(args):
    manifest = json.loads((args.assets / "manifest.json").read_text())
    state.compare(manifest["environment"], environment(), "environment")
    sources = list((ROOT / "experiments/finetuning").glob("*.py")) + [ROOT / "experiments/criu/session.py"]
    return {"environment": manifest["environment"], "identity": manifest["identity"],
            "config": manifest["config"], "dropout": args.dropout, "until": args.until,
            "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(sources)}}


def compare_run(reference, actual, until):
    for update in range(until + 1):
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
    session.adopt_restored_children()
    events = []
    result = {"mode": args.mode, "lifecycle_passed": False, "numerical_passed": False,
              "unqualified_compatibility": False, "events": events}
    pid = None
    process = None

    def event(name, **values):
        row = {"event": name, "monotonic_ns": time.monotonic_ns(), **values}
        events.append(row)
        print(json.dumps(row), flush=True)

    def launch(directory, pause=(), extra=()):
        arguments = ["--assets", args.assets, "--run-dir", directory,
                     "--dropout", args.dropout, "--until", args.until]
        if pause:
            arguments += ["--pause-at", ",".join(map(str, pause))]
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
                session.command(["sync", "-f", output], output / f"sync-{generation}.log", env)
                event("filesystem_synced", generation=generation)
                session.command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                                output / f"gpu-after-exit-{generation}.csv", env)
                event("gpu_observed_after_exit", generation=generation)
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
                (output / f"restored-{generation}.maps").write_text(Path(f"/proc/{pid}/maps").read_text())
                if args.mode == "criu":
                    (output / f"inspect-{generation}").touch(exist_ok=False)
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
            compare_run(args.reference, output, args.until)
        result["numerical_passed"] = True
        event("verification_passed")
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        if pid:
            session.cleanup(pid, output if args.mode != "reference" or not (output / "repeat").exists() else output / "repeat", process)
        result["warnings"] = session.warnings(output)
        result["compatibility"] = "qualified: sharing ownership and any resource warnings require interpretation"
        result["image_bytes"] = sum(p.stat().st_size for p in output.glob("images-*/*.img"))
        result["application_bytes"] = (output / "application.pt").stat().st_size if (output / "application.pt").exists() else 0
        if events:
            result["trial_seconds"] = (time.monotonic_ns() - events[0]["monotonic_ns"]) / 1e9
        write_json(output / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "application", "criu"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--assets", type=Path, default=ROOT / "runs/finetuning/assets")
    parser.add_argument("--tools", type=Path, default=ROOT / "runs/tools")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--dropout", type=float, choices=(0.0, 0.1), default=0.0)
    parser.add_argument("--until", type=int, choices=(2, 4), default=4)
    parser.add_argument("--capture", type=lambda s: [int(n) for n in s.split(",")], default=[2])
    args = parser.parse_args()
    if args.mode != "reference" and (args.capture != sorted(set(args.capture)) or any(n < 1 or n >= args.until for n in args.capture)):
        parser.error("Capture updates must be increasing, unique, and before the final update")
    if args.mode == "application" and len(args.capture) != 1:
        parser.error("Application comparison uses one capture")
    run(args)
