"""Capture an existing external-control trainer and publish one local snapshot."""

import argparse
import os
from pathlib import Path
import shutil
import time

import control
from control import session


def capture(args):
    os.umask(0o077)
    run, snapshot = args.run_dir.resolve(), args.snapshot.resolve()
    deadline = time.monotonic() + args.timeout
    control.wait(run / "job.json", deadline)
    with control.lock(run, deadline):
        job = control.read(run / "job.json")
        previous = control.read(run / "control/phase.json")
        if previous["status"] not in ("idle", "restored", "cancelled"):
            raise ValueError("Job has an unfinished or ambiguous operation")
        if not session.matches(job["identity"]):
            raise ValueError("Registered trainer is not live")
        if not 1 <= args.at < job["until"]:
            raise ValueError("Capture target must precede final update")
        if snapshot.exists() or snapshot.is_symlink():
            raise ValueError("Snapshot path must be fresh")
        capture_id, attempt = control.attempt(run)
        emit = lambda name, **values: control.event(attempt, name, capture_id=capture_id, **values)
        request = {"capture_id": capture_id, "job_id": job["job_id"], "at": args.at,
                   "deadline": deadline - control.monotonic_offset()}
        dumped = False
        try:
            control.phase(run, "requested", capture_id=capture_id)
            control.write(run / "control/request.json", request)
            emit("capture_requested")
            ready = control.wait(attempt / "ready.json", deadline, job["identity"])
            if ready["capture_id"] != capture_id or ready["identity"] != job["identity"]:
                raise ValueError("Pause acknowledgement identity mismatch")
            emit("ready", generation=ready["update"])
            # Validate storage and dependencies while the trainer is safely paused.
            snapshot.mkdir(mode=0o700, parents=True)
            volume = control.storage(snapshot)
            memory = control.read(run / "memory.json")
            # A conservative admission floor, not a promise about image size.
            required = 2 * (int(memory["VmRSS"].split()[0]) * 1024 + memory["reserved_vram"])
            if volume["free"] < required:
                raise ValueError(f"Insufficient snapshot capacity: need at least {required} free bytes")
            volume["admission_floor_bytes"] = required
            old = previous.get("snapshot")
            old_volume = control.storage(Path(old)) if old else None
            deps = control.dependencies(job, args.tools, deadline)
            shutil.copyfile(attempt / "before.json", snapshot / "before.json")
            tools = control.tools_config(args.tools)
            env = session.child_environment(args.tools / "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint")
            if not session.matches(job["identity"]):
                raise ValueError("Trainer identity changed before dump")
            control.remaining(deadline)
            control.phase(run, "dumping", capture_id=capture_id)
            dumped = True  # Any error after arming must not automatically resume.
            emit("dump_started")
            session.criu("dump", run, capture_id, tools, env, job["identity"]["pid"],
                         images=snapshot / "images", attempt=attempt, timeout=control.remaining(deadline))
            emit("dump_completed")
            session.observe_exit(job["identity"], deadline, removed=True)
            emit("original_exit_verified")
            session.command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                            attempt / "gpu-after-exit.csv", env, timeout=control.remaining(deadline))
            if (attempt / "gpu-after-exit.csv").read_text().strip():
                raise RuntimeError("GPU compute processes remain after original exit")
            emit("gpu_observed_after_exit")
            manifest = {"schema": control.SCHEMA, "capture_id": capture_id, "job": job,
                        "run": str(run), "request": request, "update": ready["update"],
                        "dependencies": deps, "pre_staging_memory": memory, "storage": volume, "previous_storage": old_volume,
                        "external_files": {name: control.file_hash(run / name)
                                           for name in ("updates.jsonl", "trainer.stderr")},
                        "dump_log_sha256": control.file_hash(attempt / "dump.log")}
            control.publish(snapshot, manifest, run, deadline, emit)
            return manifest
        except BaseException:
            if not dumped:
                control.phase(run, "cancelled", capture_id=capture_id)
            else:
                # Preserve enough phase evidence to reject any ambiguous retry.
                try:
                    control.phase(run, "failed", capture_id=capture_id, snapshot=str(snapshot))
                except OSError:
                    pass
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--at", type=int, default=1, help="Earliest completed update to acknowledge")
    parser.add_argument("--timeout", type=float, default=300)
    capture(parser.parse_args())
