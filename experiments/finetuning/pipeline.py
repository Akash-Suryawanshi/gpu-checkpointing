"""Shared lifecycle steps for the three explicit fine-tuning pipelines.

The controller starts and supervises the trainer process: a running copy of
train.py. Empty marker files form a handshake: one process creates a file to
announce readiness or grant permission, and the other waits for that filename.
These files carry no model state. Keep route-specific save/dump/restore decisions
in the pipeline files so each experiment reads from top to bottom.
"""

from argparse import Namespace
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import time

import state

ROOT = Path(__file__).resolve().parents[2]
# Keep Linux process/CRIU command details in their dedicated session module.
sys.path.insert(0, str(ROOT / "experiments/criu"))
import session


@dataclass
class Trial:
    """Controller data shared with one pipeline, including its live child.

    A PID is Linux's numeric process identifier; Popen is Python's handle for a
    process it started. CRIU-restored trainers have a PID but no Popen handle.
    Update these fields as the active trainer changes: the outer controller
    needs them for memory sampling and cleanup even when a pipeline raises.
    ``directory`` identifies that trainer's files. No model or optimizer lives
    here; ``env`` contains the environment variables passed to child processes.
    """

    args: Namespace
    tools: dict
    env: dict
    events: list = field(default_factory=list)
    pid: int | None = None
    process: subprocess.Popen | None = None
    directory: Path | None = None


def event(trial, name, **values):
    """Record an event using the controller's elapsed-time clock, in nanoseconds."""
    row = {"event": name, "monotonic_ns": time.monotonic_ns(), **values}
    trial.events.append(row)
    print(json.dumps(row), flush=True)


def launch(trial, directory, pause=(), extra=()):
    """Start a trainer and retain its identity before waiting on any markers."""
    args = trial.args
    arguments = ["--assets", args.assets, "--run-dir", directory,
                 "--dropout", args.dropout, "--until", args.until]
    if pause:
        arguments += ["--pause-at", ",".join(map(str, pause))]
    if args.timing:
        arguments += ["--timing"]
    # The directory also identifies the child during guarded failure cleanup.
    trial.directory = directory
    trial.process = session.launch(sys.executable, ROOT / "experiments/finetuning/train.py",
                                   [*arguments, *extra], directory, trial.env)
    trial.pid = trial.process.pid


def capture_boundary(trial, generation):
    """Wait for a complete update; check its reference before requesting capture.

    A generation is identified by its completed update number. A diagnostic
    trial compares this boundary and later compares the restored
    boundary before permitting another update. A timing trial skips these large
    inspections only after run.py has verified matching correctness evidence.
    """
    args = trial.args
    session.wait_marker(args.run_dir / f"ready-{generation}", trial.pid, 300)
    expected = None
    if not args.timing:
        expected = json.loads((args.reference / f"state-{generation}.json").read_text())
        state.compare(expected, json.loads((args.run_dir / f"state-{generation}.json").read_text()))
    event(trial, "capture_requested", generation=generation)
    return expected


def handoff(trial, generation, require_clean_exit=False):
    """Verify exit, observe an empty GPU, sync images, then exercise job B.

    A disappearance during CUDA staging can be temporary, so GPU availability
    is observed only after the original has been reaped: the controller waits
    for its exit and Linux removes its process entry. Job B independently
    allocates GPU memory and computes a result. Its time is recorded separately
    from capture and restore. ``sync -f`` flushes pending writes on this local
    filesystem; it does not copy the snapshot to another host or remote store.
    """
    output = trial.args.run_dir
    code = session.reap(trial.pid, trial.process)
    # Application saving exits normally; CRIU ends its original during dump,
    # so that route requires verified disappearance without requiring exit 0.
    if require_clean_exit and code != 0:
        raise RuntimeError("Application save process failed")
    event(trial, "original_exit_verified", generation=generation, pid=trial.pid)
    # CRIU restores the same numeric PID. Retain it for cleanup if restore fails
    # after creating that child, but discard the original Popen handle.
    trial.process = None
    event(trial, "gpu_observation_started", generation=generation)
    gpu_log = output / f"gpu-after-exit-{generation}.csv"
    session.command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                    gpu_log, trial.env)
    if gpu_log.read_text().strip():
        raise RuntimeError("GPU compute processes remain after original exit")
    event(trial, "gpu_observed_after_exit", generation=generation)
    session.command(["sync", "-f", output], output / f"sync-{generation}.log", trial.env)
    event(trial, "filesystem_synced", generation=generation)
    event(trial, "job_b_requested", generation=generation)
    # A separate process allocates GPU memory and checks a reduction, giving
    # functional evidence that another job can actually use CUDA after exit.
    session.command([sys.executable, ROOT / "experiments/finetuning/job_b.py"],
                    output / f"job-b-{generation}.log", trial.env, timeout=60)
    event(trial, "job_b_completed", generation=generation)
    event(trial, "restore_requested", generation=generation)


def inspect_restore(trial, generation, expected, *, request_inspection=False):
    """Release a diagnostic trainer only after untouched restore state matches.

    Each pipeline requests inspection in its own way. This common half waits
    for the observation, then compares reference → before → after through
    session.release_verified. A mismatch never creates the continue marker.

    First record restore return and diagnostic memory mappings. CRIU also needs
    an inspection request; application loading starts inspection itself. Timing
    skips diagnostics but still releases CRIU's saved wait.
    """
    output = trial.args.run_dir
    event(trial, "restore_returned", generation=generation, pid=trial.pid)
    if not trial.args.timing:
        # Linux exposes this process's mapped memory ranges for later diagnosis.
        (output / f"restored-{generation}.maps").write_text(Path(f"/proc/{trial.pid}/maps").read_text())
    if request_inspection:
        (output / f"inspect-{generation}").touch(exist_ok=False)
    if not trial.args.timing:
        session.wait_marker(output / f"inspected-{generation}", trial.pid)
        session.release_verified(output, generation, expected)
        event(trial, "continuation_permitted", generation=generation)


def finish(trial, record_exit=True):
    """Require both the trainer's completion marker and a successful exit."""
    session.wait_marker(trial.directory / "done", trial.pid, 300)
    if session.reap(trial.pid, trial.process) != 0:
        raise RuntimeError("Trainer did not exit successfully")
    if record_exit:
        event(trial, "final_exit_verified", pid=trial.pid)
    # The outer finally block must no longer treat this reaped PID as owned.
    trial.pid, trial.process = None, None


def compare_run(reference, actual, until, timing=False):
    """Compare boundary fingerprints and every loss without tolerance.

    Diagnostic runs save every boundary, including initialization. Timing runs
    save only their final fingerprint outside the measured next-update endpoint;
    all losses are still compared in their original update order.
    """
    for update in ([until] if timing else range(until + 1)):
        state.compare(json.loads((reference / f"state-{update}.json").read_text()),
                      json.loads((actual / f"state-{update}.json").read_text()))
    expected = [json.loads(s)["loss"] for s in (reference / "updates.jsonl").read_text().splitlines()]
    observed = [json.loads(s)["loss"] for s in (actual / "updates.jsonl").read_text().splitlines()]
    state.compare(expected, observed, "losses")
