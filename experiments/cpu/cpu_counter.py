"""Observable CPU workload; deliberately has no checkpoint-loading path."""

import argparse
import json
import os
import secrets
import time
from pathlib import Path

PAUSE_STEP = 20
FINAL_STEP = 25
RECORD_BYTES = 5


def emit(event: str, state: dict[str, int | str]) -> None:
    """Publish one JSON event immediately instead of holding it in an output buffer."""
    print(json.dumps({"event": event, **state}), flush=True)


def run(run_dir: Path, mode: str) -> None:
    """Read fixed-size records and optionally pause with an open file after step 20.

    The random token exists in process memory and appears in observations. This
    program never reads it back: continuation must come from a restored process,
    rather than a custom application checkpoint loader.
    """
    token = secrets.token_hex(16)
    state: dict[str, int | str] = {"pid": os.getpid(), "token": token, "step": 0, "offset": 0}
    emit("started", state)
    # Unbuffered reads make tell() the actual Linux file position, not a read-ahead offset.
    # The file's bytes stay on disk. Its open-file position is separate kernel
    # state that CRIU must reconstruct along with this Python loop's memory.
    with (run_dir / "input.txt").open("rb", buffering=0) as source:
        for step in range(1, FINAL_STEP + 1):
            record = source.read(RECORD_BYTES)
            if record != f"{step:04d}\n".encode():
                raise RuntimeError(f"Wrong input at step {step}: {record!r}")
            state.update(step=step, offset=source.tell())
            emit("step", state)
            if step == PAUSE_STEP and mode == "pause":
                emit("before", state)
                # A marker is an empty file used as a message to the controller;
                # it contains no application state and is not the checkpoint.
                (run_dir / "ready").touch(exist_ok=False)
                # A fresh run directory prevents a release from an older experiment.
                # Timeout is supervised externally: a saved clock deadline would age while paused.
                while not (run_dir / "release").exists():
                    time.sleep(0.05)
                state.update(pid=os.getpid(), offset=source.tell())
                # Re-observe Linux's PID and open-file position after release.
                emit("after", state)
    emit("done", state)
    if mode == "pause":
        (run_dir / "done").touch(exist_ok=False)


def main() -> None:
    """Parse the baseline/pause mode and use an absolute run path for all markers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "pause"), required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.mode)


if __name__ == "__main__":
    main()
