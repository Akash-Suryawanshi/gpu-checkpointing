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
    print(json.dumps({"event": event, **state}), flush=True)


def run(run_dir: Path, mode: str) -> None:
    token = secrets.token_hex(16)
    state: dict[str, int | str] = {"pid": os.getpid(), "token": token, "step": 0, "offset": 0}
    emit("started", state)
    # Unbuffered reads make tell() the actual Linux file position, not a read-ahead offset.
    with (run_dir / "input.txt").open("rb", buffering=0) as source:
        for step in range(1, FINAL_STEP + 1):
            record = source.read(RECORD_BYTES)
            if record != f"{step:04d}\n".encode():
                raise RuntimeError(f"Wrong input at step {step}: {record!r}")
            state.update(step=step, offset=source.tell())
            emit("step", state)
            if step == PAUSE_STEP and mode == "pause":
                emit("before", state)
                (run_dir / "ready").touch(exist_ok=False)
                # A fresh run directory prevents a release from an older experiment.
                # Timeout is supervised externally: a saved clock deadline would age while paused.
                while not (run_dir / "release").exists():
                    time.sleep(0.05)
                state.update(pid=os.getpid(), offset=source.tell())
                emit("after", state)
    emit("done", state)
    if mode == "pause":
        (run_dir / "done").touch(exist_ok=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "pause"), required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.mode)


if __name__ == "__main__":
    main()
