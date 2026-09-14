"""Compare CPU workload evidence; process lifecycle is checked by checkpoint.sh."""

import argparse
import json
from pathlib import Path

from cpu_counter import FINAL_STEP, PAUSE_STEP, RECORD_BYTES


def read_events(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def verify(baseline: list[dict], restored: list[dict]) -> dict[str, int | bool]:
    expected_steps = [(step, step * RECORD_BYTES) for step in range(1, FINAL_STEP + 1)]
    for name, events in (("baseline", baseline), ("restored", restored)):
        steps = [(event["step"], event["offset"]) for event in events if event["event"] == "step"]
        if steps != expected_steps:
            raise ValueError(f"{name}: skipped/repeated step or wrong file position")
        expected_events = ["started"] + ["step"] * FINAL_STEP + ["done"]
        if name == "restored":
            expected_events[PAUSE_STEP + 1:PAUSE_STEP + 1] = ["before", "after"]
        if [event["event"] for event in events] != expected_events:
            raise ValueError(f"{name}: unexpected lifecycle; possibly restarted from the beginning")
        token = events[0]["token"]
        if not token or any(event["token"] != token for event in events):
            raise ValueError(f"{name}: in-memory token changed")
        if events[-1]["step"] != FINAL_STEP or events[-1]["offset"] != FINAL_STEP * RECORD_BYTES:
            raise ValueError(f"{name}: wrong final state")
    before = next(event for event in restored if event["event"] == "before")
    after = next(event for event in restored if event["event"] == "after")
    for key in ("step", "offset", "token"):
        if before[key] != after[key]:
            raise ValueError(f"State changed across pause: {key}")
    if before["step"] != PAUSE_STEP or before["offset"] != PAUSE_STEP * RECORD_BYTES:
        raise ValueError("Wrong checkpoint boundary")
    return {"state_matches": True, "resumed_after_step": PAUSE_STEP, "final_step": FINAL_STEP}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify(read_events(args.run_dir / "baseline.jsonl"), read_events(args.run_dir / "process.jsonl"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
