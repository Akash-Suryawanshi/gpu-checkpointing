"""Join host-observer events; restored-worker clocks never enter this timeline."""

import argparse
import json
from pathlib import Path


def breakdown(observation, events):
    marks = {row["event"]: row["monotonic_ns"] for row in events}
    ordered = [observation["start_ns"], marks["restore_requested"],
               marks["restore_returned"], observation["first_ns"]]
    if any(type(value) is not int for value in ordered) or ordered != sorted(ordered):
        raise ValueError("Restore events are outside the controller interval")
    names = ("before_criu_seconds", "criu_seconds", "after_criu_to_token_seconds")
    result = dict(zip(names, ((end - start) / 1e9 for start, end in zip(ordered, ordered[1:]))))
    for label in ("payload", "dependency"):
        start = marks.get(label + "_validation_started")
        end = marks.get(label + "_validation_completed")
        if start is not None and end is not None:
            if not ordered[0] <= start <= end <= ordered[1]:
                raise ValueError("Validation events are outside pre-CRIU interval")
            result[label + "_validation_seconds"] = (end - start) / 1e9
    return result


def read_run(run):
    observation = json.loads((run / "observations.json").read_text())[0]
    phase = json.loads((run / "control/phase.json").read_text())
    events = run / "attempts" / phase["attempt_id"] / "events.jsonl"
    return breakdown(observation, [json.loads(line) for line in events.read_text().splitlines()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(read_run(parser.parse_args().run), indent=2))
