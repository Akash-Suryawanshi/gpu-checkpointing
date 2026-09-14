"""Acceptance checks for the observer, without pretending to exercise CRIU."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from verify_cpu import read_events, verify


def events(paused: bool) -> list[dict]:
    result = [{"event": "started", "step": 0, "offset": 0, "token": "original-memory"}]
    for step in range(1, 26):
        state = {"step": step, "offset": step * 5, "token": "original-memory"}
        result.append({"event": "step", **state})
        if paused and step == 20:
            result.extend({"event": event, **state} for event in ("before", "after"))
    result.append({"event": "done", **state})
    return result


class EvidenceTests(unittest.TestCase):
    def test_live_counter_waits_and_continues_without_loading_state(self) -> None:
        """AC3: readiness fixes step/file position until release; this is not a CRIU test."""
        script = Path(__file__).resolve().parents[1] / "cpu_counter.py"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "input.txt").write_bytes(b"".join(f"{step:04d}\n".encode() for step in range(1, 26)))
            with (run_dir / "baseline.jsonl").open("w") as output:
                subprocess.run([sys.executable, str(script), "--run-dir", directory, "--mode", "baseline"], stdout=output, check=True, timeout=10)
            with (run_dir / "process.jsonl").open("w") as output:
                process = subprocess.Popen([sys.executable, str(script), "--run-dir", directory, "--mode", "pause"], stdout=output, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 10
                    while not (run_dir / "ready").exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("Counter did not become ready")
                        time.sleep(0.01)
                    before_release = read_events(run_dir / "process.jsonl")
                    self.assertEqual(before_release[-1]["step"], 20)
                    self.assertEqual(before_release[-1]["offset"], 100)
                    self.assertIsNone(process.poll())
                    (run_dir / "release").touch()
                    _, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, stderr.decode())
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()
            self.assertTrue(verify(read_events(run_dir / "baseline.jsonl"), read_events(run_dir / "process.jsonl"))["state_matches"])

    def test_continuation_and_restarted_process_are_distinguished(self) -> None:
        """AC1: accept continuous state; reject a fresh process or repeated training."""
        baseline, restored = events(False), events(True)
        self.assertTrue(verify(baseline, restored)["state_matches"])
        restarted = copy.deepcopy(restored)
        for event in restarted[22:]:
            event["token"] = "new-process-memory"
        with self.assertRaisesRegex(ValueError, "token changed"):
            verify(baseline, restarted)
        with self.assertRaisesRegex(ValueError, "skipped/repeated"):
            verify(baseline, restored[:22] + restored[1:])

    def test_wrong_restored_file_position_is_rejected(self) -> None:
        """AC2: a matching counter alone cannot hide a lost open-file position."""
        restored = events(True)
        next(event for event in restored if event["event"] == "after")["offset"] = 0
        with self.assertRaisesRegex(ValueError, "offset"):
            verify(events(False), restored)


if __name__ == "__main__":
    unittest.main()
