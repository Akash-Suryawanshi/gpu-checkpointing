"""Controller safety checks; these CPU tests do not claim to exercise CRIU."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import session


class LifecycleTests(unittest.TestCase):
    def test_continuation_requires_matching_inspection_for_this_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            expected = {"optimizer": {"step": 2}}
            (run / "state-2.json").write_text(json.dumps(expected))
            (run / "after-2.json").write_text(json.dumps({"optimizer": {"step": 3}}))
            with self.assertRaisesRegex(ValueError, "inspection"):
                session.release_verified(run, 2, expected)
            (run / "inspected-2").touch()
            with self.assertRaisesRegex(ValueError, "step"):
                session.release_verified(run, 2, expected)
            self.assertFalse((run / "continue-2").exists())
            (run / "after-2.json").write_text(json.dumps(expected))
            session.release_verified(run, 2, expected)
            self.assertTrue((run / "continue-2").exists())
            with self.assertRaisesRegex(ValueError, "inspection"):
                session.release_verified(run, 3, expected)
            self.assertFalse((run / "continue-3").exists())

    def test_incomplete_dump_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            tools = {"libraries": "unused", "criu": "unused", "plugin": "unused"}
            with patch.object(session, "command"), patch.object(session.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    session.criu("dump", run, 2, tools, {"PATH": "/usr/bin"}, 1)
            self.assertFalse((run / "continue-2").exists())

    def test_exit_wait_is_bounded_and_cleanup_targets_the_launched_run(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(run)])
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    session.reap(process.pid, process, timeout=0.01)
            finally:
                session.cleanup(process.pid, run, process)
            self.assertFalse(session.alive(process.pid))
            with self.assertRaisesRegex(RuntimeError, "exited"):
                session.wait_marker(run / "ready-2", process.pid, timeout=0.01)


if __name__ == "__main__":
    unittest.main()
