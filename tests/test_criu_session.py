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
    def test_second_restore_uses_a_fresh_pidfile(self):
        """REGRESSION (2026-09-14 repeated restore): CRIU refuses an existing PID file.

        Distinct capture generations must use distinct filenames even when the
        restored process has the same numeric process identifier (PID).
        """
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            tools = {"libraries": "unused", "criu": "unused", "plugin": "unused"}

            def fake_restore(arguments, log, env, timeout):
                """Reproduce CRIU's exclusive file creation without running CRIU."""
                pidfile = Path(arguments[arguments.index("--pidfile") + 1])
                with pidfile.open("x") as out:
                    out.write("123")  # Like CRIU, refuse to overwrite a previous PID file.
                images = Path(arguments[arguments.index("--images-dir") + 1])
                images.mkdir()
                (images / "restore.log").write_text("cuda_plugin: initialized: resuming devices on pid 123")

            with patch.object(session, "command", side_effect=fake_restore), patch.object(session.subprocess, "run"):
                self.assertEqual(session.criu("restore", run, 2, tools, {"PATH": "/usr/bin"}), 123)
                self.assertEqual(session.criu("restore", run, 3, tools, {"PATH": "/usr/bin"}), 123)

    def test_continuation_requires_matching_inspection_for_this_generation(self):
        """Critical: a missing, stale, or mismatched inspection must not release training.

        The continue marker is permission to update, so write it only after the
        current capture's before/after state has matched the reference.
        """
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
        """Critical: a successful command exit alone cannot establish a completed image."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            tools = {"libraries": "unused", "criu": "unused", "plugin": "unused"}
            with patch.object(session, "command"), patch.object(session.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    session.criu("dump", run, 2, tools, {"PATH": "/usr/bin"}, 1)
            self.assertFalse((run / "continue-2").exists())

    def test_exit_wait_is_bounded_and_cleanup_targets_the_launched_run(self):
        """Critical: failure cleanup stops the owned child without targeting another run.

        Use a real CPU child to check the bounded wait and Linux process removal.
        A different run directory must fail the controller's ownership guard.
        """
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(run)])
            try:
                # The PID identifies a process, not its purpose. The run path in
                # its argument list is the additional identity check used here.
                session.cleanup(process.pid, run / "different-run", process)
                self.assertIsNone(process.poll())
                with self.assertRaises(subprocess.TimeoutExpired):
                    session.reap(process.pid, process, timeout=0.01)
            finally:
                session.cleanup(process.pid, run, process)
            self.assertFalse(session.alive(process.pid))
            with self.assertRaisesRegex(RuntimeError, "exited"):
                session.wait_marker(run / "ready-2", process.pid, timeout=0.01)


if __name__ == "__main__":
    unittest.main()
