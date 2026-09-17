"""CPU acceptance checks for independent ownership and snapshot publication."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import control
import session


class ProtocolTests(unittest.TestCase):


    def test_lock_excludes_other_workers_and_death_does_not_reset_phase(self):
        """Critical: lock release after death must not authorize an ambiguous retry."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            code = ("from pathlib import Path; import control,sys,time; p=Path(sys.argv[1]);\n"
                    "with control.lock(p):\n"
                    " control.phase(p,'dumping',capture_id='interrupted'); time.sleep(60)")
            worker = subprocess.Popen([sys.executable, "-c", code, folder])
            try:
                control.wait(run / "control/phase.json", time.monotonic() + 5)
                with self.assertRaisesRegex(RuntimeError, "owns"):
                    with control.lock(run):
                        self.fail("Concurrent operation acquired lock")
            finally:
                worker.kill()
                worker.wait(timeout=5)
            with control.lock(run):
                self.assertEqual(control.read(run / "control/phase.json")["status"], "dumping")


    def test_attempt_creation_requires_parent_directory_sync(self):
        """REGRESSION (publication audit): losing the attempt entry loses restore controls."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            original = control.sync_directory
            def fail_parent(path):
                if Path(path) == run / "attempts":
                    raise OSError("attempt parent sync failed")
                original(path)
            with patch.object(control, "sync_directory", side_effect=fail_parent):
                with self.assertRaisesRegex(OSError, "parent sync"):
                    control.attempt(run)


    def test_worker_waits_for_registration_lock_but_keeps_its_deadline(self):
        """REGRESSION (startup race): visible job registration may still hold its lock."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            child = None
            try:
                with control.lock(run):
                    with self.assertRaises(TimeoutError):
                        with control.lock(run, time.monotonic() + .05):
                            self.fail("Concurrent worker acquired registration lock")
                    code = ("from pathlib import Path; import control,sys,time; p=Path(sys.argv[1]); "
                            "control.write(p/'waiting.json',{});\n"
                            "with control.lock(p,time.monotonic()+5): control.write(p/'acquired.json',{})")
                    child = subprocess.Popen([sys.executable, "-c", code, folder])
                    control.wait(run / "waiting.json", time.monotonic()+5)
                    self.assertFalse((run / "acquired.json").exists())
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertTrue((run / "acquired.json").exists())
            finally:
                if child is not None:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    unittest.main()
