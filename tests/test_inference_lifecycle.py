"""Admission and CUDA state-machine failure contracts, without simulating CUDA."""

import os
import time
import unittest
from unittest.mock import patch

import park


class LifecycleTests(unittest.TestCase):
    def test_every_helper_failure_stops_without_guessed_recovery(self):
        """AC5: failures before/after all four actions never trigger a guessed retry."""
        states = ["running", "", "locked", "locked", "", "checkpointed",
                  "checkpointed", "", "locked", "locked", "", "running"]
        for fail in range(len(states)):
            with self.subTest(fail=fail):
                calls = []
                def command(*args):
                    index = len(calls)
                    calls.append(args)
                    if index == fail:
                        raise TimeoutError("injected")
                    return states[index]
                parking = park.Parking(None, {}, time.monotonic() + 1, lambda *a, **k: None)
                with patch.object(parking, "command", side_effect=command):
                    with self.assertRaises(TimeoutError):
                        parking.park()
                        parking.wake()
                self.assertEqual(len(calls), fail + 1)
        with patch.object(parking, "command", return_value="unknown") as call:
            with self.assertRaises(ValueError):
                parking.park()
            self.assertEqual(call.call_count, 1)

    def test_expired_or_wrong_identity_never_calls_helper(self):
        """Critical: deadline and identity checks precede every helper subprocess."""
        for deadline, matches in [(time.monotonic() - 1, True), (time.monotonic() + 5, False)]:
            parking = park.Parking("helper", {"uid": os.getuid()}, deadline, lambda *a, **k: None)
            with patch.object(park.session, "matches", return_value=matches), patch.object(park.subprocess, "run") as run:
                with self.assertRaises((TimeoutError, ValueError)):
                    parking.park()
                run.assert_not_called()

    def test_capacity_counts_staging_once_and_restore_rss_separately(self):
        """AC3: park uses available RAM; restore additionally needs the absent RSS."""
        memory = {"reserved_vram": 8 * park.GIB, "rss_bytes": 3 * park.GIB}
        observed = {"compute_pids": [], "gpu_free": 20 * park.GIB, "available": 12 * park.GIB}
        park.admission(observed, memory, "park")
        with patch.object(park.subprocess, "run") as helper:
            with self.assertRaisesRegex(ValueError, "host memory"):
                park.admission(observed, memory, "restore")
            helper.assert_not_called()
        self.assertEqual(park.job_size(18 * park.GIB, 22 * park.GIB), (256, "availability_only"))
        self.assertEqual(park.job_size(6 * park.GIB, 22 * park.GIB), (12288, "allocation_enabled_by_release"))

    def test_release_failure_and_early_registration_cleanup_reap_only_owned_child(self):
        """REGRESSION (inference release gap): missing result must not leak adopted workers."""
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        import lifecycle
        import control
        for registered in (True, False):
            with self.subTest(registered=registered), tempfile.TemporaryDirectory() as folder:
                run = Path(folder)
                (run / "snapshot").mkdir()
                (run / "control").mkdir()
                (run / "attempts/a").mkdir(parents=True)
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(run)])
                identity = lifecycle.session.identity(child.pid)
                job = {"job_id": "j", "run": str(run), "identity": identity if registered else {"pid": -1}}
                control.write(run / "job.json", job)
                control.write(run / "snapshot/manifest.json", {"run": str(run), "capture_id": "c", "job": job})
                control.phase(run, "restored" if registered else "restoring", capture_id="c", attempt_id="a")
                (run / "attempts/a/restored-a.pid").write_text(str(child.pid))
                try:
                    with patch.object(lifecycle.session, "matches", return_value=False), \
                            patch.object(lifecycle.session, "kill_identified") as kill:
                        with self.assertRaises(ValueError):
                            lifecycle.cleanup(run)
                        kill.assert_not_called()
                    lifecycle.cleanup(run, process=child)
                    self.assertFalse(Path(f"/proc/{child.pid}").exists())
                finally:
                    if child.poll() is None:
                        child.kill()
                    child.wait()

    def test_restore_helper_failure_after_release_transfers_cleanup_to_supervisor(self):
        """AC6: inject failure after actual release and before result; reap the orphan."""
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        import lifecycle
        import control
        lifecycle.session.adopt_restored_children()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "snapshot").mkdir()
            (run / "control").mkdir()
            (run / "attempts/c").mkdir(parents=True)
            job = {"job_id": "j", "run": str(run), "identity": {"pid": -1}}
            control.write(run / "job.json", job)
            control.write(run / "snapshot/manifest.json", {"run": str(run), "capture_id": "c", "job": job, "update": 1})
            control.write(run / "snapshot/before.json", {"weights": "untouched"})
            # A separate helper uses the real restore/release code. Only CRIU and
            # its admission are replaced; the child/adoption/registration are real.
            script = '''
import sys, subprocess
from pathlib import Path
from argparse import Namespace
import restore, control
run = Path(sys.argv[1])
manifest = control.read(run / "snapshot/manifest.json")
control.validate = lambda *args: manifest
original_write = control.write
def fail_result(path, value):
    if path.name == "result.json":
        raise RuntimeError("injected after release")
    original_write(path, value)
control.write = fail_result
def reconstruct(*args, **kwargs):
    attempt = kwargs["attempt"]
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(run)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (attempt / f"restored-{attempt.name}.pid").write_text(str(child.pid))
    original_write(attempt / "after.json", {"weights": "untouched"})
    original_write(attempt / "inspected.json", {"attempt_id": attempt.name})
    return child.pid
restore.session.criu = reconstruct
restore.restore(Namespace(snapshot=run / "snapshot", tools=run, timeout=5))
'''
            helper = subprocess.run([sys.executable, "-c", script, str(run)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(helper.returncode, 0)
            self.assertIn("injected after release", helper.stderr)
            phase = control.read(run / "control/phase.json")
            attempt = run / "attempts" / phase["attempt_id"]
            identity = control.read(run / "job.json")["identity"]
            try:
                self.assertTrue((attempt / "continue.json").exists())
                self.assertFalse((attempt / "result.json").exists())
                self.assertTrue(lifecycle.session.matches(identity))
                lifecycle.cleanup(run)
                self.assertFalse(Path(f"/proc/{identity['pid']}").exists())
            finally:
                if lifecycle.session.matches(identity):
                    lifecycle.session.kill_identified(identity)
                    lifecycle.session.reap(identity["pid"], timeout=5)
