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
            with patch.object(park.session, "matches", return_value=matches), patch.object(park.subprocess, "Popen") as run:
                with self.assertRaises((TimeoutError, ValueError)):
                    parking.park()
                run.assert_not_called()

    def test_capacity_counts_staging_once_and_restore_rss_separately(self):
        """AC3: park uses available RAM; restore additionally needs the absent RSS."""
        memory = {"reserved_vram": 8 * park.GIB, "rss_bytes": 3 * park.GIB}
        observed = {"compute_pids": [], "gpu_free": 20 * park.GIB, "available": 12 * park.GIB}
        park.admission(observed, memory, "park")
        with patch.object(park.subprocess, "Popen") as helper:
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

    def test_interrupt_kills_and_reaps_helper_descendants(self):
        """REGRESSION (cancelled helper): interruption must not leave a command running."""
        import sys
        import tempfile
        from pathlib import Path
        import lifecycle
        lifecycle.session.adopt_restored_children()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / "child.pid"
            script = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            with self.assertRaises(KeyboardInterrupt):
                with lifecycle.helper([sys.executable, "-c", script, marker], root / "helper.log",
                                      os.environ, time.monotonic() + 5, privileged=False) as command:
                    limit = time.monotonic() + 5
                    while not marker.exists():
                        park.control.remaining(limit)
                        time.sleep(0.005)
                    child = int(marker.read_text())
                    raise KeyboardInterrupt()
            self.assertFalse(Path(f"/proc/{command.pid}").exists())
            self.assertFalse(Path(f"/proc/{child}").exists())

    def test_controller_has_no_unowned_blocking_commands(self):
        """REGRESSION (cancelled job B): direct run() can leave an unreaped child."""
        import ast
        from pathlib import Path
        source = Path(__file__).resolve().parents[1] / "experiments/inference/run.py"
        calls = [ast.unparse(node.func) for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.Call)]
        # The group owner has a real descendant-cancellation check above. Keep
        # controller subprocesses inside it until their exits are collected.
        self.assertNotIn("session.command", calls)
        self.assertNotIn("subprocess.run", calls)

    def test_interrupted_cuda_helper_is_reaped_before_return(self):
        """REGRESSION (cancelled direct helper): kill alone leaves an uncollected exit."""
        import signal
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helper = root / "helper"
            marker = root / "pid"
            helper.write_text("#!/usr/bin/python3\nimport os,time\nfrom pathlib import Path\n"
                f"Path({str(marker)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n")
            helper.chmod(0o700)
            parking = park.Parking(helper, {"pid": os.getpid(), "uid": os.getuid()},
                                   time.monotonic() + 5, lambda *a, **k: None)
            previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.5)
                # Incidental destructor polling is not a cleanup guarantee.
                with patch.object(park.session, "matches", return_value=True), \
                        patch.object(park.subprocess.Popen, "__del__", lambda self: None), \
                        self.assertRaises(KeyboardInterrupt):
                    parking.command("--get-state")
                self.assertTrue(marker.exists())
                self.assertFalse(Path(f"/proc/{int(marker.read_text())}").exists())
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
                if marker.exists() and Path(f"/proc/{int(marker.read_text())}").exists():
                    pid = int(marker.read_text())
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
