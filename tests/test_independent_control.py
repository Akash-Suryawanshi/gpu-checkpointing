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

    def test_memory_backed_snapshot_storage_is_rejected(self):
        """AC1: a snapshot on tmpfs cannot satisfy the local storage contract."""
        with self.assertRaisesRegex(ValueError, "tmpfs"):
            control.storage(Path("/dev/shm"))

    def test_publication_order_and_failure_preserve_previous_snapshot(self):
        """AC1: every write/sync/rename failure rejects publication and preserves old data."""
        # First count the actual persistence operations, then fail each one once.
        # The payload, filesystem writes, hashes and locking remain real.
        failure_points = [None]
        for fail_at in failure_points:
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                run = root / "job"
                (run / "control").mkdir(parents=True)
                old = root / "old"
                old.mkdir()
                (old / "COMPLETE").write_text("previous")
                snapshot = root / "new"
                (snapshot / "images").mkdir(parents=True)
                for name in ("inventory.img", "pstree.img"):
                    (snapshot / "images" / name).write_bytes(b"payload")
                (snapshot / "before.json").write_text("{}")
                manifest = {"capture_id": "new", "schema": 1}
                count = 0
                operations = []
                original_sync, original_replace, original_dump = os.fsync, os.replace, json.dump

                def call(kind, fn, *args, **kwargs):
                    nonlocal count
                    count += 1
                    operations.append(kind)
                    if count == fail_at:
                        raise OSError("injected storage failure")
                    return fn(*args, **kwargs)

                def emit(name):
                    if name == "local_snapshot_published":
                        self.assertEqual(control.read(run / "control/phase.json")["status"], "published")
                        self.assertTrue((snapshot / "COMPLETE").exists())
                    operations.append(name)

                with patch.object(os, "fsync", side_effect=lambda *a: call("sync", original_sync, *a)), \
                     patch.object(os, "replace", side_effect=lambda *a: call("rename", original_replace, *a)), \
                     patch.object(json, "dump", side_effect=lambda *a, **kw: call("write", original_dump, *a, **kw)):
                    try:
                        control.publish(snapshot, manifest, run, time.monotonic() + 5, emit)
                    except OSError:
                        self.assertFalse((snapshot / "COMPLETE").exists())
                        self.assertNotIn("local_snapshot_published", operations)
                    else:
                        self.assertLess(operations.index("payload_synced"), operations.index("rename"))
                        self.assertEqual(operations[-1], "local_snapshot_published")
                self.assertEqual((old / "COMPLETE").read_text(), "previous")
                if fail_at is None:
                    failure_points.extend(range(1, count + 1))

    def test_dangling_development_symlink_does_not_block_runtime_fingerprints(self):
        """REGRESSION (independent preflight): extracted libmd.so links can be dangling."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "libmd.so.0").write_bytes(b"runtime")
            (root / "libmd.so").symlink_to("absent-development-library")
            self.assertEqual(control.runtime_libraries(root), [root / "libmd.so.0"])

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

    def test_restored_trainer_keeps_worker_identity_for_the_next_capture(self):
        """REGRESSION (second generation): trainer clock-domain ticks must not replace host identity."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            code = ("from pathlib import Path; import control,sys,time; p=Path(sys.argv[1]); "
                    "j=control.register(p,p,4); control.wait(p/'go.json',time.monotonic()+5); "
                    "control.boundary(p,j,2,lambda: {'update':2}); "
                    "control.write(p/'observed.json',j)")
            child = subprocess.Popen([sys.executable, "-c", code, folder])
            try:
                job = control.wait(run / "job.json", time.monotonic() + 5)
                capture, before = control.attempt(run)
                attempt_id, after = control.attempt(run)
                control.phase(run, "requested", capture_id=capture)
                control.write(run / "control/request.json", {"capture_id": capture,
                              "job_id": job["job_id"], "at": 2, "deadline": time.monotonic()+5})
                control.write(run / "go.json", {})
                control.wait(before / "ready.json", time.monotonic() + 5)
                # Emulate only the differing observer start tick, not a CRIU lifecycle.
                job["identity"]["start_ticks"] += 1234
                control.write(run / "job.json", job)
                control.phase(run, "restoring", capture_id=capture)
                signal = {"capture_id": capture, "attempt_id": attempt_id}
                control.write(before / "inspect.json", signal)
                control.wait(after / "inspected.json", time.monotonic() + 5)
                control.write(after / "continue.json", signal)
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertEqual(control.read(run / "observed.json")["identity"], job["identity"])
                self.assertEqual(control.read(run / "job.json")["identity"], job["identity"])
            finally:
                child.kill()
                child.wait()

    def test_request_deadline_uses_the_host_clock_after_restore(self):
        """REGRESSION (second capture): restored monotonic offsets must not extend expiry."""
        with patch.object(control.time, "monotonic", return_value=70), \
             patch.object(control, "monotonic_offset", return_value=-30):
            self.assertTrue(control.request_expired({"deadline": 99}))
            self.assertFalse(control.request_expired({"deadline": 101}))

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

    def test_expired_acknowledgement_cannot_resume_while_worker_owns_lock(self):
        """REGRESSION (pause lease race): expiry must not race a worker arming its dump."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            job = control.register(run, run, 4)
            capture, attempt = control.attempt(run)
            child = None
            try:
                with control.lock(run):
                    control.phase(run, "requested", capture_id=capture)
                    control.write(run / "control/request.json", {"capture_id": capture,
                        "job_id": job["job_id"], "at": 2, "deadline": time.monotonic() + .5})
                    code = "from pathlib import Path; import control,sys; p=Path(sys.argv[1]); control.boundary(p,control.read(p/'job.json'),2,lambda: {'update':2})"
                    child = subprocess.Popen([sys.executable, "-c", code, folder])
                    control.wait(attempt / "ready.json", time.monotonic() + 5)
                    time.sleep(.6)
                    self.assertIsNone(child.poll(), "Expired acknowledgement resumed while capture still owned the lock")
                # A dead/finished worker releases the lock. With no dump armed,
                # the expired pause can now end without touching CUDA state.
                self.assertEqual(child.wait(timeout=5), 0)
            finally:
                if child is not None:
                    child.kill()
                    child.wait()

    def test_external_pause_cancellation_expiry_and_matching_resume(self):
        """AC2: stale, cancelled, expired, and final-boundary requests cannot hang training."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            job = control.register(run, run, 4)
            capture, attempt = control.attempt(run)
            request = {"capture_id": capture, "job_id": job["job_id"], "at": 2,
                       "deadline": time.monotonic() + 10}
            control.write(run / "control/request.json", request)
            control.phase(run, "requested", capture_id=capture)
            seen = []
            control.boundary(run, job, 1, lambda: seen.append(1))
            self.assertEqual(seen, [])
            code = "from pathlib import Path; import control,sys; p=Path(sys.argv[1]); control.boundary(p,control.read(p/'job.json'),2,lambda: {'update':2})"
            child = subprocess.Popen([sys.executable, "-c", code, folder])
            try:
                ready = control.wait(attempt / "ready.json", time.monotonic() + 5)
                self.assertEqual(ready["update"], 2)
                self.assertIsNone(child.poll())
                control.phase(run, "cancelled", capture_id=capture)
                self.assertEqual(child.wait(timeout=5), 0)
            finally:
                child.kill()
                child.wait()
            # Expired requests and the final update produce a bounded rejection.
            for update, deadline in ((2, time.monotonic() - 1), (4, time.monotonic() + 10)):
                control.write(run / "control/request.json", {**request, "deadline": deadline})
                control.phase(run, "requested", capture_id=capture)
                control.boundary(run, job, update, lambda: seen.append(update))
                self.assertEqual(seen, [])
                self.assertTrue((attempt / "rejected.json").exists())
            control.write(run / "control/request.json", {**request, "job_id": "stale"})
            control.boundary(run, job, 2, lambda: seen.append(2))
            self.assertEqual(seen, [])



if __name__ == "__main__":
    unittest.main()
