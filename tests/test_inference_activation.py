"""Reusable image contract: one immutable publication, many sequential attempts."""

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import control


def publish_reusable(root):
    run, snapshot = root / "job", root / "snapshot"
    run.mkdir()
    job = control.register(run, run, 2)
    job["identity"]["pid"] = 999999999  # the saved PID: absent, so admissible
    control.write(run / "job.json", job)
    capture, attempt = control.attempt(run)
    request = {"capture_id": capture}
    control.write(run / "control/request.json", request)
    (snapshot / "images").mkdir(parents=True)
    for name in ("inventory.img", "pstree.img"):
        (snapshot / "images" / name).write_bytes(b"image")
    (snapshot / "before.json").write_text("{}")
    (snapshot / "external").mkdir()
    external = {}
    for name in ("updates.jsonl", "trainer.stderr"):
        (run / name).write_text("captured\n")
        (snapshot / "external" / name).write_text("captured\n")
        external[name] = control.file_hash(run / name)
    manifest = {"schema": control.SCHEMA, "capture_id": capture, "run": str(run), "job": job,
                "request": request, "dependencies": {"fingerprint": "expected"},
                "external_files": external, "contract": control.CONTRACT}
    control.publish(snapshot, manifest, run, time.monotonic() + 5, lambda name: None)
    return run, snapshot, job, capture


class ActivationTests(unittest.TestCase):
    def test_reusable_image_admits_sequential_attempts_and_rejects_live_or_foreign_ones(self):
        """AC (activation contract): the same publication restores again only after the
        previous attempt is terminal; consumed markers, worker log output, and refreshed
        registration are reset from the immutable baseline, never by editing the phase."""
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(control, "dependencies", return_value={"fingerprint": "expected"}):
            run, snapshot, job, capture = publish_reusable(Path(folder))
            deadline = time.monotonic() + 5
            control.validate(snapshot, run, folder, deadline)
            # Attempt A: restore consumed the marker, the worker logged, registration was refreshed.
            control.activation_state(run, "restoring", capture_id=capture, attempt_id="a")
            control.write(run / "attempts" / capture / "inspect.json", {"capture_id": capture, "attempt_id": "a"})
            with (run / "trainer.stderr").open("a") as stream:
                stream.write("restored worker output\n")
            control.write(run / "job.json", {**job, "identity": {**job["identity"], "start_ticks": 7}})
            with self.assertRaisesRegex(ValueError, "not terminal"):
                control.validate(snapshot, run, folder, deadline)
            control.activation_state(run, "released", capture_id=capture, attempt_id="a")
            control.validate(snapshot, run, folder, deadline)
            self.assertFalse((run / "attempts" / capture / "inspect.json").exists())
            self.assertEqual((run / "trainer.stderr").read_text(), "captured\n")
            self.assertEqual(control.read(run / "job.json"), job)
            self.assertEqual(control.read(run / "control/phase.json")["status"], "published")
            # Conflicts: the saved PID is held, the baseline copy changed, a legacy image
            # reaches this path, or the run was registered for another job.
            with patch.object(control.Path, "exists", return_value=True):
                with self.assertRaisesRegex(ValueError, "PID"):
                    control.validate(snapshot, run, folder, deadline)
            control.write(run / "job.json", {**job, "job_id": "other"})
            with self.assertRaisesRegex(ValueError, "identity"):
                control.validate(snapshot, run, folder, deadline)
            control.write(run / "job.json", job)
            (snapshot / "external/trainer.stderr").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "baseline|payload"):
                control.validate(snapshot, run, folder, deadline)

    def test_unknown_contract_is_rejected_before_any_payload_read(self):
        """Critical: a manifest naming a contract this code does not implement never restores."""
        with tempfile.TemporaryDirectory() as folder:
            run, snapshot, job, capture = publish_reusable(Path(folder))
            manifest = control.read(snapshot / "manifest.json")
            manifest["contract"] = "reusable-inference-v2"
            control.write(snapshot / "manifest.json", manifest)
            with patch.object(control, "inventory") as inventory, \
                 self.assertRaisesRegex(ValueError, "contract"):
                control.validate(snapshot, run, folder, time.monotonic() + 5)
            inventory.assert_not_called()

    def test_worker_serves_only_the_attempt_that_released_it(self):
        """Critical: a restored worker must never read requests from another attempt or
        from a path outside its attempt root, even when the record is well formed."""
        import worker
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "attempts/a/requests").mkdir(parents=True)
            restoration = {"capture_id": "c", "attempt_id": "a"}
            record = {"status": "restoring", **restoration, "requests": "attempts/a/requests"}
            root, requests = worker.attempt_paths(run, record, restoration)
            self.assertEqual(requests, (run / "attempts/a/requests").resolve())
            bad = [{**record, "attempt_id": "b"}, {**record, "requests": "attempts/b/requests"},
                   {**record, "requests": "attempts/a/../../requests"}, None]
            for wrong in bad:
                with self.subTest(record=wrong), self.assertRaises(ValueError):
                    worker.attempt_paths(run, wrong, restoration)
            with self.assertRaises(ValueError):
                worker.attempt_paths(run, record, None)
