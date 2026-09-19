"""Restore admission and release safety, without emulating CRIU."""

from pathlib import Path
import tempfile
import unittest

import control


class RestoreTests(unittest.TestCase):
    def test_mismatched_inspection_never_releases_continuation(self):
        """Critical: an independent restore compares untouched evidence before permission."""
        import restore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control.write(root / "before.json", {"rng": "original"})
            control.write(root / "after.json", {"rng": "changed"})
            control.write(root / "inspected.json", {"attempt_id": "r"})
            restoration = {"capture_id": "c", "attempt_id": "r"}
            with self.assertRaisesRegex(ValueError, "rng"):
                restore.release(root, root, restoration, root / "continue.json")
            self.assertFalse((root / "continue.json").exists())

    def test_corrupt_or_ambiguous_snapshot_is_rejected_before_criu(self):
        """AC4: missing completion, corruption, stale phase, and drift block admission."""
        import os
        import time
        from unittest.mock import patch
        cases = ("valid", "completion", "digest", "schema", "payload", "phase", "identity", "stale", "environment", "external")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                run, snapshot = root / "job", root / "snapshot"
                run.mkdir()
                job = control.register(run, run, 4)
                job["identity"]["pid"] = 999999999
                control.write(run / "job.json", job)
                capture, attempt = control.attempt(run)
                request = {"capture_id": capture}
                control.write(run / "control/request.json", request)
                (snapshot / "images").mkdir(parents=True)
                for name in ("inventory.img", "pstree.img"):
                    (snapshot / "images" / name).write_bytes(b"image")
                (snapshot / "before.json").write_text("{}")
                external = {}
                for name in ("updates.jsonl", "trainer.stderr"):
                    (run / name).touch()
                    external[name] = control.file_hash(run / name)
                manifest = {"schema": control.SCHEMA, "capture_id": capture, "run": str(run),
                            "job": job, "request": request, "dependencies": {"fingerprint": "expected"},
                            "external_files": external}
                control.publish(snapshot, manifest, run, time.monotonic() + 5, lambda name: None)
                if case == "completion":
                    (snapshot / "COMPLETE").unlink()
                elif case == "digest":
                    (snapshot / "manifest.json").write_text("{}")
                elif case == "schema":
                    manifest["schema"] = -1
                    control.write(snapshot / "manifest.json", manifest)
                elif case == "payload":
                    (snapshot / "images/pstree.img").write_bytes(b"corrupt")
                elif case == "phase":
                    control.phase(run, "dumping", capture_id=capture)
                elif case == "identity":
                    control.write(run / "job.json", {**job, "job_id": "wrong"})
                elif case == "stale":
                    control.write(attempt / "inspect.json", {})
                elif case == "external":
                    (run / "updates.jsonl").write_text("changed")
                observed = {"fingerprint": "changed" if case == "environment" else "expected"}
                with patch.object(control, "dependencies", return_value=observed), \
                     patch.object(control.session, "criu") as criu:
                    for order in ("payload-first", "dependencies-first"):
                        events = []
                        if case == "valid":
                            control.validate(snapshot, run, root, time.monotonic() + 5, order, events.append)
                            expected = ["payload_validation_started", "payload_validation_completed",
                                        "dependency_validation_started", "dependency_validation_completed"]
                            self.assertEqual(events, expected if order == "payload-first" else expected[2:] + expected[:2])
                        else:
                            with self.assertRaises((ValueError, FileNotFoundError, KeyError)):
                                control.validate(snapshot, run, root, time.monotonic() + 5, order, events.append)
                    criu.assert_not_called()
