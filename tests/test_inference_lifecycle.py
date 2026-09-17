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
