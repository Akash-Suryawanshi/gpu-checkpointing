"""Phase arithmetic must preserve the actual order of validation and restore."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("inference_profile", Path(__file__).resolve().parents[1] / "experiments/inference/disk_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_validation_is_before_criu_and_bad_clock_domains_fail(self):
        """REGRESSION (disk explanation): validation was incorrectly assigned after CRIU."""
        observation = {"start_ns": 0, "first_ns": 278_000_000_000}
        events = [{"event": "restore_requested", "monotonic_ns": 138_000_000_000},
                  {"event": "restore_returned", "monotonic_ns": 277_000_000_000}]
        self.assertEqual(profile.breakdown(observation, events), {
            "before_criu_seconds": 138, "criu_seconds": 139, "after_criu_to_token_seconds": 1})
        events[0]["monotonic_ns"] = -1
        with self.assertRaises(ValueError):
            profile.breakdown(observation, events)
