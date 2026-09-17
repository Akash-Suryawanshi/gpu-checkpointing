"""Timing must stop at the actual next update and distinguish exit from availability."""

import unittest
from argparse import Namespace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import metrics
import run as experiment


class TimingTests(unittest.TestCase):
    def test_restored_time_namespace_is_converted_before_comparing_clocks(self):
        """REGRESSION (2026-09-14 clock mismatch): align restored and controller timestamps.

        Linux can shift the restored process's monotonic clock. Subtracting its
        raw timestamp from the controller's gave a negative restoration latency.
        """
        times = {'capture_requested': 10, 'original_exit_verified': 13,
                 'gpu_observed_after_exit': 13.1, 'filesystem_synced': 15,
                 'job_b_requested': 15, 'job_b_completed': 30,
                 'restore_requested': 40, 'restore_returned': 41}
        events = [{'event': name, 'generation': 2, 'monotonic_ns': int(second * 1e9)}
                  for name, second in times.items()]
        updates = [{'update': 3, 'completed_ns': 17_700_000_000}]
        # The trainer reads 17.7 s while the controller reads 42 s at the same
        # instant: its restored clock offset is -24.3 s, written as -25 s + 0.7 s.
        with self.assertRaisesRegex(ValueError, 'clock domain'):
            metrics.latencies(events, updates)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'images-2').mkdir()
            (output / 'images-2/restore.log').write_text('timens: monotonic -25 700000000\n')
            (output / 'updates.jsonl').write_text(json.dumps(updates[0]) + '\n')
            result = {'mode': 'criu', 'events': events}
            metrics.finish(result, output, controller_offset_ns=0)
            self.assertEqual(result['latencies'][0]['restore_to_next_update_seconds'], 2)
            self.assertEqual(result['trainer_monotonic_offsets_ns'][2], -24_300_000_000)

    def test_timing_cannot_launch_from_failed_correctness_evidence(self):
        """Critical: timing must not skip inspections without matching passed diagnostics."""
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            reference = parent / 'reference'
            reference.mkdir()
            (reference / 'verified').touch()
            (reference / 'key.json').write_text('{}')
            (reference / 'result.json').write_text(json.dumps({'numerical_passed': False}))
            args = Namespace(run_dir=parent / 'trial', assets=parent, tools=parent,
                             reference=reference, validated_run=reference, timing=True,
                             mode='application', capture=[2])
            with patch.object(experiment, 'reference_key', return_value={}), patch.object(experiment.session, 'launch') as launch:
                with self.assertRaisesRegex(ValueError, 'did not pass'):
                    experiment.run(args)
                launch.assert_not_called()

    def test_hashing_and_job_b_do_not_enter_restore_latency(self):
        """Critical: restoration latency ends at the next update and excludes separate work.

        Image sync, original exit, GPU observation, and job B each have their own
        intervals; later verification must not lengthen the restoration result.
        """
        times = {"capture_requested": 10, "original_exit_verified": 13,
                 "gpu_observed_after_exit": 13.1, "filesystem_synced": 15,
                 "job_b_requested": 15, "job_b_completed": 30,
                 "restore_requested": 40, "restore_returned": 41,
                 "verification_passed": 100}
        events = [{"event": name, "generation": 2, "monotonic_ns": int(second * 1e9)}
                  for name, second in times.items()]
        measured, = metrics.latencies(events, [{"update": 3, "completed_ns": 42_000_000_000}])
        self.assertEqual(measured["capture_to_sync_seconds"], 5)
        self.assertEqual(measured["restore_to_next_update_seconds"], 2)
        self.assertEqual(measured["job_b_seconds"], 15)
        self.assertEqual(measured["capture_to_original_exit_seconds"], 3)
        self.assertAlmostEqual(measured["post_exit_observation_window_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
