"""Acceptance boundaries for externally observed inference latency."""

import copy
import unittest

import measure


class MeasureTests(unittest.TestCase):
    def test_clock_and_output_failures_cannot_be_reported_as_speedup(self):
        """Critical: invalid clocks, IDs, or responses invalidate latency evidence."""
        reference = {"tokens": [1, 2], "text": "example", "first_token": 1}
        records = []
        for index in (1, 2):
            identity = {"run_id": "run", "request_id": str(index)}
            records.append({"request": identity.copy(), "first": {**identity, "token": 1},
                "response": {**identity, "tokens": [1, 2], "text": "example"},
                "start_ns": index * 10, "request_start_ns": index * 10,
                "published_ns": index * 10 + 1, "first_ns": index * 10 + 2, "completed_ns": index * 10 + 3})
        result = measure.durations(records, "run", reference)
        self.assertEqual(result["start_to_first_token_seconds"], 2e-9)
        self.assertEqual(result["health_request_to_first_token_seconds"], 2e-9)
        self.assertNotIn("resident_request_to_first_token_seconds", result)
        changes = [(0, "first_ns", 0), (0, "first_ns", float("nan")),
                   (0, "first", {"run_id": "wrong", "request_id": "1", "token": 1}),
                   (0, "first", {"run_id": "run", "request_id": "old", "token": 1}),
                   (0, "first", {"run_id": "run", "request_id": "1", "token": 2}),
                   (1, "request", {"run_id": "run", "request_id": "1"}),
                   (0, "response", {"run_id": "run", "request_id": "1", "tokens": [2], "text": "wrong"})]
        for index, key, value in changes:
            altered = copy.deepcopy(records)
            altered[index][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                measure.durations(altered, "run", reference)
