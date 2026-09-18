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

    def test_http_trial_with_disordered_clock_or_drifted_output_is_never_accepted(self):
        """Critical: client-side TTFT evidence needs one ordered clock, a first token that is
        the first timed event, streamed tokens equal to the final answer, and the reference."""
        reference = {"tokens": [1, 2], "text": "ab", "first_token": 1}
        def observed(start):
            events = [{"event": "token", "token_id": 1, "received_ns": start + 5}, {"event": "token", "token_id": 2, "received_ns": start + 6},
                      {"event": "done", "tokens": [1, 2], "text": "ab", "received_ns": start + 7}]
            return {"start_ns": start, "headers_ns": start + 1, "first_ns": start + 5, "done_ns": start + 7, "events": events}
        record = {"status": "passed", "data_cache": "cold", "cold_cache": [{"resident_after": 0}],
                  "status_after": {"state": "READY"}, "primary": observed(100), "followup": observed(200)}
        self.assertEqual(measure.validate_http(record, reference), {"ttft_seconds": 5e-9, "followup_ttft_seconds": 5e-9})
        bad = [lambda r: r["primary"].__setitem__("first_ns", 50), lambda r: r["primary"]["events"][1].__setitem__("token_id", 9),
               lambda r: r["primary"]["events"][-1].__setitem__("text", "xx"), lambda r: r["cold_cache"][0].__setitem__("resident_after", 3),
               lambda r: r["status_after"].__setitem__("state", "FAILED")]
        for index, change in enumerate(bad):
            altered = copy.deepcopy(record)
            change(altered)
            with self.subTest(case=index), self.assertRaises(ValueError):
                measure.validate_http(altered, reference)
