"""Contracts for diagnostic-only startup instrumentation; no GPU needed."""

import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments/vllm"))
import startup_trace


class StartupTraceTests(unittest.TestCase):
    def test_nested_spans_preserve_failures_and_raw_clock_boundaries(self):
        """Critical: a failed nested phase is not a successful timing, and nested
        durations must retain boundaries so consumers cannot sum them blindly."""
        stream = io.StringIO()
        with mock.patch.object(startup_trace.time, "monotonic_ns", side_effect=[10, 20, 50, 70]):
            with self.assertRaisesRegex(ValueError, "original failure"):
                with startup_trace.span(stream, "outer"), startup_trace.span(stream, "inner"):
                    raise ValueError("original failure")
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([(row["phase"], row["event"]) for row in events],
                         [("outer", "start"), ("inner", "start"), ("inner", "end"), ("outer", "end")])
        self.assertEqual([row["monotonic_ns"] for row in events], [10, 20, 50, 70])
        self.assertEqual(events[-1]["seconds"], 60e-9)
        self.assertTrue(all(row["status"] == "failed" and row["error_type"] == "ValueError"
                            for row in events if row["event"] == "end"))

    def test_worker_instrumentation_is_optional_and_restores_original_methods(self):
        """Critical: timing runs install no hooks; diagnostics preserve return
        values and restore every worker method even when engine startup fails."""
        def original(self, *, value):
            return value

        worker = type("Worker", (), {name: original for name in startup_trace.WORKER_METHODS})
        module = types.ModuleType("vllm.v1.worker.gpu_worker")
        module.Worker = worker
        stream = io.StringIO()
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            with startup_trace.worker_spans(None):
                self.assertIs(worker.init_device, original)
            with self.assertRaisesRegex(RuntimeError, "stop startup"):
                with startup_trace.worker_spans(stream):
                    self.assertIsNot(worker.init_device, original)
                    self.assertEqual(worker().load_model(value=17), 17)
                    raise RuntimeError("stop startup")
        self.assertTrue(all(getattr(worker, name) is original for name in startup_trace.WORKER_METHODS))
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(events[-1]["phase"], "worker.load_model")
        self.assertEqual(events[-1]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
