"""Acceptance boundaries for the vLLM routes, without a GPU or an engine."""

import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import measure

spec = importlib.util.spec_from_file_location(
    "vllm_engine", Path(__file__).resolve().parents[1] / "experiments/vllm/engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def snapshot_record(route="snapshot", release=10, capture=20):
    """One passing record, in the shape trial.py assembles before it is believed."""
    observations = []
    for index in (1, 2):
        identity = {"run_id": "run", "request_id": str(index)}
        base = 100 * index
        observations.append({"request": identity.copy(), "first": {**identity, "token": 7},
            "response": {**identity, "tokens": [7, 8], "text": "blue"},
            "start_ns": base, "request_start_ns": base + 1, "published_ns": base + 2,
            "first_ns": base + 3, "completed_ns": base + 4})
    captured = route == "snapshot"
    return {"status": "passed", "cleanup": "complete", "route": route, "run_id": "run",
            "data_cache": "cold", "cold_cache": [{"resident_after": 0}],
            "activation": {"start_ns": observations[0]["start_ns"]},
            "kv_release_ns": release if captured else None,
            "capture_start_ns": capture if captured else None,
            "lifecycle": {name: captured for name in ("original_reaped", "capture_exited",
                                                      "restore_exited", "image_hashes_match")},
            "observations": observations}


REFERENCE = {"tokens": [7, 8], "text": "blue", "first_token": 7}


class EngineSettingsTests(unittest.TestCase):
    def test_the_eager_control_actually_disables_graph_capture(self):
        """Critical: the eager route exists only to remove CUDA-graph capture from
        startup. If the route never reached the engine's constructor, eager would
        measure exactly what cold measures and the compile cost would read as zero."""
        captured = {}

        class FakeLLM:
            def __init__(self, **settings):
                captured.clear()
                captured.update(settings)

        fake = types.ModuleType("vllm")
        fake.LLM = FakeLLM
        with mock.patch.dict(sys.modules, {"vllm": fake, "torch": mock.MagicMock()}):
            for route, expected in (("eager", True), ("cold", False), ("snapshot", False)):
                engine.load("/model", 0.3, engine.eager_route(route), 2048)
                self.assertEqual(captured["enforce_eager"], expected, route)
        self.assertEqual(captured["gpu_memory_utilization"], 0.3)
        self.assertEqual(captured["max_model_len"], 2048)
        # Prefix caching would let the second request reuse the first one's blocks,
        # so the follow-up timing would stop measuring a real forward pass.
        self.assertFalse(captured["enable_prefix_caching"])


class RecordAdmissionTests(unittest.TestCase):
    def test_capture_must_follow_the_key_value_cache_release(self):
        """Non-obvious correctness: nothing else orders these two events. An image
        taken before the release carries live cache blocks, which every restore then
        reads back; the order is visible only by comparing the worker's released_ns
        with the controller's capture start."""
        record = snapshot_record()
        self.assertEqual(measure.validate_vllm(record, REFERENCE)["ttft_seconds"], 3e-9)
        for release, capture in ((20, 20), (30, 20), (None, 20), (10, None)):
            broken = {**record, "kv_release_ns": release, "capture_start_ns": capture}
            with self.assertRaises(ValueError):
                measure.validate_vllm(broken, REFERENCE)
        # A launched route captures nothing, so it may not claim either event.
        with self.assertRaises(ValueError):
            measure.validate_vllm({**snapshot_record("cold"), "kv_release_ns": 10}, REFERENCE)

    def test_output_differing_from_the_reference_is_not_a_timing(self):
        """Critical: a route that answers differently is not a faster route. Every
        route has to reproduce the tokens vLLM itself recorded during preparation,
        and a cold-cache claim needs the residency evidence to back it."""
        for change in ({"response": {"run_id": "run", "request_id": "1", "tokens": [9, 8], "text": "blue"}},
                       {"response": {"run_id": "run", "request_id": "1", "tokens": [7, 8], "text": "red"}},
                       {"first": {"run_id": "run", "request_id": "1", "token": 9}}):
            record = copy.deepcopy(snapshot_record())
            record["observations"][0].update(change)
            with self.assertRaises(ValueError):
                measure.validate_vllm(record, REFERENCE)
        cached = copy.deepcopy(snapshot_record())
        cached["cold_cache"] = [{"resident_after": 4}]
        with self.assertRaises(ValueError):
            measure.validate_vllm(cached, REFERENCE)


if __name__ == "__main__":
    unittest.main()
