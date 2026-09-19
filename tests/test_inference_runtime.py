"""Runtime state machine against the real worker request loop, without CUDA."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import control
import park
import runtime

FAKE_WORKER = r'''
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, %r)
import worker, control
args = argparse.ArgumentParser()
for name in ("assets", "run-dir", "route", "kind", "run-id"):
    args.add_argument("--" + name)
args = args.parse_args()
run = Path(args.run_dir)
class Tok:
    def decode(self, ids, **_): return "".join(chr(64 + i) for i in ids)
def fake_generate(model, tok, inputs, count, first=None, on_token=None):
    tokens = [1, 2] if inputs["input_ids"] != [9] else []
    if not tokens: raise RuntimeError("worker crashed on request")
    for i, t in enumerate(tokens):
        on_token(i, t); time.sleep(0.02)
    return {"tokens": tokens, "text": tok.decode(tokens)}
worker.generate = fake_generate
worker.render = lambda tok, prompt: {"input_ids": [5]}
worker.inspect = lambda model, kind: {"audit": "prepared"}
worker.memory = lambda: {"rss_bytes": 1, "reserved_vram": 2, "VmRSS": "1 kB", "allocated_vram": 2}
control.write(run / "memory.json", worker.memory())
control.write(run / "idle.json", {"run_id": args.run_id, "kv_cache": None})
worker.serve(run, args.run_id, None, Tok(), {"max_new_tokens": 16}, 0.005)
'''

IDLE = {"compute_pids": [], "gpu_free": 20 * park.GIB, "gpu_used": 0, "gpu_total": 24 * park.GIB, "available": 20 * park.GIB}


class RuntimeTests(unittest.TestCase):
    def test_activation_streams_tokens_then_unload_releases_and_failure_cleans_up(self):
        """AC (endpoint lifecycle): ABSENT -> ACTIVATING -> READY on a fresh worker, tokens
        stream before the response completes, the second request enforces the weight audit,
        unload reaps and returns to ABSENT, and a worker crash reaches FAILED then ABSENT
        only through cleanup. Uses the real serve() loop; only CUDA and nvidia-smi are faked."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets = root / "assets"
            assets.mkdir()
            for name, value in (("reference.json", {"tokens": [1, 2]}), ("tokens.json", {"max_new_tokens": 16}),
                                ("audit.json", {"audit": "prepared"})):
                control.write(assets / name, value)
            script = root / "fake_worker.py"
            script.write_text(FAKE_WORKER % str(Path(runtime.__file__).parent))
            deadline = time.monotonic() + 20
            with patch.object(park, "resources", return_value=IDLE), \
                 patch.object(runtime.Runtime, "_key", return_value=({"model_path": str(root)}, {"k": 1})), \
                 patch.object(park.Sampler, "sample", lambda self: None):
                rt = runtime.Runtime(assets, root / "tools", root / "service", "fresh", poll_ms=5)
                rt.worker_script = script
                self.assertEqual(rt.status()["state"], "ABSENT")
                activation = rt.ensure_ready(deadline)
                self.assertEqual(rt.status()["state"], "READY")
                self.assertEqual(rt.ensure_ready(deadline), activation, "READY must not activate again")
                seen = []
                observed = rt.generate({"prompt": "hi", "max_new_tokens": 16}, deadline,
                                       on_token=lambda event: seen.append((event["index"], time.monotonic_ns())))
                self.assertEqual([i for i, _ in seen], [0, 1])
                self.assertLess(seen[0][1], observed["completed_ns"] - 15_000_000, "first token precedes completion")
                self.assertEqual(observed["response"]["tokens"], [1, 2])
                self.assertEqual(observed["first"]["token"], 1)
                second = rt.active["requests"]
                rt.generate({"input_ids": [3], "max_new_tokens": 16}, deadline)
                self.assertFalse((second / "audit-after.json").exists(),
                                 "the audit belongs after the response pair, not between responses")
                with self.assertRaises(RuntimeError):
                    rt.generate({"input_ids": [9], "max_new_tokens": 16}, deadline)
                self.assertEqual(rt.status()["state"], "FAILED")
                with self.assertRaises(RuntimeError):
                    rt.ensure_ready(deadline)
                rt.unload(deadline)
                self.assertEqual(rt.status()["state"], "ABSENT")
                again = rt.ensure_ready(deadline)
                self.assertNotEqual(again, activation)
                rt.generate({"input_ids": [3], "max_new_tokens": 16}, deadline)
                served = rt.active["requests"]
                rt.unload(deadline)
                self.assertEqual(rt.status()["state"], "ABSENT")
                self.assertTrue((served / "audit-after.json").exists(), "a served worker audits before exiting")
                self.assertTrue((root / "service/activations" / again / "completed.json").exists())
            self.assertFalse((root / "service/activations" / again / "trainer.stderr").read_text().strip(),
                             "a stopped worker exits cleanly")

    def test_benchmark_bookkeeping_and_the_weight_audit_stay_out_of_request_timing(self):
        """REGRESSION (http-ebs-02 b1, 2026-09-18): the first client request measured 145.7 s
        because ensure_ready() fingerprinted every model file inside it, which also warmed the
        cache the trial had just evicted; the second request measured 14.1 s because it waited
        for the worker's full weight audit. The plan puts both outside request timing: the key
        is server startup work, and audits run after the response pair."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets = root / "assets"
            assets.mkdir()
            for name, value in (("reference.json", {"tokens": [1, 2]}), ("tokens.json", {"max_new_tokens": 16}),
                                ("audit.json", {"audit": "prepared"})):
                control.write(assets / name, value)
            script = root / "fake_worker.py"
            script.write_text(FAKE_WORKER % str(Path(runtime.__file__).parent))
            deadline = time.monotonic() + 20
            with patch.object(park, "resources", return_value=IDLE), \
                 patch.object(park.Sampler, "sample", lambda self: None), \
                 patch.object(runtime.Runtime, "_key", return_value=({"model_path": str(root)}, {"k": 1})) as key:
                rt = runtime.Runtime(assets, root / "tools", root / "service", "fresh", poll_ms=5)
                rt.worker_script = script
                self.assertEqual(key.call_count, 1, "the comparison key is built once, at startup")
                rt.ensure_ready(deadline)
                self.assertEqual(key.call_count, 1, "activation must not re-fingerprint the model")
                rt.generate({"input_ids": [3], "max_new_tokens": 16}, deadline)
                # The audit appears only after response one; a second request must not wait for it.
                audit = rt.active["requests"] / "audit-after.json"
                if audit.exists():
                    audit.unlink()
                started = time.monotonic_ns()
                rt.generate({"input_ids": [4], "max_new_tokens": 16}, deadline)
                self.assertLess((time.monotonic_ns() - started) / 1e9, 2.0, "second request waited on the audit")
                rt.unload(deadline)
                self.assertEqual(rt.status()["state"], "ABSENT")
            # The audit is still enforced once the pair is done: a drifted one fails.
            drifted = {"dir": root, "requests": root, "index": 1}
            control.write(root / "prepared-audit.json", {"audit": "prepared"})
            control.write(root / "audit-after.json", {"audit": "changed"})
            with self.assertRaisesRegex(ValueError, "Immutable model state changed"):
                rt._verify_audit(drifted)
            control.write(root / "audit-after.json", {"audit": "prepared"})
            rt._verify_audit(drifted)

    def test_the_weight_audit_never_runs_between_two_responses(self):
        """REGRESSION (http-ebs-03 b1-fresh, 2026-09-18): a warm follow-up measured 14.4 s
        because the worker fingerprinted every weight after response one, reading no request
        while it did. Moving the controller's wait was not enough; the worker itself must
        audit only after the owner stops sending, which is what the plan requires."""
        import re
        source = (Path(runtime.__file__).parent / "worker.py").read_text()
        serve = source[source.index("def serve("):source.index("def main(")]
        loop = serve[serve.index("while (request :="):]
        body, _, after = loop.partition('control.write(requests / "audit-after.json"')
        self.assertTrue(after, "the audit must still be written")
        self.assertNotIn("audit-after", body, "the audit must not run inside the request loop")
        # It must also precede the completion marker the owner waits on.
        self.assertLess(serve.index('"audit-after.json"'), serve.index('"completed.json"'))
