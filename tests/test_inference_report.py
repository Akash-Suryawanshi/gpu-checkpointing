"""Report admission uses real evidence files, including external diagnostics."""

import json
from pathlib import Path
import tempfile
import unittest

import measure
import importlib.util
spec = importlib.util.spec_from_file_location("inference_report", Path(__file__).resolve().parents[1] / "experiments/inference/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(path, kind="diagnostic", diagnostic=None):
    path.mkdir(parents=True)
    key = {"assets": {"identity": {"model": "test"}}, "sample_ms": 100}
    run = {"run_id": "run", "route": "fresh", "kind": kind, "block": 1, "key": key, "diagnostic": diagnostic}
    write(path / "run.json", run)
    reference = {"tokens": [1, 2], "text": "ok", "first_token": 1}
    write(path / "reference.json", reference)
    write(path / "tokens.json", {"benchmark": {"input_ids": [3]}, "eos_token_id": 2, "max_new_tokens": 16})
    audit = {"training": False, "kv_cache": None, "fingerprints": "unchanged"}
    for name in ("audit-before.json", "audit-after.json", "prepared-audit.json"):
        write(path / name, audit)
    records = []
    for i in (1, 2):
        identity = {"run_id": "run", "request_id": str(i)}
        records.append({"request": {**identity, "input_ids": [3]}, "first": {**identity, "token": 1},
            "response": {**identity, "tokens": [1, 2], "text": "ok"}, "start_ns": i * 10,
            "request_start_ns": i * 10, "published_ns": i * 10 + 1, "first_ns": i * 10 + 2, "completed_ns": i * 10 + 3})
    write(path / "observations.json", records)
    write(path / "memory.json", {"rss_bytes": 100, "reserved_vram": 200})
    (path / "resources.jsonl").write_text('{"gpu_used": 200, "rss_bytes": 100}\n')
    key["assets"]["artifacts"] = {source: measure.file_hash(path / saved) for source, saved in
                                (("reference.json", "reference.json"), ("tokens.json", "tokens.json"),
                                 ("audit.json", "prepared-audit.json"))}
    write(path / "run.json", run)
    result = {"status": "passed", "cleanup": "complete", "run_sha256": measure.file_hash(path / "run.json"),
              "durations": measure.durations(records, "run", reference),
              "evidence": {p.name: measure.file_hash(p) for p in path.iterdir() if p.name != "run.json"}}
    write(path / "result.json", result)
    return run, result


class ReportTests(unittest.TestCase):
    def test_external_diagnostics_and_missing_schedule_rows_remain_visible(self):
        """AC7: external diagnostics are revalidated; missing pairs never yield speedups."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            diagnostic = root / "outside"
            run, _ = fixture(diagnostic)
            proof = measure.diagnostic_record(diagnostic, run["key"], "fresh")
            campaign = root / "campaign"
            fixture(campaign / "b1-fresh", "timing", proof)
            schedule = [{"block": b, "route": r, "run": f"b{b}-{r}"} for b in (1, 2, 3) for r in measure.ROUTES]
            write(campaign / "schedule.json", schedule)
            summary = report.summarize(campaign)
            self.assertEqual(len(summary["rows"]), 12)
            self.assertEqual(summary["rows"][0]["status"], "passed")
            self.assertEqual(sum(r["status"] == "not_run" for r in summary["rows"]), 11)
            self.assertEqual(summary["speedups"], {})
            (diagnostic / "result.json").write_text('{}')
            summary = report.summarize(campaign)
            self.assertEqual(summary["rows"][0]["status"], "failed")
            self.assertIn("hash", summary["rows"][0]["reason"])

    def test_changed_keys_audits_or_missing_diagnostics_cannot_pass(self):
        """Critical: provenance and immutable-state checks remain enabled in reporting."""
        for change in ("key", "audit", "missing", "omitted_hash"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                run, _ = fixture(root / "d")
                proof = measure.diagnostic_record(root / "d", run["key"], "fresh")
                trial, result = fixture(root / "t", "timing", proof)
                measure.validate_run(root / "t")
                if change == "key":
                    trial["key"]["gpu"] = "other"
                    write(root / "t/run.json", trial)
                elif change == "audit":
                    write(root / "t/audit-after.json", {})
                elif change == "omitted_hash":
                    result["evidence"].pop("audit-after.json")
                    write(root / "t/result.json", result)
                else:
                    (root / "d/result.json").unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    measure.validate_run(root / "t")

    def test_mixed_model_campaign_is_rejected(self):
        """AC7: a report must never pool different models or host/settings keys."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "one", "timing")
            run, _ = fixture(root / "two", "timing")
            run.update(route="ram")
            run["key"]["assets"]["identity"]["model"] = "different"
            write(root / "two/run.json", run)
            write(root / "schedule.json", [{"block": 1, "route": "fresh", "run": "one"},
                                           {"block": 1, "route": "ram", "run": "two"}])
            with self.assertRaisesRegex(ValueError, "Mixed"):
                report.summarize(root)
