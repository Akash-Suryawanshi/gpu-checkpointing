"""Admit inference evidence before computing observer-clock durations."""

import hashlib
import json
from pathlib import Path

ROUTES = ("fresh", "resident", "ram", "disk")


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic_record(path, key, route, expected_hash=None):
    path = Path(path)
    result = path / "result.json"
    if expected_hash is not None and file_hash(result) != expected_hash:
        raise ValueError("Diagnostic hash changed")
    run = read(path / "run.json")
    if run["kind"] != "diagnostic" or run["route"] != route or run["key"] != key:
        raise ValueError("Incompatible diagnostic")
    validate_run(path, check_diagnostic=False)
    return {"path": str(path.resolve()), "sha256": file_hash(result)}


def durations(records, run_id, reference):
    previous, seen = -1, set()
    for record in records:
        request, first, response = (record[k] for k in ("request", "first", "response"))
        identifier = request["request_id"]
        if not identifier or identifier in seen:
            raise ValueError("Duplicate request identity")
        seen.add(identifier)
        for item in (request, first, response):
            if item["run_id"] != run_id or item["request_id"] != identifier:
                raise ValueError("Stale or wrong message identity")
        start, published, received, completed = (record[k] for k in
            ("start_ns", "published_ns", "first_ns", "completed_ns"))
        times = (previous, start, record["request_start_ns"], published, received, completed)
        if any(type(t) is not int for t in times) or list(times) != sorted(times):
            raise ValueError("Observer events are out of order")
        previous = completed
        if (not response["tokens"] or first["token"] != response["tokens"][0]
                or first["token"] != reference["first_token"]
                or response["tokens"] != reference["tokens"] or response["text"] != reference["text"]):
            raise ValueError("First/full/reference output differs")
    if len(records) != 2:
        raise ValueError("Primary and health requests required")
    return {"start_to_first_token_seconds": (records[0]["first_ns"] - records[0]["start_ns"]) / 1e9,
            "health_request_to_first_token_seconds": (records[1]["first_ns"] - records[1]["start_ns"]) / 1e9,
            "message_write_seconds": [(r["published_ns"] - r["request_start_ns"]) / 1e9 for r in records]}


def validate_run(path, check_diagnostic=True):
    path = Path(path)
    run, result = read(path / "run.json"), read(path / "result.json")
    if result["status"] != "passed" or result.get("cleanup") != "complete":
        raise ValueError("Incomplete or failed run")
    if result["run_sha256"] != file_hash(path / "run.json"):
        raise ValueError("Run key changed")
    for name, expected in result["evidence"].items():
        if file_hash(path / name) != expected:
            raise ValueError("Evidence changed: " + name)
    if run["kind"] == "timing" and check_diagnostic:
        diagnostic = run["diagnostic"]
        if not isinstance(diagnostic, dict):
            raise ValueError("Diagnostic required")
        diagnostic_record(diagnostic["path"], run["key"], run["route"], diagnostic["sha256"])
    elif run["kind"] != "diagnostic":
        raise ValueError("Diagnostic required")
    reference, records = read(path / "reference.json"), read(path / "observations.json")
    tokens = read(path / "tokens.json")
    for record in records:
        if record["request"]["input_ids"] != tokens["benchmark"]["input_ids"]:
            raise ValueError("Prompt changed")
        generated = record["response"]["tokens"]
        eos = tokens["eos_token_id"]
        eos = eos if isinstance(eos, list) else [eos]
        if not 1 <= len(generated) <= tokens["max_new_tokens"] <= 16 or any(t in eos for t in generated[:-1]):
            raise ValueError("Decoder length/EOS contract violated")
    before, after = read(path / "audit-before.json"), read(path / "audit-after.json")
    if before != after or after != read(path / "prepared-audit.json"):
        raise ValueError("Immutable model state changed")
    if after["training"] or after["kv_cache"] is not None:
        raise ValueError("Invalid idle state")
    if run["route"] in ("ram", "disk"):
        reuse = read(path / "reuse.json")
        if not reuse["passed"] or not reuse["exited"]:
            raise ValueError("GPU reuse incomplete")
    if run["route"] == "disk" and not all(result.get(k) for k in
            ("original_reaped", "capture_exited", "restore_exited", "image_hashes_match")):
        raise ValueError("Disk lifecycle incomplete")
    measured = durations(records, run["run_id"], reference)
    if result["durations"] != measured:
        raise ValueError("Stored duration disagrees with raw events")
    return measured
