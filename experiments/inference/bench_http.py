"""Client-side TTFT through the HTTP boundary, with cold-cache setup and correctness checks."""

import argparse
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

import cold_cache
import io_stats
import measure
from worker import control


def call(url, method, path, body=None):
    """One new TCP connection per call: no keep-alive, no retries."""
    parts = urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=3600)
    data = json.dumps(body).encode() if body is not None else None
    connection.request(method, path, body=data, headers={"Content-Type": "application/json", "Connection": "close"})
    response = connection.getresponse()
    return connection, response


def request(url, model, prompt, max_new_tokens):
    """Time from before the connection opens to the first token event, in one client clock."""
    body = {"model": model, "prompt": prompt, "max_new_tokens": max_new_tokens, "temperature": 0,
            "request_id": uuid.uuid4().hex}
    events = []
    start = time.monotonic_ns()
    connection, response = call(url, "POST", "/generate", body)
    headers_ns = time.monotonic_ns()
    first_ns = None
    try:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode()}")
        while line := response.readline():
            now = time.monotonic_ns()
            event = json.loads(line)
            if event.get("request_id") != body["request_id"]:
                raise ValueError("Event for another request")
            if event["event"] == "token" and first_ns is None:
                first_ns = now
            events.append({"received_ns": now, **event})
            if event["event"] in ("done", "error"):
                break
    finally:
        response.close()
        connection.close()
    if not events or events[-1]["event"] != "done":
        raise RuntimeError(f"Request ended without done: {events[-1] if events else 'no events'}")
    return {"request": body, "start_ns": start, "headers_ns": headers_ns, "first_ns": first_ns,
            "done_ns": events[-1]["received_ns"], "events": events}


def trial(args):
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True)
    assets = args.assets.resolve()
    manifest, tokens = control.read(assets / "manifest.json"), control.read(assets / "tokens.json")
    reference = control.read(assets / "reference.json")
    record = {"route": args.route, "block": args.block, "url": args.url, "model": args.model,
              "data_cache": args.data_cache, "assets_manifest_sha256": control.file_hash(assets / "manifest.json")}
    connection, response = call(args.url, "POST", f"/admin/models/{args.model}/unload")
    status = json.loads(response.read())
    connection.close()
    if response.status != 200 or status["state"] != "ABSENT":
        raise RuntimeError(f"Could not reset model: {status}")
    record["reset"] = status
    files = [Path(manifest["model_path"]) / name for name in manifest["identity"]["files"] if name.endswith(".safetensors")]
    devices = [manifest["model_path"]]
    if args.snapshot_run:
        files += [p for p in (args.snapshot_run / "snapshot/images").rglob("*") if p.is_file()]
        devices.append(args.snapshot_run)
    if args.data_cache == "cold":
        record["cold_cache"] = cold_cache.evict(files)  # benchmark setup, never an endpoint operation
    io_before = io_stats.sample(devices)
    prompt = tokens["benchmark"]["prompt"]
    primary = request(args.url, args.model, prompt, tokens["max_new_tokens"])
    record["io"] = io_stats.delta(io_before, io_stats.sample(devices))
    followup = request(args.url, args.model, prompt, tokens["max_new_tokens"])
    connection, response = call(args.url, "GET", f"/models/{args.model}")
    record["status_after"] = json.loads(response.read())
    connection.close()
    checks = {}
    for name, observed in (("primary", primary), ("followup", followup)):
        done = observed["events"][-1]
        first = next(e for e in observed["events"] if e["event"] == "token")
        checks[name] = {"tokens_match": done["tokens"] == reference["tokens"], "text_match": done["text"] == reference["text"],
                        "first_token_match": first["token_id"] == reference["first_token"],
                        "same_activation": done["activation_id"] == primary["events"][-1]["activation_id"]}
    record.update(primary=primary, followup=followup, checks=checks,
                  ttft_seconds=(primary["first_ns"] - primary["start_ns"]) / 1e9,
                  headers_seconds=(primary["headers_ns"] - primary["start_ns"]) / 1e9,
                  followup_ttft_seconds=(followup["first_ns"] - followup["start_ns"]) / 1e9,
                  server_phases_ns=primary["events"][-1].get("server_phases_ns"))
    passed = all(all(c.values()) for c in checks.values()) and record["status_after"]["state"] == "READY"
    record["status"] = "passed" if passed else "failed"
    control.write(output / "result.json", record)
    print(json.dumps({k: record[k] for k in ("route", "status", "ttft_seconds", "followup_ttft_seconds")}), flush=True)
    if not passed:
        raise SystemExit(1)


def summarize(args):
    rows = []
    for path in sorted(args.runs.rglob("result.json")):
        record = control.read(path)
        row = {"run": str(path.parent.relative_to(args.runs)), "route": record["route"], "block": record.get("block"),
               "device_read_bytes": sum(d["read_bytes"] for d in record.get("io", []))}
        try:
            # Accepted rows pass the shared validator; failures stay visible with their reason.
            reference = control.read(args.assets / "reference.json")
            row.update(measure.validate_http(record, reference), status="passed")
        except (ValueError, KeyError) as error:
            row.update(status="failed", reason=str(error))
        rows.append(row)
    summary = {"rows": rows, "routes": {}}
    for route in sorted({r["route"] for r in rows}):
        values = sorted(r["ttft_seconds"] for r in rows if r["route"] == route and r["status"] == "passed")
        if values:
            summary["routes"][route] = {"trials": len(values), "median": values[len(values) // 2] if len(values) % 2 else
                                        (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2,
                                        "min": values[0], "max": values[-1]}
    args.output.mkdir(parents=True)
    control.write(args.output / "summary.json", summary)
    print(json.dumps(summary["routes"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("trial")
    run.add_argument("--url", required=True)
    run.add_argument("--model", default="qwen3-8b")
    run.add_argument("--route", choices=("fresh", "snapshot"), required=True)
    run.add_argument("--block", type=int)
    run.add_argument("--assets", type=Path, required=True)
    run.add_argument("--snapshot-run", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--data-cache", choices=("uncontrolled", "cold"), default="cold")
    run.set_defaults(func=trial)
    total = commands.add_parser("summarize")
    total.add_argument("--runs", type=Path, required=True)
    total.add_argument("--assets", type=Path, required=True)
    total.add_argument("--output", type=Path, required=True)
    total.set_defaults(func=summarize)
    arguments = parser.parse_args()
    arguments.func(arguments)
