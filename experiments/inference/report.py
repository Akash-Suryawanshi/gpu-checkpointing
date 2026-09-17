"""Read-only schedule-aware inference report; accepted evidence uses one validator."""

import argparse
import csv
import html
import json
from pathlib import Path
import statistics

import measure


def summarize(root):
    schedule = measure.read(root / "schedule.json")
    seen, rows, keys = set(), [], set()
    for slot in schedule:
        pair = slot["block"], slot["route"]
        if pair in seen or pair[0] not in (1, 2, 3) or pair[1] not in measure.ROUTES:
            raise ValueError("Duplicate or invalid schedule slot")
        seen.add(pair)
        path = (root / slot["run"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Schedule run must be inside campaign")
        row = {**slot, "status": "not_run"}
        if path.exists():
            row["status"] = "failed"
            try:
                run = measure.read(path / "run.json")
                if (run["block"], run["route"]) != pair or run["kind"] != "timing":
                    raise ValueError("Run does not match scheduled slot")
                keys.add(measure.digest(run["key"]))
                result = measure.read(path / "result.json")
                if result["status"] != "passed":
                    raise ValueError(result.get("reason", "Incomplete result"))
                row.update(measure.validate_run(path))
                row.update(status="passed", model=run["key"]["assets"]["identity"]["model"],
                           image_bytes=result.get("image_bytes", 0), preparation_seconds=result.get("preparation_seconds"),
                           save_seconds=result.get("save_seconds"))
                memory = measure.read(path / "memory.json")
                row.update(rss_bytes=memory["rss_bytes"], reserved_vram=memory["reserved_vram"])
                samples = [json.loads(line) for line in (path / "resources.jsonl").read_text().splitlines()]
                valid = [sample for sample in samples if "error" not in sample]
                row.update(sample_interval_ms=run["key"]["sample_ms"],
                           observed_peak_gpu_bytes=max((s["gpu_used"] for s in valid), default=0),
                           observed_peak_rss_bytes=max((s.get("rss_bytes", 0) for s in valid), default=0))
                if run["route"] in ("ram", "disk"):
                    row["reuse_claim"] = measure.read(path / "reuse.json")["claim"]
                    row["idle_host_rss_bytes"] = measure.read(path / "released-resources.json").get("rss_bytes", 0)
            except (ValueError, KeyError, FileNotFoundError) as error:
                row.update(status="failed", reason=str(error))
        rows.append(row)
    if len(keys) > 1:
        raise ValueError("Mixed model/host/settings campaigns cannot be compared")
    aggregates, speedups = {}, {}
    for route in measure.ROUTES:
        good = [r for r in rows if r["route"] == route and r["status"] == "passed"]
        if good:
            values = [r["start_to_first_token_seconds"] for r in good]
            aggregates[route] = {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
        pairs = []
        for block in (1, 2, 3):
            fresh = next((r for r in rows if r["route"] == "fresh" and r["block"] == block and r["status"] == "passed"), None)
            other = next((r for r in good if r["block"] == block), None)
            if fresh and other and other["start_to_first_token_seconds"] > 0:
                pairs.append(fresh["start_to_first_token_seconds"] / other["start_to_first_token_seconds"])
        if route != "fresh" and len(pairs) == 3:
            speedups[route] = {"paired": pairs, "median": statistics.median(pairs), "min": min(pairs), "max": max(pairs)}
    return {"rows": rows, "aggregates": aggregates, "speedups": speedups,
            "limitations": "Uncontrolled local file cache; three trials do not establish tail latency. "
                            "Sampled peaks can miss short spikes. Same-host recovery is not host-loss recovery."}


def chart(values, title, unit):
    maximum = max((value for _, value in values), default=1) or 1
    height = 75 + 38 * len(values)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 {height}" role="img">',
             f'<title>{html.escape(title)}</title><rect width="100%" height="100%" fill="#faf9f6"/>',
             f'<text x="10" y="25" font-family="Georgia" font-size="19">{html.escape(title)} ({unit})</text>']
    for index, (label, value) in enumerate(values):
        y = 50 + index * 38
        parts.append(f'<text x="10" y="{y + 18}" font-size="15">{html.escape(label)}</text>'
                     f'<rect x="235" y="{y}" width="{480 * value / maximum:.2f}" height="24" fill="#54768a"/>'
                     f'<text x="730" y="{y + 18}" font-size="15">{value:.3f}</text>')
    if not values:
        parts.append('<text x="10" y="55">No accepted measurements</text>')
    return "".join(parts) + "</svg>"


def main(args):
    summary = summarize(args.runs)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in summary["rows"] for key in row))
    with (args.output / "runs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["rows"])
    latency = [(f"block {r['block']} {r['route']}", r["start_to_first_token_seconds"])
               for r in summary["rows"] if r["status"] == "passed"]
    resources = [(f"{route} idle host RSS", statistics.median(r.get("idle_host_rss_bytes", r["rss_bytes"])
                    for r in summary["rows"] if r["route"] == route and r["status"] == "passed") / 1024**3)
                 for route in measure.ROUTES if route in summary["aggregates"]]
    for name, svg in (("latency.svg", chart(latency, "Observer start to first token", "seconds")),
                      ("resources.svg", chart(resources, "Median retained host memory", "GiB"))):
        (args.output / name).write_text(svg)
    rows = "".join(f"<tr><td>{r['block']}</td><td>{r['route']}</td><td>{r['status']}</td><td>"
                   + (f"{r['start_to_first_token_seconds']:.4f}" if r['status'] == 'passed' else html.escape(r.get('reason', '')))
                   + "</td></tr>" for r in summary["rows"])
    stats = "".join(f"<tr><td>{route}</td><td>{s['n']}</td><td>{s['median']:.4f}</td><td>{s['min']:.4f}–{s['max']:.4f}</td></tr>"
                    for route, s in summary["aggregates"].items())
    speedups = " ".join(f"{route}: median {s['median']:.2f}× (range {s['min']:.2f}–{s['max']:.2f}×)."
                        for route, s in summary["speedups"].items()) or "No aggregate speedup: three complete matching block pairs are required."
    (args.output / "index.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning inference snapshots</title>
<style>body{{font:19px/1.7 Georgia,'Times New Roman',serif;background:#faf9f6;color:#242424;max-width:900px;margin:40px auto;padding:0 20px}}h1,h2{{font-weight:400}}a{{color:#54768a}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{text-align:left;border-bottom:1px solid #ccc;padding:8px;overflow-wrap:anywhere}}p{{margin:1.4em 0}}</style>
<h1>Learning inference snapshots</h1><p>Each accepted time ends when the controller receives token one.
Fresh starts before launching a worker; resident starts before publishing its request;
RAM starts before restoring GPU state; disk starts before launching the independent restore command, including validation.</p>
<figure><img src="latency.svg" alt="Individual accepted latency measurements"><figcaption>Durable file messages use the recorded poll interval; inspection is never subtracted.</figcaption></figure>
<table><tr><th>Block</th><th>Route</th><th>Status</th><th>Seconds or failure</th></tr>{rows}</table>
<table><tr><th>Route</th><th>Count</th><th>Median seconds</th><th>Min–max seconds</th></tr>{stats}</table>
<p>{html.escape(speedups)}</p><figure><img src="resources.svg" alt="Host memory retained while idle"><figcaption>RAM parking keeps a live CPU process. Disk can end that process but retains image files and external dependencies.</figcaption></figure>
<p>Job B's reuse claim, preparation and save durations, sampled GPU/RSS peaks, image bytes, and the separate health-request latency are in the downloadable records. A small availability probe does not prove a previously impossible allocation.</p>
<p>{html.escape(summary['limitations'])}</p><p><a href="summary.json">Summary JSON</a> · <a href="runs.csv">All rows as CSV</a></p></html>''')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
