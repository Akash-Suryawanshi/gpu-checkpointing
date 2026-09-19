"""List the external files a published CRIU image reopens, grouped by role.

Validation hashes model files at every activation. This reports whether the
restored process actually opens them, which decides if that cost is a
correctness prerequisite or an environment-equivalence check.
"""

import argparse
import json
from pathlib import Path
import re

# CRIU's verbose restore log names each regular file it will reopen.
COLLECTED = re.compile(r"Collected \[([^\]]*)\]")


def collected(log):
    """Absolute paths of every regular file the restore stage reopened."""
    return sorted({"/" + name.lstrip("/") for name in COLLECTED.findall(Path(log).read_text())})


def classify(paths, model_path, run):
    model_path, run = str(Path(model_path).resolve()), str(Path(run).resolve())
    groups = {"model_files": [], "run_files": [], "libraries": [], "other": []}
    for path in paths:
        if path.startswith(model_path + "/") or path.endswith(".safetensors"):
            groups["model_files"].append(path)
        elif path.startswith(run + "/"):
            groups["run_files"].append(path)
        elif path.endswith(".so") or ".so." in path or path.endswith((".dll", ".dylib")):
            groups["libraries"].append(path)
        else:
            groups["other"].append(path)
    return groups


def audit(run):
    """Join one attempt's restore log with the manifest's hashed dependencies."""
    run = Path(run).resolve()
    manifest = json.loads((run / "snapshot/manifest.json").read_text())
    logs = sorted((run / "attempts").glob("*/restore.log"))
    if not logs:
        raise ValueError("No restore log: the image has not been activated from this run")
    model_path = json.loads((Path(manifest["job"]["assets"]) / "manifest.json").read_text())["model_path"]
    groups = classify(collected(logs[-1]), model_path, run)
    hashed = [name for name in manifest["dependencies"]["files"] if name.startswith(str(model_path))]
    hashed_bytes = sum(Path(name).stat().st_size for name in hashed if Path(name).exists())
    return {"restore_log": str(logs[-1]), "reopened_total": sum(len(v) for v in groups.values()),
            "reopened": {key: len(value) for key, value in groups.items()},
            "model_files_reopened": groups["model_files"], "run_files_reopened": groups["run_files"],
            "model_files_hashed_at_activation": len(hashed), "model_bytes_hashed_at_activation": hashed_bytes,
            "conclusion": ("Model files are hashed but never reopened: that read is an environment-equivalence "
                           "check, not a restore prerequisite." if not groups["model_files"] else
                           "The restored process reopens model files; activation-time verification is a prerequisite.")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
