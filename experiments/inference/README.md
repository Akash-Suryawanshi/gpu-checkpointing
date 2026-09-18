# Running the inference snapshot comparison

**Local validation passed with pinned Qwen3-8B on A10G.** All four routes, three
timing blocks, and separate capture/restore commands passed; see the
[measured results](../results.md#inference-activation-and-gpu-reuse--2026-09-17).
The [plan](../../docs/inference-snapshot-plan.md) retains the acceptance contract.
The [cold-start experiments](cold-start.md) add verified cold file caches,
CRIU phase profiling, and ServerlessLLM-inspired packed/pipelined weight loaders.

```text
prepare -> CPU checks -> GPU diagnostics -> timing blocks -> static report
                                 |
                     live park/use/wake + staged disk recovery
```

Run from the repository root on an otherwise idle GPU. Failed attempts remain
in the campaign; change the campaign name after changing code, environment, or model.

## Command contract

| Program | Arguments |
| --- | --- |
| `prepare_assets.py` | Required: `--model PATH --source PATH --output PATH`. Source JSON contains `repo` and pinned `revision`. |
| `run.py trial` | Required: `--route fresh\|resident\|ram\|disk --kind diagnostic\|timing --assets PATH --output PATH --tools PATH --timeout SECONDS`. |
| Timing options | Required: `--validated-run PATH --block 1\|2\|3`. Optional `--discard-image-after-success` for accepted disk timings only. |
| Disk options | `--disk-action full\|capture\|restore`, default `full`; staged actions are diagnostic-only and share the capture run's output directory. |
| Observation options | `--poll-ms 5 --sample-ms 100` defaults; polling applies to inference request/marker waits. Include actual values in compatibility checks. |
| `report.py` | Required: `--runs PATH --output NEW_PATH`. Reads `schedule.json`, attempts, and their diagnostic evidence. |

Output directories must be new except the explicit staged disk `restore` action.
All deadlines include subprocesses and waits. A run exits nonzero on failure and
records whether cleanup completed; an unsafe cleanup failure stops further GPU work.

## Prepare

Select the already-downloaded 8B model first; no download occurs inside a measured run.

```bash
BASE=/mnt/data/gpu-checkpointing-inference-8b
MODEL="$BASE/model"
SOURCE="$BASE/source.json"
LABEL=qwen3-8b-bf16-a10g-01
```

If capacity blocks 8B, replace those four variables with the smaller-model selection:

```bash
BASE=/mnt/data/gpu-checkpointing-inference-05b
MODEL="$PWD/runs/tools/qwen2.5-0.5b"
SOURCE="$MODEL/asset-manifest.json"
LABEL=qwen25-05b-bf16-a10g-01
```

Then prepare the selected model on the data volume:

```bash
PY="$PWD/runs/finetuning/venv/bin/python"
TOOLS="$PWD/runs/tools"
ASSETS="$BASE/assets-v2"
CAMPAIGN="$BASE/campaigns/$LABEL"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/huggingface"
export XDG_CACHE_HOME="$BASE/cache" TORCH_HOME="$BASE/cache/torch"
export CUDA_CACHE_PATH="$BASE/cache/cuda"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH"
mkdir -p "$(dirname "$CAMPAIGN")"
mkdir "$CAMPAIGN"
df -h / "$BASE"
nvidia-smi
"$PY" experiments/inference/prepare_assets.py \
  --model "$MODEL" --source "$SOURCE" --output "$ASSETS"
```

Preparation and child workers use the explicit cache-path allowlist. The recorded
8B preparation passed at `assets-v2`; earlier partial directories remain as failure
evidence. Use a new asset path after changing preparation inputs, and a new campaign
after changing sources or environment.

Run the entire diagnostic matrix for each model; never reuse 8B reference tokens
or timings for the smaller-model campaign.

## CPU checks and diagnostics

Run the existing test suite with all shared import roots available:

```bash
PYTHONPATH=experiments/inference:experiments/finetuning:experiments/criu:experiments/cpu \
  "$PY" -m unittest discover -s tests -v
```

Run each diagnostic sequentially. Inspect any failure and cleanup status before
continuing; the larger timeout permits disk I/O but does not waive admission checks.

```bash
mkdir "$CAMPAIGN/diagnostics"
diag() {
  "$PY" experiments/inference/run.py trial \
    --route "$1" --kind diagnostic --assets "$ASSETS" \
    --output "$CAMPAIGN/diagnostics/$2" --tools "$TOOLS" --timeout "$3"
}
diag fresh fresh-1 900 && diag fresh fresh-2 900 && \
  diag resident resident 900 && diag ram ram 1800 && diag disk disk 3600
```

Require all diagnostics to pass before timing. A smaller model may prove GPU
availability with the 256 MiB job B probe without proving an allocation that
would have been impossible before parking; the report must distinguish those claims.

## Three timing blocks

Publish the intended schedule first so interrupted and failed attempts stay visible.
The schedule's `run` paths are relative to the campaign directory.

```bash
"$PY" - "$CAMPAIGN" <<'PY'
import json
from pathlib import Path
import sys

campaign = Path(sys.argv[1])
orders = ("fresh resident ram disk", "resident ram disk fresh", "disk fresh resident ram")
schedule = [{"block": block, "route": route, "run": f"timing/b{block}-{route}"}
            for block, order in enumerate(orders, 1) for route in order.split()]
with (campaign / "schedule.json").open("x") as output:
    json.dump(schedule, output, indent=2)
PY
mkdir "$CAMPAIGN/timing"
time_route() {
  local block="$1" route="$2" validation="$2" timeout=900
  local image_args=()
  if [ "$route" = fresh ]; then validation=fresh-1; fi
  if [ "$route" = ram ]; then timeout=1800; fi
  if [ "$route" = disk ]; then
    timeout=3600
    image_args=(--discard-image-after-success)
  fi
  "$PY" experiments/inference/run.py trial \
    --route "$route" --kind timing --assets "$ASSETS" \
    --output "$CAMPAIGN/timing/b$block-$route" --tools "$TOOLS" \
    --timeout "$timeout" --validated-run "$CAMPAIGN/diagnostics/$validation" \
    --block "$block" "${image_args[@]}"
}
run_blocks() {
  local block route
  for block in 1 2 3; do
    case "$block" in
      1) local order="fresh resident ram disk" ;;
      2) local order="resident ram disk fresh" ;;
      3) local order="disk fresh resident ram" ;;
    esac
    for route in $order; do
      time_route "$block" "$route" || return 1
    done
  done
}
run_blocks
```

The batch stops on failure. Generate the report even after interruption; continue
only after cleanup is verified, keeping the failed attempt and using only the
remaining scheduled slots. Never rerun an occupied slot to replace a failure.

```bash
"$PY" experiments/inference/report.py \
  --runs "$CAMPAIGN" --output "$CAMPAIGN/report"
```

Open `report/index.html` as a static file; it needs no application server.
Keep `summary.json`, `runs.csv`, `latency.svg`, and `resources.svg` alongside it.
A subsequent report build uses a new output path, preserving the previous report.

## Live sequences

In a second terminal, observe allocations with `watch -n 0.5 nvidia-smi`.
This display supports the explanation; saved resource samples and job B's checked
calculation provide the evidence.

```bash
mkdir "$CAMPAIGN/live"
"$PY" experiments/inference/run.py trial \
  --route ram --kind diagnostic --assets "$ASSETS" \
  --output "$CAMPAIGN/live/ram" --tools "$TOOLS" --timeout 1800
```

The controller prints the worker identity, parked state, released GPU memory,
job B result, wake/inspection, first token, and second-response check.
For disk, explicitly end the capture invocation before starting the restore:

```bash
"$PY" experiments/inference/run.py trial \
  --route disk --kind diagnostic --disk-action capture --assets "$ASSETS" \
  --output "$CAMPAIGN/live/disk" --tools "$TOOLS" --timeout 3600
# Continue only after successful publication, original exit, and job B completion.
"$PY" experiments/inference/run.py trial \
  --route disk --kind diagnostic --disk-action restore --assets "$ASSETS" \
  --output "$CAMPAIGN/live/disk" --tools "$TOOLS" --timeout 3600
```

Show the capture record and confirm no model worker remains between commands.
After restore, show matching output and continued operation after the restore
helper exits. This validates independent commands on the same host, not host-loss recovery.

## Request-triggered endpoint

**Host validation pending.** The endpoint serves the same worker protocol the
CLI validated; `runtime.py` is shared. Fresh versus snapshot is server
configuration, and clients send identical requests.

```text
bench_http.py --POST /generate--> api.py --Runtime--> worker (fresh launch | reusable restore)
   START before connect            no CUDA state        tokens-N.jsonl tailed per event
   STOP at first token event  <-- ndjson token events <--+
```

Publish a reusable image first: a passed disk diagnostic, then a staged timing
capture with the contract. The image stays in that run directory across activations.

```bash
"$PY" experiments/inference/run.py trial --route disk --kind diagnostic \
  --assets "$ASSETS" --output "$CAMPAIGN/diagnostics/disk" --tools "$TOOLS" --timeout 3600
"$PY" experiments/inference/run.py trial --route disk --kind timing --disk-action capture \
  --contract reusable-inference-v1 --validated-run "$CAMPAIGN/diagnostics/disk" --block 1 \
  --assets "$ASSETS" --output "$CAMPAIGN/reusable" --tools "$TOOLS" --timeout 3600
```

Serve one route per server process, then measure from a separate client. The
client resets the model, evicts the selected files, times a new connection to
the first token event, checks the exact reference output, and sends a warm follow-up.

```bash
"$PY" experiments/inference/api.py --route snapshot --snapshot-run "$CAMPAIGN/reusable" \
  --assets "$ASSETS" --tools "$TOOLS" --root "$CAMPAIGN/http/server-snapshot" --port 8090
"$PY" experiments/inference/bench_http.py trial --url http://127.0.0.1:8090 --route snapshot \
  --assets "$ASSETS" --snapshot-run "$CAMPAIGN/reusable" --block 1 --data-cache cold \
  --output "$CAMPAIGN/http/b1-snapshot"
"$PY" experiments/inference/bench_http.py summarize --runs "$CAMPAIGN/http" --assets "$ASSETS" --output "$CAMPAIGN/http/summary"
```

For `--route fresh`, omit `--snapshot-run`. `POST /admin/models/{id}/unload`
resets between trials; `GET /models/{id}` reports state and the last error.
One request at a time: concurrent requests get `429`, nonzero temperature `400`.

Choose a port nothing else holds. `GET /healthz` returns the server's own
identifier, model, and route, and the client refuses to trial an endpoint that
does not identify itself; a plain listener answering on the port is not a ready
model server. The server prints its identifier when it starts.

Start the server before evicting caches. It fingerprints the model once at
startup, which reads every weight file; doing that inside a request would both
inflate the measurement and warm the cache the trial just evicted. The full
weight audit is verified at unload, after the response pair, so neither request
waits for it.

## Container validation

Requires Docker with NVIDIA Container Toolkit configured on the host. The image
fixes the Ubuntu base digest, Python package versions, and CRIU/NVIDIA source commits. Ubuntu build dependencies follow the package repository; their
archive hashes are saved in `/opt/tools/package-hashes.txt`.

```text
host model files --read-only bind mount--> /model
host results disk ----read/write mount--> /data
pinned image + host NVIDIA driver ------> CPU/GPU probes -> model diagnostics
```

Bind mounts expose host files at fixed container paths; they do not copy the files.
Keep these paths stable for recovery. The host supplies the GPU driver through
NVIDIA Container Toolkit; no model files or saved images enter the image build.

The tested host lacked root-volume space. In a dedicated terminal, start an
isolated Docker daemon on the data volume; this leaves the existing daemon alone:

```bash
DOCKER_DATA=/mnt/data/gpu-checkpointing-inference-container
mkdir -p "$DOCKER_DATA/tmp"
sudo env DOCKER_TMPDIR="$DOCKER_DATA/tmp" dockerd \
  --config-file /etc/docker/daemon.json --data-root "$DOCKER_DATA/docker" \
  --exec-root /tmp/inference-docker-exec --host unix:///tmp/inference-docker.sock \
  --pidfile /tmp/inference-docker.pid --bridge none --iptables=false \
  --ip-forward=false --ip-masq=false
```

In the working terminal, build from the repository root and enter the container:

```bash
export DOCKER_HOST=unix:///tmp/inference-docker.sock
mkdir -p /mnt/data/gpu-checkpointing-inference-container/runtime-01
docker build --network=host -f experiments/inference/Dockerfile \
  -t gpu-checkpointing-inference:validated .
docker run --rm -it --runtime=nvidia --gpus all --privileged \
  --pid=host --cgroupns=host --shm-size=1g --network=none \
  --mount type=bind,src=/mnt/data/gpu-checkpointing-inference-8b/model,dst=/model,readonly \
  --mount type=bind,src=/mnt/data/gpu-checkpointing-inference-8b/source.json,dst=/source.json,readonly \
  --mount type=bind,src=/mnt/data/gpu-checkpointing-inference-container/runtime-01,dst=/data \
  gpu-checkpointing-inference:validated
```

Validation used privileged kernel access, the host's process-ID namespace, and
its cgroup view (resource limits). This records a working permission set, not a
minimum-permission deployment profile or evidence of cross-host compatibility.
The network-disabled run emitted `sudo` hostname-resolution warnings; the commands
and restore probes still succeeded.

Inside the container, record `nvidia-smi`, `uname -a`, `df -h`, `/proc/meminfo`, and
`/proc/self/cgroup`, and `dpkg-query -W`. Require these probes to pass before preparing a model:

```bash
sudo unshare --mount --pid --fork --net true
python experiments/criu/probe.py cpu --tools /opt/tools --output /data/probe-cpu
python experiments/criu/probe.py gpu --tools /opt/tools --output /data/probe-gpu --timeout 300
```

Use the preparation and diagnostic commands above with `PY=/opt/venv/bin/python`,
`TOOLS=/opt/tools`, `BASE=/data`, `MODEL=/model`, `SOURCE=/source.json`, a new asset
path, and a new campaign. Export the same five cache variables under `/data` first.
Never reuse host diagnostics for this image or pool container diagnostics with
host timings. Repeat probes and diagnostics after changing the target environment.
