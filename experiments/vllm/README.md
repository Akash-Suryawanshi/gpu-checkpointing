# Running the vLLM snapshot comparison

**Complete on Qwen2.5-0.5B: cold 27.65 s, snapshot 30.31 s.** The
[plan](../../docs/vllm-snapshot-plan.md) owns the requirements and the stop
conditions; this file owns the commands. Capture, restore, eviction and
ownership are reused from the [inference runbook](../inference/README.md).

```text
assets -> gate 2 probe (0.5B) -> diagnostics (cold, eager, snapshot) -> timing blocks
             capture/restore            one per route                  three per route
```

## Environment

vLLM needs its own interpreter: from 0.20 on it pins a CUDA 13 build of PyTorch,
which requires driver 580 or newer. This host has 570.172.08, so the newest
usable release is **vLLM 0.19.1** with torch 2.10.0 (CUDA 12.8).

```bash
BASE=/opt/dlami/nvme/gpu-checkpointing-vllm
python3 -m venv "$BASE/venv"
TMPDIR="$BASE/tmp" "$BASE/venv/bin/python" -m pip install "vllm==0.19.1"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/huggingface" XDG_CACHE_HOME="$BASE/cache"
export TORCH_HOME="$BASE/cache/torch" CUDA_CACHE_PATH="$BASE/cache/cuda"
export VLLM_CACHE_ROOT="$BASE/cache/vllm"
PY="$BASE/venv/bin/python"
TOOLS="$PWD/runs/tools"
```

Images go on instance storage: EBS has 19 GiB free, which will not hold one.

## Command contract

| Program | Arguments |
| --- | --- |
| `assets.py` | `--model PATH --source PATH --output PATH --gpu-fraction F [--max-model-len N]`. Records the reference tokens vLLM itself produces. |
| `trial.py trial` | `--route cold\|eager\|snapshot --kind diagnostic\|timing --assets PATH --output PATH --tools PATH --timeout SECONDS`. |
| Timing options | `--validated-run PATH --block 1\|2\|3`. |
| Conditions | `--data-cache uncontrolled\|cold`, `--compile-cache warm\|cold`, `--poll-ms 5`, `--sample-ms 100`. |

Output directories must be new. The route is not a comparison-key setting, so
the three routes of one campaign share a key and can be aggregated; every other
setting must be identical across them.

## Prepare

```bash
MODEL="$PWD/runs/tools/qwen2.5-0.5b"        # gate 2 probe; 8B for the headline
CAMPAIGN=/opt/dlami/nvme/gpu-checkpointing-vllm/campaigns/qwen25-05b-a10g-01
ASSETS="$CAMPAIGN/assets"
mkdir -p "$CAMPAIGN"
"$PY" experiments/vllm/assets.py --model "$MODEL" \
  --source "$MODEL/asset-manifest.json" --output "$ASSETS" --gpu-fraction 0.30
```

## Gate 2: capture and restore once

```bash
"$PY" experiments/vllm/trial.py trial --route snapshot --kind diagnostic \
  --assets "$ASSETS" --output "$CAMPAIGN/diagnostics/snapshot" \
  --tools "$TOOLS" --timeout 3600
```

This passes. Nothing extra is needed to make it pass: `engine.py` sets
`USE_LIBUV=0` and `session.py` dumps with `--tcp-established`, which together
remove the io_uring ring and the self-connected socket that CRIU refuses. Run
the `cold` diagnostic the same way before any timing.

Run both diagnostics with the same `--data-cache` setting the timings will use:
the condition is part of the comparison key, so a cold timing needs a cold
diagnostic. It matters more than it looks — an image written moments earlier is
still in the page cache, and a restore that reads it from there measures nothing.

## Timing blocks

Each timed run cites its route's accepted diagnostic. Warm the compiled-kernel
cache once with a throwaway activation; `--compile-cache warm` refuses an empty
cache, because a first block that had to compile would be charged for work the
other two inherit.

```bash
for block in 1 2 3; do
  for route in cold eager snapshot; do
    "$PY" experiments/vllm/trial.py trial --route "$route" --kind timing \
      --block "$block" --validated-run "$CAMPAIGN/diagnostics/$route" \
      --data-cache cold --compile-cache warm \
      --assets "$ASSETS" --output "$CAMPAIGN/timing/$route-$block" \
      --tools "$TOOLS" --timeout 3600
  done
done
```

## Tests

```bash
PYTHONPATH=experiments/inference:experiments/finetuning:experiments/criu:experiments/cpu \
  python3 -m unittest discover -s tests -v
```

## Running on other hardware

The result here is storage-bound, so another host can reverse it. Re-derive
these before trusting any comparison on a new machine.

| Re-derive | Why | How |
| --- | --- | --- |
| vLLM version | wheels past 0.19.1 pin a CUDA 13 torch needing driver 580+ | read the wheel's `Requires-Dist` for `nvidia-*-cu12` against `cu13` |
| volume bandwidth | it decides the answer | read the raw device, as in the [bandwidth measurement](../results.md#both-volumes-are-bandwidth-limited-and-our-loaders-already-saturate-them--2026-09-18) |
| `--gpu-fraction` | the preallocated key-value cache is image bytes | keep it low, or unmap the cache before capture |
| compiled-kernel cache | `warm` refuses an empty cache | one throwaway activation first |
| CRIU tools | built per host | rebuild under `runs/tools` |

The snapshot wins when reading the image costs less than the startup it removes:

```text
image_bytes / bandwidth  +  restore overhead   <   cold TTFT
  6.42 GiB / 0.21 GiB/s  +  ~0 s               <   27.65 s      -> 30.6 s, lost
  6.42 GiB / 2.00 GiB/s  +  ~3 s               <   27.65 s      -> 6.2 s, won
```

Run a fresh campaign directory and fresh assets: assets record the engine
configuration, and images are bound to the checkout path that produced them.
