#!/usr/bin/env bash
# The vLLM cold-vs-snapshot campaign on the H100 host. One comparison key:
# every run gets the identical environment, because the cache paths and the
# engine environment are part of that key. See experiments/vllm/README.md.
set -euo pipefail

REPO=/home/ubuntu/gpu-checkpointing
BASE=/home/ubuntu/vllm-h100
CAMPAIGN="$BASE/campaigns/qwen25-05b-h100-01"
MODEL="$REPO/runs/tools/qwen2.5-0.5b"
ASSETS="$CAMPAIGN/assets"
TOOLS="$REPO/runs/tools"
PY="$BASE/venv/bin/python"
FRACTION=0.063          # 9067 KV blocks, matching the A10G campaign's ~9100

# Identical for every run: these paths enter the comparison key.
export TMPDIR="$BASE/tmp"
export HF_HOME="$BASE/cache/huggingface"
export XDG_CACHE_HOME="$BASE/cache"
export TORCH_HOME="$BASE/cache/torch"
export CUDA_CACHE_PATH="$BASE/cache/cuda"
export VLLM_CACHE_ROOT="$BASE/cache/vllm"

cd "$REPO"

step() { printf '\n===== %s =====\n' "$*"; }

step "assets (gpu-fraction $FRACTION)"
"$PY" experiments/vllm/assets.py --model "$MODEL" \
  --source "$MODEL/asset-manifest.json" --output "$ASSETS" --gpu-fraction "$FRACTION"

# Diagnostics carry the same --data-cache the timings use: the condition is part
# of the comparison key, so a cold timing needs a cold diagnostic.
for route in snapshot cold; do
  step "diagnostic: $route"
  "$PY" experiments/vllm/trial.py trial --route "$route" --kind diagnostic \
    --data-cache cold --compile-cache warm \
    --assets "$ASSETS" --output "$CAMPAIGN/diagnostics/$route" \
    --tools "$TOOLS" --timeout 3600
done

# Three blocks per route, alternating which route goes first in each block.
for block in 1 2 3; do
  if (( block % 2 )); then order="snapshot cold"; else order="cold snapshot"; fi
  for route in $order; do
    step "timing: block $block, route $route"
    "$PY" experiments/vllm/trial.py trial --route "$route" --kind timing \
      --block "$block" --validated-run "$CAMPAIGN/diagnostics/$route" \
      --data-cache cold --compile-cache warm \
      --assets "$ASSETS" --output "$CAMPAIGN/timing/$route-$block" \
      --tools "$TOOLS" --timeout 3600
  done
done

step "campaign complete"
