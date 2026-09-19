#!/usr/bin/env bash
# Two payload-policy campaigns, each a full campaign under one comparison key.
# Cold runs are the control: the policy touches only the snapshot path.
set -uo pipefail
REPO=/home/ubuntu/gpu-checkpointing
BASE=/home/ubuntu/vllm-h100
PY="$BASE/venv/bin/python"
ASSETS="$BASE/campaigns/qwen25-05b-h100-01/assets"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/huggingface" XDG_CACHE_HOME="$BASE/cache"
export TORCH_HOME="$BASE/cache/torch" CUDA_CACHE_PATH="$BASE/cache/cuda"
export VLLM_CACHE_ROOT="$BASE/cache/vllm"
cd "$REPO"

campaign() {
  local name="$1" policy="$2" workers="$3"
  local C="$BASE/campaigns/$name"
  echo "##### campaign $name: policy=$policy workers=$workers #####"
  for route in snapshot cold; do
    echo "===== $name diagnostic $route ====="
    "$PY" experiments/vllm/trial.py trial --route "$route" --kind diagnostic \
      --data-cache cold --compile-cache warm \
      --payload-policy "$policy" --payload-workers "$workers" \
      --assets "$ASSETS" --output "$C/diagnostics/$route" \
      --tools "$REPO/runs/tools" --timeout 3600 \
      && echo "OK $name diag $route" || echo "FAILED $name diag $route"
  done
  for block in 1 2 3; do
    if (( block % 2 )); then order="snapshot cold"; else order="cold snapshot"; fi
    for route in $order; do
      echo "===== $name block $block $route ====="
      "$PY" experiments/vllm/trial.py trial --route "$route" --kind timing \
        --block "$block" --validated-run "$C/diagnostics/$route" \
        --data-cache cold --compile-cache warm \
        --payload-policy "$policy" --payload-workers "$workers" \
        --assets "$ASSETS" --output "$C/timing/$route-$block" \
        --tools "$REPO/runs/tools" --timeout 3600 \
        && echo "OK $name block $block $route" || echo "FAILED $name block $block $route"
    done
  done
  # Keep manifests, logs and results; the image payload is not reused.
  rm -rf "$C"/diagnostics/*/snapshot/images "$C"/timing/*/snapshot/images
  echo "##### campaign $name complete #####"
  df -h / | tail -1
}

campaign qwen25-05b-h100-skipval publication-only-v1 1
campaign qwen25-05b-h100-parallel parallel-chunked-v1 16
echo "##### all policy campaigns complete #####"
