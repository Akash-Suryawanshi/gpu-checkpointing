#!/usr/bin/env bash
# Manual NVIDIA transitions only: no CRIU or full-process restoration here.
set -euo pipefail
umask 077
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/probe-gpu.sh /path/to/cuda-checkpoint /path/to/new-run-dir' >&2
  exit 2
fi
cuda_tool="$(realpath "$1")"
run_dir="$(realpath -m "$2")"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
pid=''
cleanup() {
  status=$?
  if [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] &&
     tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fxq -- "$root/gpu_memory_probe.py"; then
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "GPU probe failed; retained evidence: $run_dir" >&2
  fi
}
trap cleanup EXIT
wait_marker() {
  local marker="$1"
  local deadline=$((SECONDS + 60))
  until [[ -f "$run_dir/$marker" ]]; do
    kill -0 "$pid" 2>/dev/null || { cat "$run_dir/process.stderr" >&2; return 1; }
    (( SECONDS < deadline )) || { echo "Timed out: $marker" >&2; return 1; }
    sleep 0.1
  done
}
observe() {
  local stage="$1"
  "$cuda_tool" --get-state --pid "$pid" > "$run_dir/$stage.cuda-state.txt"
  grep -E '^(VmRSS|VmHWM):' "/proc/$pid/status" > "$run_dir/$stage.host-memory.txt"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$run_dir/$stage.gpu-processes.csv"
  cat "$run_dir/$stage.cuda-state.txt"
}
transition() {
  local action="$1"
  python3 -c 'import time; print(time.monotonic_ns())' > "$run_dir/$action.start-ns"
  timeout --kill-after=5 30 "$cuda_tool" --action "$action" --pid "$pid" > "$run_dir/$action.log" 2>&1
  python3 -c 'import time; print(time.monotonic_ns())' > "$run_dir/$action.end-ns"
}
"$cuda_tool" --help > "$run_dir/tool-help.txt"
nvidia-smi > "$run_dir/nvidia-smi.txt"
python3 "$root/gpu_memory_probe.py" --run-dir "$run_dir" > "$run_dir/process.jsonl" 2> "$run_dir/process.stderr" &
pid=$!
printf '%s\n' "$pid" > "$run_dir/pid.txt"
wait_marker ready
observe before
transition lock
transition checkpoint
observe suspended
# B runs only after A reports checkpointed; it exits before A is restored.
grep -Fxq checkpointed "$run_dir/suspended.cuda-state.txt"
timeout --kill-after=5 60 python3 -c 'import torch; x=torch.ones(1024 * 1024, device="cuda"); value=x.sum().item(); assert value == 1024 * 1024; torch.cuda.synchronize(); print("job_B_gpu_sum", value)' > "$run_dir/job-b.txt" 2>&1
transition restore
transition unlock
observe restored
touch "$run_dir/release"
wait_marker done
wait "$pid"
pid=''
cat "$run_dir/process.jsonl"
cat "$run_dir/job-b.txt"
echo "GPU-only probe passed: $run_dir"
