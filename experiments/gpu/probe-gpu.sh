#!/usr/bin/env bash
# Manual NVIDIA transitions only: no CRIU or full-process restoration here.
# The driver stages GPU state in this still-living process's CPU RAM. This probe
# does not create an on-disk process image or demonstrate recovery from its exit.
set -euo pipefail
umask 077
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash experiments/gpu/probe-gpu.sh /path/to/cuda-checkpoint /path/to/new-run-dir' >&2
  exit 2
fi
cuda_tool="$(realpath "$1")"
run_dir="$(realpath -m "$2")"
mkdir -p "$(dirname "$run_dir")"
# Reject reuse: an old release marker would let the workload continue too early.
mkdir "$run_dir"
pid=''
cleanup() {
  # The EXIT trap runs on success and failure. /proc is Linux's live process view;
  # check the command before forcefully killing a leftover child, then collect
  # its exit status with wait. A PID can be reused, so its number alone is unsafe.
  status=$?
  if [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] &&
     tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fxq -- "$root/experiments/gpu/gpu_memory_probe.py"; then
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "GPU probe failed; retained evidence: $run_dir" >&2
  fi
}
trap cleanup EXIT
wait_marker() {
  # Poll empty control files with a deadline. kill -0 sends no signal; it checks
  # whether this PID exists and we have permission to signal it, not GPU progress.
  local marker="$1"
  local deadline=$((SECONDS + 60))
  until [[ -f "$run_dir/$marker" ]]; do
    kill -0 "$pid" 2>/dev/null || { cat "$run_dir/process.stderr" >&2; return 1; }
    (( SECONDS < deadline )) || { echo "Timed out: $marker" >&2; return 1; }
    sleep 0.1
  done
}
observe() {
  # Record the NVIDIA state plus two different memory views: VmRSS is resident
  # CPU RAM and VmHWM its high-water mark; nvidia-smi reports GPU allocations.
  local stage="$1"
  "$cuda_tool" --get-state --pid "$pid" > "$run_dir/$stage.cuda-state.txt"
  grep -E '^(VmRSS|VmHWM):' "/proc/$pid/status" > "$run_dir/$stage.host-memory.txt"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$run_dir/$stage.gpu-processes.csv"
  cat "$run_dir/$stage.cuda-state.txt"
}
transition() {
  # The manual controller owns all four NVIDIA actions in this probe. Do not
  # repeat these calls after a CRIU/DMTCP plugin already restored the same GPU.
  # Monotonic timestamps surround the bounded helper invocation, not image I/O.
  local action="$1"
  python3 -c 'import time; print(time.monotonic_ns())' > "$run_dir/$action.start-ns"
  timeout --kill-after=5 30 "$cuda_tool" --action "$action" --pid "$pid" > "$run_dir/$action.log" 2>&1
  python3 -c 'import time; print(time.monotonic_ns())' > "$run_dir/$action.end-ns"
}
"$cuda_tool" --help > "$run_dir/tool-help.txt"
nvidia-smi > "$run_dir/nvidia-smi.txt"
python3 "$root/experiments/gpu/gpu_memory_probe.py" --run-dir "$run_dir" > "$run_dir/process.jsonl" 2> "$run_dir/process.stderr" &
pid=$!
printf '%s\n' "$pid" > "$run_dir/pid.txt"
wait_marker ready
observe before
# Lock blocks further CUDA entry; checkpoint stages GPU state into host RAM and
# releases supported GPU resources. The original CPU process is still waiting.
transition lock
transition checkpoint
observe suspended
# B runs only after A reports checkpointed; it exits before A is restored. Its
# allocation and reduction provide functional evidence that GPU work can run.
grep -Fxq checkpointed "$run_dir/suspended.cuda-state.txt"
timeout --kill-after=5 60 python3 -c 'import torch; x=torch.ones(1024 * 1024, device="cuda"); value=x.sum().item(); assert value == 1024 * 1024; torch.cuda.synchronize(); print("job_B_gpu_sum", value)' > "$run_dir/job-b.txt" 2>&1
transition restore
# Restore rebuilds CUDA state; unlock permits subsequent CUDA calls to proceed.
transition unlock
observe restored
touch "$run_dir/release"
wait_marker done
wait "$pid"
pid=''
cat "$run_dir/process.jsonl"
cat "$run_dir/job-b.txt"
echo "GPU-only probe passed: $run_dir"
