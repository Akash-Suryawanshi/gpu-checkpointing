#!/usr/bin/env bash
# First milestone: one CPU process, one dump, one restore, with retained evidence.
set -euo pipefail
umask 077
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -ne 2 || "$1" != cpu ]]; then
  echo 'Usage: bash experiments/cpu/checkpoint.sh cpu /absolute/path/to/new-run-directory' >&2
  exit 2
fi
run_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$2")"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir" # Refuse to overwrite an earlier experiment or reuse release markers.
pid=''
cleanup() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "FAILED: retained evidence in $run_dir" >&2
    if [[ -z "$pid" && -f "$run_dir/restored.pid" ]]; then
      pid="$(cat "$run_dir/restored.pid")"
      [[ "$pid" =~ ^[0-9]+$ ]] || pid=''
    fi
    # Never act on a PID alone: it might now belong to another process.
    if [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] &&
       tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fxq -- "$root/experiments/cpu/cpu_counter.py" &&
       tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fxq -- "$run_dir"; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT
bash "$root/experiments/check-host.sh" cpu > "$run_dir/host.log" 2>&1 || {
  cat "$run_dir/host.log" >&2
  exit 1
}
mkdir "$run_dir/images"
python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b"".join(f"{i:04d}\n".encode() for i in range(1,26)))' "$run_dir/input.txt"
python3 "$root/experiments/cpu/cpu_counter.py" --mode baseline --run-dir "$run_dir" > "$run_dir/baseline.jsonl"

wait_for_file() {
  local target="$1"
  local deadline=$((SECONDS + 60))
  while [[ ! -f "$target" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $target; see process.jsonl and process.stderr" >&2
      return 1
    fi
    sleep 0.1
  done
}
now() { python3 -c 'import time; print(time.monotonic_ns())'; }
record_time() { printf '%s %s\n' "$1" "$(now)" >> "$run_dir/timing-ns.txt"; }

# Disconnected stdin and regular output files avoid checkpointing the driving terminal.
setsid python3 "$root/experiments/cpu/cpu_counter.py" --mode pause --run-dir "$run_dir" \
  < /dev/null > "$run_dir/process.jsonl" 2> "$run_dir/process.stderr" &
pid=$!
printf '%s\n' "$pid" > "$run_dir/original.pid"
wait_for_file "$run_dir/ready"
record_time dump_start
timeout --kill-after=5 60 criu dump --tree "$pid" --images-dir "$run_dir/images" \
  --shell-job --log-file dump.log -v4
record_time dump_returned
# Default dump ends the target. Reap it before CRIU attempts to reclaim the saved PID.
wait "$pid" 2>/dev/null || true
if [[ -e "/proc/$pid" ]]; then
  echo 'Original PID still exists; refusing to claim full process restoration.' >&2
  exit 1
fi
printf 'original_process_gone\n' > "$run_dir/lifecycle.txt"
pid=''
sync -f "$run_dir/images"
record_time local_sync_completed
record_time restore_start
timeout --kill-after=5 60 criu restore --images-dir "$run_dir/images" \
  --shell-job --restore-detached --pidfile "$run_dir/restored.pid" --log-file restore.log -v4
pid="$(cat "$run_dir/restored.pid")"
[[ "$pid" =~ ^[0-9]+$ ]] || { echo 'Invalid restored PID' >&2; exit 1; }
record_time restore_returned
printf 'restore_command_succeeded\n' >> "$run_dir/lifecycle.txt"
touch "$run_dir/release"
wait_for_file "$run_dir/done"
record_time final_step_observed
python3 "$root/experiments/cpu/verify_cpu.py" "$run_dir" > "$run_dir/comparison.json"
du -sb "$run_dir/images" > "$run_dir/image-directory-bytes.txt"
printf 'cpu_full_restore_verified\n' >> "$run_dir/lifecycle.txt"
cat "$run_dir/comparison.json"
echo "CPU restore evidence: $run_dir"
