#!/usr/bin/env bash
# Read-only capability probe. Run inside the same Linux environment as the experiment.
set -euo pipefail
mode="${1:-cpu}"
case "$mode" in cpu|gpu) ;; *) echo 'Usage: bash experiments/check-host.sh [cpu|gpu]' >&2; exit 2 ;; esac
uname -srm
if [[ "$(uname -s)" != Linux ]]; then
  echo 'BLOCKED: CRIU requires Linux. This machine can author the demo but cannot execute restore.' >&2
  exit 1
fi
id
python3 --version
for tool in criu timeout setsid sync; do
  command -v "$tool" || { echo "BLOCKED: missing $tool" >&2; exit 1; }
done
criu --version
timeout --kill-after=5 60 criu check
if [[ "$mode" == gpu ]]; then
  command -v nvidia-smi
  command -v cuda-checkpoint
  nvidia-smi
  cuda-checkpoint --help
  python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(16, device="cuda"); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "sum:", x.sum().item()); torch.cuda.synchronize()'
  echo 'GPU execution probe passed. CUDA plugin support and full restore still require an experiment.'
fi
