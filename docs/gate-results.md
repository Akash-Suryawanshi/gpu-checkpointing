# Hard gate results

## Resumed-instance terminal check — 2026-09-14

The user resumed the existing instance. Its code-server workspace at `/home/gpu-checkpointing` became reachable and a read-only terminal command produced these live observations:

```text
Ubuntu jammy (22.04)
GPU: NVIDIA L4
Driver: 595.58.03
criu: not found on PATH
cuda-checkpoint: not found on PATH
CapEff: 00000000a80425fb
CapBnd: 00000000a80425fb
Seccomp: 2
```

The effective/bounding capability masks do not include CAP_SYS_PTRACE, CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE. This is a reason to probe actual CRIU behavior, not a completed CRIU failure test. No tools have been installed and no dump/restore has run yet. The driver differs from the September 12 observation, so historical driver information must not be reused as the current environment fingerprint.

Source transfer is prepared locally and the project is published privately at `https://github.com/Akash-Suryawanshi/gpu-checkpointing`. Browser command control has been intermittent; remote cloning/authentication remains unverified.

## Live dashboard check — 2026-09-14

Located the existing `gpu-checkpoint-poc` instance in the authenticated Jarvis dashboard: one L4, region IN2, status **Paused**. Jupyter and VS Code access were disabled. The displayed cost was ₹57.02 without an hourly unit; it was not established as the resume rate.

Automatic approval review rejected the resume control because it could start GPU billing without explicit approval. No resume or remote execution occurred. The CPU source bundle is prepared locally; resuming the existing instance is the next prerequisite.

## Implementation check — 2026-09-14

The CPU experiment is now implemented in `cpu_counter.py`, `checkpoint.sh` and `verify_cpu.py`. Three local acceptance checks passed, including a live process waiting at step 20 and continuing after release. These checks do not run CRIU.

`bash scripts/check-host.sh cpu` and the full command's gate both correctly stop on this Darwin ARM64 development machine. No Linux restore was attempted. The current Jarvis connection is not yet established; the host results below remain historical.

## Local development machine

Checked on 2026-09-12:

```text
cuda-checkpoint: MISSING
criu: MISSING
nvidia-smi: MISSING
os: Darwin
arch: arm64
```

This is expected. CUDA checkpointing is a Linux NVIDIA driver feature, so the Mac is only an editing machine.

## Jarvis execution host

Checked inside `/home/gpu-checkpointing` on the launched instance:

```bash
command -v cuda-checkpoint
command -v criu
nvidia-smi
```

Observed:

```text
cuda-checkpoint:MISSING
criu:MISSING
NVIDIA-SMI 580.126.20
Driver Version: 580.126.20
CUDA Version: 13.0
```

The GPU driver is visible, but the transparent checkpoint path is not currently available in the container because both user-space tools are absent. Do not silently replace this with `torch.save`; first decide whether the tools can be enabled, or record an application-level fallback as a separate experiment.
