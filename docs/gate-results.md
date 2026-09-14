# Hard gate results

## DMTCP native CUDA investigation — 2026-09-14

The [fine-tuning feasibility audit](r-2026-09-14-finetuning-feasibility.html) supersedes the earlier assumption that the only DMTCP CUDA route is experimental CRAC. DMTCP v4.2.0 introduced a native NVIDIA plugin; the live experiment used the later 4.2 maintenance commit `b175bb5ccadd2f02d11cf052f586d2d9ac62ad53` (September 7, 2026), with no local source patch and `--disable-prgname-prefix`. The tool checkout is `/home/gpu-checkpointing-tools/dmtcp-4.2.0`; its directory name is historical, so use the Git SHA to identify the build.

- **CPU process restore passed.** `runs/dmtcp-cpu-02/`: original real PID 11118 exited; real PID 11128 restored from a 38,118,736-byte image. The token, file position and steps 1–25 matched. Image-finalization observation: 0.510 seconds, before filesystem sync.
- **GPU lifecycle and data checks passed with unresolved compatibility warnings.** `runs/dmtcp-gpu-01/`: a 3,535,032,587-byte image was written, original real PID 11208 exited, job B ran successfully, and restored real PID 11262 returned the same complete 64 MiB tensor hash and passed the next GPU operation's endpoint checks. DMTCP preserved virtual PID 40000 in both application records. Image-finalization observation: 1.313 seconds, before sync. The image occupied 3,535,036,416 bytes of filesystem blocks.
- **The stricter acceptance controller rejected the GPU run after its numerical checks.** Four `/dev/zero (deleted)` anonymous shared-memory warnings occurred. Pinned `src/writeckpt.cpp:479–494` saves such contents but restores them as private anonymous memory; original sharing/aliasing semantics may be lost. Four writable 2 MiB mappings with different backing identities were visible before capture; their ownership and driver-side dependencies are not established. Do not report either an unqualified compatibility pass or a failed tensor restoration. A separate session-leader restoration warning also occurred.
- **No unsupported `anon_inode:` mapping appeared in the probe's saved pre-checkpoint maps.** This is not a guarantee for a later training workload. Pinned `src/plugin/ipc/file/fileconnlist.cpp:451–463` treats kernel-backed anonymous mappings separately from `/dev/zero` memory.
- **No security settings or driver were changed.** Both GPU jobs exited, and the final GPU compute-process query was empty. No fine-tuning process, optimizer state, replacement host or real spot interruption was tested.

Build findings: the active PyTorch runtime is CUDA 13.0, but system `nvcc` and development headers are 12.6. Those headers failed to declare the plugin's checkpoint API types; CUDA 13 runtime headers alone then lacked `crt/host_defines.h`. Matching `nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-nvcc==13.0.88` and `nvidia-cuda-crt==13.0.88` were installed in isolated `/home/gpu-checkpointing-tools/cuda13-build` and `cuda13-crt` directories. The plugin compiled using their include paths and linked the installed `libcuda.so.1`. No package was replaced in the existing Python environment. The Makefile version banner still reports the 12.6 nvcc on PATH; compilation include paths are authoritative for the headers used.

Live capacity: L4 23,034 MiB total; container `memory.max` 133,143,986,176 bytes (124 GiB), despite `free` reporting much larger host RAM; `/home` available 106,499,379,200 bytes at inspection. PyTorch 2.11.0+cu130, native allocator with no allocator environment overrides, Transformers 5.7.0 installed, PEFT absent.

The bounded controller is retained remotely at `runs/probe-dmtcp.py`. Focused evidence is copied locally under `runs/remote-2026-09-14/dmtcp-cpu-02/` and `dmtcp-gpu-01/`. Raw coordinator logs contain inherited environment data and must not be published or printed wholesale; subsequent probe launches used a minimal environment. The initial CPU controller attempt (`dmtcp-cpu-01`) checked for an image too early after `--bcheckpoint`; it was corrected to wait for image finalization before termination. This was a controller failure, not evidence against CPU restore.

Sources: [DMTCP release](https://github.com/dmtcp/dmtcp/releases/tag/v4.2.0), [pinned CUDA plugin](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/plugin/cuda/cuda-ckpt.cpp), [shared-memory serialization](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/writeckpt.cpp#L479-L494), [unsupported mapping diagnostics](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/plugin/ipc/file/fileconnlist.cpp#L451-L463).

## SSH experiments — 2026-09-14

Connected using the user-provided SSH endpoint and existing key, retaining host-key checks. Transferred the committed repository using a Git bundle into `/home/gpu-checkpointing-poc`; no GitHub credential was copied to the instance. The checkout initially matched commit `495f23d`.

Environment: Ubuntu 22.04.5, Linux 6.8.0-110-generic x86_64, L4, NVIDIA driver 595.58.03, Python 3.10.20, PyTorch 2.11.0+cu130. All three CPU workload/evidence tests passed in 0.112 seconds.

### CRIU CPU process restore: blocked

Installed Ubuntu's official CRIU 3.16.1 package. Both `criu check` through `checkpoint.sh` and a direct dump of the disposable counter returned exit 1. The latter log ends with:

```text
Unable create a network namespace: Operation not permitted
kerndat_has_nftables_concat failed when initializing kerndat.
Could not initialize kernel features detection.
```

This failure is a startup probe, before capture of the counter. CRIU tests an nftables feature in a temporary network namespace so its test firewall table cannot collide with another CRIU invocation. Our counter does not itself require a new network namespace. [Version-pinned source](https://github.com/checkpoint-restore/criu/blob/v3.16.1/criu/kerndat.c).

The container's capability masks remain `a80425fb`, without CAP_CHECKPOINT_RESTORE, CAP_SYS_PTRACE or CAP_SYS_ADMIN. Current CRIU's documented unprivileged mode still requires checkpoint-related permissions; skipping an old startup probe alone would not prove restore support. `Seccomp: 2` establishes filtering, not which exact rule caused this denial. [CRIU v4.2.1 permissions](https://github.com/checkpoint-restore/criu/blob/v4.2.1/Documentation/criu.txt#L840-L871).

### Rootless namespace probe — 2026-09-14

An ordinary `unshare --user --map-root-user true` attempt returned `Operation not permitted`. The visible `kernel.unprivileged_userns_clone` value was 1 and `user.max_user_namespaces` was 4642501, so those settings alone do not establish that namespace creation is permitted inside this container. No security settings were changed. DMTCP had not been tested at this point; its subsequent native CUDA investigation and qualified results are recorded above.

### GPU-only suspension and restoration: passed

Downloaded NVIDIA cuda-checkpoint commit `00d5cce84c628088d6caa203fc4af40c1538b6f7`; its x86_64 utility reports 595.58.03. The original CPU process stayed alive throughout. There was no CRIU image and no persistent GPU snapshot file.

| Observation | Before | Suspended | Restored |
| --- | --- | --- | --- |
| CUDA state | running | checkpointed | running |
| Process listed by nvidia-smi | PID 1029, 252 MiB | absent | PID 1029, 252 MiB |
| Host RSS | 609740 KiB | 816396 KiB | 609832 KiB |

Tensor payload: 67,108,864 bytes (64 MiB). SHA-256 before and after: `bcfcc724743f7bf094ad3ecaf64d1d5fcc08e80c5801a5c00d368c99bcf8f709`. Subsequent tensor addition returned correct endpoint values. Job B allocated a CUDA tensor and returned the expected sum 1048576 while A was suspended, then exited before A was restored. Both probe processes exited; the final nvidia-smi process query was empty.

One-run timings, including command launch and timestamp overhead: lock 0.0283 s, checkpoint 0.2218 s, restore 0.2069 s, unlock 0.0296 s. These are not training-checkpoint benchmarks. RSS and GPU process memory include framework/driver resources, not only the 64 MiB tensor. Suspended host RSS increased by approximately 202 MiB; the driver and allocator account for more state than the visible tensor alone.

Raw evidence is retained on the instance under `runs/cpu-01`, `runs/cpu-direct-01` and `runs/gpu-only-01`, and copied locally under ignored `runs/remote-2026-09-14/`. No host security settings were changed. An environment with CRIU permissions remains necessary for that route; subsequent DMTCP experiments above establish a separate path with unresolved shared-memory qualifications.

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
