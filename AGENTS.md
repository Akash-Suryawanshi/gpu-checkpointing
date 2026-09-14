# Learning GPU snapshotting

Understand what GPU snapshotting saves, how it works, and where it can help. Use a bounded POC to make the mechanism and trade-offs observable; no production platform is currently being built.

## Optional local context

If `.local-notes/context.md` exists, read it before planning work. It contains machine-local context and is intentionally untracked; its absence on another checkout is normal. Keep personal context out of tracked documentation.

## Knowledge Updates

### 2026-09-14 - Native DMTCP can reacquire the GPU after writing a checkpoint
**Finding**: At the pinned DMTCP revision, `src/dmtcpworker.cpp:502` finalizes the image before the normal resume path, and `plugin/cuda/cuda-ckpt.cpp:333` calls GPU restore after both checkpoint resume and image restart. The release metric in `docs/implementation-plan.md:166` therefore needs to distinguish temporary staging-time release from availability after verified original-process exit; see the [pinned plugin](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/plugin/cuda/cuda-ckpt.cpp#L333-L378).
**Impact**: Do not treat the first disappearance from GPU monitoring as the completed handoff. Record original-process exit and subsequent GPU availability, with job B as functional evidence, before reporting the GPU as available to another job.

### 2026-09-14 - DMTCP final image names have a version-specific completion meaning
**Finding**: The preserved single-process controller uses a fresh run directory and `--no-gzip`, then waits for a final `*.dmtcp` image in `experiments/dmtcp/probe-dmtcp.py:96`. In the pinned DMTCP build, [processinfo.h:114](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/processinfo.h#L114) appends `.temp` while writing and [dmtcpworker.cpp:490–504](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/dmtcpworker.cpp#L490-L504) renames after the write barrier in the default non-forked path.
**Impact**: Final-name appearance is meaningful for this bounded uncompressed probe; do not replace it with arbitrary file-size sleeps or generalize it to compressed/forked/multi-process capture. Filesystem sync and survival of host loss are separate conditions.

### 2026-09-14 - DMTCP has a native NVIDIA CUDA route beyond historical CRAC
**Finding**: DMTCP v4.2.0 introduced a native CUDA plugin; maintenance SHA `b175bb5ccadd2f02d11cf052f586d2d9ac62ad53` contains later CUDA/PyTorch fixes. On the existing restricted host, CPU restore passed and a complete PyTorch GPU process image restored after original exit with matching tensor data, but anonymous shared-memory warnings remain; see `experiments/results.md:59`.
**Impact**: Do not describe CRIU permission changes or old CRAC as the only options. Preserve the distinction between measured restoration success and unqualified fine-tuning compatibility.

### 2026-09-14 - PyTorch CUDA runtime and compiler headers are separate installations
**Finding**: PyTorch uses CUDA 13.0, while `/usr/local/cuda` resolves to CUDA 12.6 development files. DMTCP's CUDA plugin required matching CUDA 13 runtime and CRT headers in isolated tools directories; the exact build and capacity observations are in `experiments/results.md:88`.
**Impact**: Inspect actual include paths and package dependencies, not just `torch.version.cuda` or a Makefile version banner. Read the cgroup RAM limit as well as host-wide `free` output when sizing GPU staging memory.

### 2026-09-14 - DMTCP warnings and logs require targeted evidence handling
**Finding**: DMTCP saves `/dev/zero (deleted)` contents as private memory, potentially losing sharing relationships; its separate `anon_inode` warning concerns missing kernel objects. Coordinator logs include the inherited environment, and this revision's blocking checkpoint command did not establish image finalization for our controller; see `experiments/results.md:67`.
**Impact**: Do not silently allowlist unsupported-resource warnings, publish whole coordinator logs, or end the original merely because the command returned. Launch probes with a minimal environment and verify finalized images and original-process exit separately.

### 2026-09-14 - Namespace sysctls do not establish container permissions
**Finding**: The live container exposes enabled user-namespace settings, but `unshare --user --map-root-user true` still fails with `Operation not permitted`; see `experiments/results.md:113`. The operation was tested without changing any security settings.
**Impact**: Check actual namespace creation before suggesting rootless CRIU or nested rootless containers. A globally enabled feature can still be denied within the instance.

### 2026-09-14 - Distinguish CRIU startup probes from workload requirements
**Finding**: Both the packaged CRIU gate and direct CPU dump fail in `kerndat_has_nftables_concat` before process capture; see `experiments/results.md:103`. The L4 GPU-only probe nevertheless passes in the same container, preserving a 64 MiB tensor while the original CPU process stays alive.
**Impact**: Do not say the CPU counter needs networking or equate GPU suspension with a saved process image. Keep the container-permission failure and successful NVIDIA-only probe as separate outcomes.

### 2026-09-14 - Recheck the driver after instance resume
**Finding**: The resumed instance reports driver 595.58.03, whereas the original observation reported 580.126.20; see `experiments/results.md:133`. The saved workspace remaining available does not establish that the surrounding driver environment stayed identical.
**Impact**: Record the live GPU, driver and tool versions for each experiment after a resume. Do not reuse historical compatibility results without checking the current execution environment.

### 2026-09-12 - Separate the development machine from the execution host
**Finding**: The initial workspace is on macOS ARM64. NVIDIA's [CUDA checkpoint API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html) is Linux-only; ARM CPU support in cuda-checkpoint does not imply macOS support.
**Impact**: Author notes and code locally, but validate CUDA checkpoint/restore on a compatible NVIDIA Linux host. Do not report a local simulation as GPU checkpoint validation.

### 2026-09-12 - PyTorch caching is different from CUDA managed memory
**Finding**: The supplied September 4 brief incorrectly associates ordinary PyTorch lazy/cached allocations with unsupported UVM memory. [PyTorch's CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management) describes a caching allocator and separately discusses opt-in cudaMallocManaged allocations; [NVIDIA's limitations](https://github.com/NVIDIA/cuda-checkpoint#functionality) apply to UVM and specified IPC APIs.
**Impact**: Do not exclude ordinary PyTorch workloads solely because UVM checkpointing is unsupported. Record the actual allocator configuration and CUDA features when evaluating a POC.

### 2026-09-12 - Choose one owner for CUDA restore
**Finding**: Earlier drafts mixed CRIU integration with unconditional manual CUDA restore/unlock; the maintained distinction is in `docs/02-gpu-checkpointing.md:123`. The upstream [CUDA plugin restore hook](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/cuda/cuda_plugin.c#L493-L505) restores and unlocks CUDA itself, even if CUDA was already checkpointed before the dump.
**Impact**: Select and verify either manual CUDA transitions or plugin-managed transitions for the installed CRIU version. Unconditional manual restore/unlock after plugin-managed restore can encounter an already-running process and fail.

### 2026-09-13 - QLoRA optimizer choice can introduce unsupported managed memory
**Finding**: bitsandbytes [optimizer.py:339–355](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py#L339-L355) uses paged buffers for sufficiently large optimizer tensors when paging is enabled; [pythonInterface.cpp:501–504](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/pythonInterface.cpp#L501-L504) allocates those buffers with `cudaMallocManaged`. NVIDIA cuda-checkpoint excludes UVM, while four-bit model weights alone do not establish that a workload uses UVM.
**Impact**: Start fine-tuning experiments with a non-paged optimizer. Probe checkpoint compatibility after optimizer state has been initialized, and do not generalize a QLoRA failure to all quantized fine-tuning or wait for visible CPU paging before checking this constraint.

### 2026-09-14 - CUDA plugin coordination is not automatic incremental GPU capture
**Finding**: The inspected upstream [cuda_plugin.c:359–373](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/cuda/cuda_plugin.c#L359-L373) temporarily resumes NVIDIA's helper thread to stage device state while coordinating frozen application threads. Its [initialization at lines 525–529](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/cuda/cuda_plugin.c#L525-L529) disables CUDA checkpoint handling during CRIU pre-dump.
**Impact**: Explain the helper-thread handoff when teaching CPU/GPU consistency. Do not present CRIU pre-dump or a rotating latest directory as an existing incremental GPU checkpoint optimization; verify installed-version behavior before proposing it.
