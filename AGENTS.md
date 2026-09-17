# Learning GPU snapshotting

Experiments for learning GPU snapshots and their trade-offs.

## Learning audience

Write clear technical prose for a CS student without assumed OS knowledge.
Define unfamiliar terms at first use; shortening must not turn explanations into jargon.

## Documentation style

- Keep files short and paragraphs to 1–3 short sentences. Cut repetition and filler.
- Pair every conceptual or process explanation with a compact diagram showing
  sequence, state, data movement, or relationships. Use SVG, Mermaid, or plain
  text; label arrows and add a brief caption or text equivalent.
- Link to existing explanations and diagrams instead of repeating them.
  Chapters own mechanisms, the runbook owns commands, results owns evidence,
  and plans own requirements.
- Keep necessary definitions, runnable commands, evidence, and limitations.
  Comments explain non-obvious operations; follow the [teaching plan](docs/readability-plan.md).

## Optional local context

Read `.local-notes/context.md` if present before planning. It is intentionally
untracked; never copy personal context into public docs.

## Git workflow

Commit each milestone on a feature branch, never `main`. The user reviews and
merges; do not merge on their behalf.

## Knowledge Updates

### 2026-09-17 - CRIU buffer flushing is not snapshot durability
**Finding**: Pinned `criu/bfd.c:114` checks earlier buffer errors; `bflush():235` calls `write_all()`, not `fsync()`. `pipeline.py:115` separately invokes `sync -f`.
**Impact**: Persist payload, directory entries, and completion in order; see [publication contract](docs/independent-lifecycle-details.md). Local persistence does not establish survival of volume deletion.

### 2026-09-17 - Independent capture workers need explicit process ownership
**Finding**: `pipeline.py:65` creates the trainer; `:99` calls `session.reap():75`, which assumes child ownership. Subreapers adopt orphaned descendants, not unrelated trainers.
**Impact**: Separate exit observation from parent-owned reaping. Successful restore must transfer supervision without depending on the old `Trial`.

### 2026-09-17 - PR body updates can bypass an obsolete CLI query
**Finding**: Installed `gh pr edit` fails on retired GraphQL `projectCards`. REST `gh api .../pulls/2 --method PATCH --input <json-file>` works.
**Impact**: Use the personal CLI profile from local context; changing accounts does not fix this query error.

### 2026-09-17 - Zombie cleanup needs the raw command-line bytes
**Finding**: Splitting empty `/proc/PID/cmdline` gives truthy `[b'']`. `experiments/criu/session.py:112` checks raw bytes and exited state; a live child can briefly have empty arguments.
**Impact**: Keep zombie and live-child ownership regression checks. Require matching job identity before signaling a running process.

### 2026-09-14 - CRIU restores a different monotonic clock domain
**Finding**: Pinned `criu/timens.c` restores clock offsets; one trial recorded `-25 705761319`. Uncorrected trainer/controller comparison produced negative latency.
**Impact**: Preserve raw times and measured offsets; convert clock domains and reject negative latency. Same-process durations cancel the offset.

### 2026-09-14 - Repeated CRIU restores need fresh PID filenames
**Finding**: CRIU 4.2.1 opens `--pidfile` with `O_EXCL` (`criu/log.c:414`). Reuse failed after partial reconstruction.
**Impact**: Use fresh image/PID/handshake files per generation; reap the original before reusing its numeric PID.

### 2026-09-14 - CRIU and preliminary LoRA restoration pass on the EC2 host
**Finding**: EC2 A10G/570.172.08 restored CPU, GPU-tensor, and initialized LoRA processes with pinned CRIU `9539417f`. GPU access required approved host execution.
**Impact**: Use the [EC2 evidence](experiments/results.md#ec2-criu-validation--2026-09-14), not historical container blockers. Retain sharing/syscall-warning qualifications; same-host success is not spot recovery.

### 2026-09-14 - Native DMTCP can reacquire the GPU after writing a checkpoint
**Finding**: Pinned DMTCP finalizes images before resume; its [CUDA hook](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/plugin/cuda/cuda-ckpt.cpp#L333-L378) restores GPU state on both original resume and image restart.
**Impact**: Measure availability after verified original exit, not the first disappearance from monitoring. Keep job B as functional evidence.

### 2026-09-14 - DMTCP final image names have a version-specific completion meaning
**Finding**: Pinned `dmtcpworker.cpp:490–504` renames `.temp` after the write barrier for the bounded, uncompressed, non-forked probe.
**Impact**: Use fresh directories and final-name evidence for that route only. Command return, stable size, compressed/forked capture, and durable storage are separate questions.

### 2026-09-14 - DMTCP has a native NVIDIA CUDA route beyond historical CRAC
**Finding**: DMTCP v4.2.0 added native CUDA; pinned `b175bb5c` includes later fixes. The earlier container restored GPU tensor bytes, with shared-memory warnings.
**Impact**: Do not describe CRIU or historical CRAC as the only options, or infer full fine-tuning compatibility from that probe.

### 2026-09-14 - PyTorch CUDA runtime and compiler headers are separate installations
**Finding**: The earlier host used PyTorch CUDA 13.0 with `/usr/local/cuda` headers at 12.6. DMTCP needed isolated matching runtime/CRT headers.
**Impact**: Inspect actual include paths. Check cgroup RAM limits as well as host memory before GPU staging; build details remain in [results](experiments/results.md).

### 2026-09-14 - DMTCP warnings and logs require targeted evidence handling
**Finding**: DMTCP captures `/dev/zero (deleted)` bytes privately; `anon_inode` warnings concern missing kernel objects. Coordinator logs contain inherited environment values.
**Impact**: Preserve sharing/resource warnings, publish targeted evidence, use a minimal environment, and verify image finalization independently of blocking-command return.

### 2026-09-14 - Namespace sysctls do not establish container permissions
**Finding**: The earlier container enabled namespace sysctls but rejected actual `unshare --user --map-root-user true`.
**Impact**: Probe namespace creation before proposing rootless CRIU or nested containers; do not change security policy to hide the blocker.

### 2026-09-14 - Distinguish CRIU startup probes from workload requirements
**Finding**: Packaged CRIU failed in `kerndat_has_nftables_concat` before CPU capture. NVIDIA-only suspension still preserved the L4 tensor in the live process.
**Impact**: Do not infer a networking requirement for the CPU counter or equate GPU suspension with a complete saved process.

### 2026-09-14 - Recheck the driver after instance resume
**Finding**: The earlier resumed instance reported driver 595.58.03 instead of 580.126.20.
**Impact**: Record live GPU/driver/tool versions after resume; a persistent workspace does not establish an unchanged execution environment.

### 2026-09-12 - Separate the development machine from the execution host
**Finding**: Initial development used macOS ARM64; NVIDIA CUDA checkpointing is Linux-only.
**Impact**: Validate on a compatible NVIDIA Linux host. ARM support does not imply macOS support; simulation is not GPU validation.

### 2026-09-12 - PyTorch caching is different from CUDA managed memory
**Finding**: Ordinary PyTorch caching is distinct from opt-in `cudaMallocManaged`; see [CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management).
**Impact**: Inspect actual allocation features. Do not exclude normal PyTorch solely because NVIDIA checkpointing excludes UVM.

### 2026-09-12 - Choose one owner for CUDA restore
**Finding**: CRIU’s CUDA plugin restores/unlocks GPU state itself, including when CUDA was staged before dump.
**Impact**: Select one owner for the installed route. Unconditional manual restore/unlock afterward can fail on an already-running process; see the [GPU chapter](docs/02-gpu-checkpointing.md).

### 2026-09-13 - QLoRA optimizer choice can introduce unsupported managed memory
**Finding**: Paged bitsandbytes buffers can use `cudaMallocManaged`; four-bit weights alone do not imply this. Source links remain in the [research note](research/single-gpu-finetuning.md).
**Impact**: Start with a non-paged optimizer and probe after Adam state exists. Do not generalize a paging failure to all QLoRA.

### 2026-09-14 - CUDA plugin coordination is not automatic incremental GPU capture
**Finding**: Pinned `cuda_plugin.c:359–373` resumes NVIDIA’s helper while coordinating application threads. Lines 525–529 disable CUDA handling during CRIU pre-dump.
**Impact**: Explain this handoff. Neither CPU pre-dump nor a rotating latest directory establishes incremental GPU capture.

### 2026-09-17 - Extracted CRIU dependencies include dangling linker symlinks
**Finding**: The local extracted dependency directory contains an unversioned `libmd.so` link without its target. `experiments/finetuning/control.py:213` fingerprints usable runtime library files and omits dangling development links.
**Impact**: Do not hash every `*.so*` glob result blindly; that rejects a working CRIU runtime before capture.

### 2026-09-17 - CRIU statistics must leave the immutable image directory
**Finding**: Pinned `criu/stats.c:205` writes statistics through `AT_FDCWD`; `criu/crtools.c:219` defaults the work directory to the images directory. An absolute log path alone still leaves `stats-restore` inside the hashed payload.
**Impact**: `experiments/criu/session.py:213` passes `--work-dir` pointing to the mutable attempt directory for both dump and restore.

### 2026-09-17 - Restored trainers cannot publish host-clock PID start ticks
**Finding**: The first independent two-generation trial read start ticks `78305919` inside the restored trainer and `78308751` in its host worker for the same PID. `experiments/finetuning/restore.py:43` now refreshes registration from the host worker; `control.py:144` adopts that record instead of replacing it with the trainer's time-namespace view.
**Impact**: Process identity comparisons and acknowledgements must use one observer clock domain. Failure cleanup must also read the refreshed `job.json` when restore exits after release but before writing its final result.

### 2026-09-17 - Pause expiry must be serialized with dump arming
**Finding**: A deadline test showed `experiments/finetuning/control.py:144` could resume an acknowledged trainer while its capture worker still held the operation lock. Expiry now rechecks the unarmed phase under that lock; workers also wait with their deadline when initial registration briefly owns it.
**Impact**: An expired request alone is not permission to resume once a worker may be preparing a dump. Worker death releases the lock, but a persisted dumping phase still forbids automatic continuation.

### 2026-09-17 - Check the installed loader before claiming parallel-loading gains
**Finding**: Installed Transformers 5.0.0 `core_model_loading.py:1116–1121` uses a thread pool by default; `HF_DEACTIVATE_ASYNC_LOAD` disables it. Its versioned environment-variable documentation instead describes `HF_ENABLE_PARALLEL_LOADING`.
**Impact**: Benchmark the real default and verify the execution host's implementation; a serial control is not evidence of improvement over the default.

### 2026-09-17 - Inference and training scripts share short module names
**Finding**: Both experiment directories contain `run.py` and `report.py`; the combined runbook import path can select the wrong module. `tests/test_finetuning_metrics.py:14` and `tests/test_inference_report.py:12` load their target scripts by absolute file location.
**Impact**: Keep combined-suite imports explicit where script names overlap; changing path order alone can silently test the other controller.

### 2026-09-17 - Cancel the whole inference helper group before reaping its leader
**Finding**: `experiments/inference/lifecycle.py:22` uses `waitid(..., WNOWAIT)` to keep the helper PID reserved while stopping its process group. `experiments/criu/session.py` keeps GNU timeout in that group for inference; its default group creation would let privileged descendants escape group cleanup.
**Impact**: A controller interruption must stop capture/restore commands before cleaning up model workers. Preserve the separate restored-worker identity checks, and report an unresolved helper or partial restore as failed cleanup.

### 2026-09-17 - Killing a direct command does not prove its exit was collected
**Finding**: An injected SIGTERM during `experiments/criu/session.py:181` reached Python's `KeyboardInterrupt` cleanup, but the child still had a `/proc` entry when the call returned. `experiments/inference/lifecycle.py:22` explicitly stops and reaps owned helper groups.
**Impact**: Use that owner for inference job B as well as capture/restore; GPU reuse requires completed cleanup, not merely sending a kill signal.

### 2026-09-17 - The isolated CRIU build still needs host build prerequisites
**Finding**: Pinned `criu/Makefile.packages:30` links UUID and Netlink. `experiments/criu/build.sh` omits UUID development files, and its private `usr/lib` search misses Ubuntu's extracted `lib/libnl-3` layout; the old host supplied both globally.
**Impact**: The clean container explicitly installs `uuid-dev`/`libnl-3-dev` for building and their runtime libraries. Keep compiler logs; the generic package-check message lists unrelated test dependencies too.
