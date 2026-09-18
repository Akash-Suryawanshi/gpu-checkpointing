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

### 2026-09-18 - Pinned buffers need explicit alignment for direct I/O
**Finding**: Early `direct` diagnostics failed the alignment guard; the packed 8B file ends with a 2048-byte partial page. `experiments/inference/stream_weights.py:20` explicitly aligns buffer views, and the loader requests a full aligned buffer at EOF while hashing only the returned model bytes.
**Impact**: Pinned memory alone does not establish `O_DIRECT` alignment. Distinguish aligned request sizes from a permitted short EOF return; preserve failed diagnostics instead of silently using buffered I/O.

### 2026-09-18 - Disk validation precedes the CRIU restore event
**Finding**: `experiments/finetuning/restore.py:27` validates before emitting `restore_requested`. In the third 8B timing, controller observations place 137.60 seconds before that event, 139.45 inside the CRIU command, and 0.57 after return; the 4,351,144 restored pages are 16.60 GiB, not 17.8 GiB.
**Impact**: Join host-clock controller and helper events before attributing latency. Do not interpret the difference between total latency and CRIU duration as post-restore delay.

### 2026-09-18 - The 8B snapshot already has an aggregated page image
**Finding**: The retained 8B manifests put over 99.9% of payload bytes in `images/pages-8.img`. CRIU logs record repeated 2,147,479,552-byte `preadv` returns taking about 16.36 seconds, approximately 125.2 MiB/s.
**Impact**: Prioritize storage bandwidth and repeated full-file passes over small-file aggregation. An NVMe device name alone does not identify local storage: the original data mount is EBS; the host also has separately mounted instance storage.

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

### 2026-09-17 - Container GPU identity checks need a consistent process view
**Finding**: `experiments/inference/park.py:43` compares NVIDIA-reported process IDs with `/proc` identities. The validated container launch in `experiments/inference/README.md` uses `--pid=host` and `--cgroupns=host`, so identity checks and ancestor-memory admission use the host views.
**Impact**: Keep that tested profile when reproducing these results; changing namespace isolation requires fresh probes and diagnostics, not an assumption that container IDs will match GPU monitoring.

### 2026-09-18 - Apply a campaign's settings to every route in it
**Finding**: The validation-flags arm passed `--validation-order`/`--validation-workers` to its restore route only. Those settings enter the comparison key, so the campaign held two keys and `experiments/inference/report.py:66` refused to aggregate it. The individual runs remain valid and its fresh key matched the control campaign exactly.
**Impact**: Vary one setting across campaigns, never across routes inside one. When a campaign does hold several keys, curate one record per key group and state the cross-campaign comparison rather than weakening the report's refusal.

### 2026-09-18 - Snapshot images are bound to their checkout path
**Finding**: `experiments/finetuning/control.py:241` keys dependency hashes by absolute path, so the same commit in a second git worktree produces a different dependency record. An image captured in one worktree fails validation from another even with identical file contents, and any source edit invalidates every unreleased image.
**Impact**: Run a campaign to completion from one fixed checkout, and capture and restore an image from that same path. Treat images as unusable after a source change; discard only their `snapshot/images` payload and keep manifests, logs, and results.

### 2026-09-18 - A background server in a script ignores SIGINT
**Finding**: The campaign script could not stop its endpoint server: `/proc/PID/status` showed `SigIgn` covering SIGINT. A non-interactive shell without job control sets background commands to ignore SIGINT and SIGQUIT, and Python keeps an inherited SIG_IGN. `wait` then blocked for fifteen minutes, making the script's own SIGKILL escalation unreachable.
**Impact**: `experiments/inference/api.py` now installs handlers for SIGINT and SIGTERM itself. Send SIGTERM from scripts, bound every wait, and escalate; check `SigIgn` before concluding a process is hung.

### 2026-09-18 - A busy worker delays the next request as surely as a blocking controller
**Finding**: After moving the controller's audit wait to unload, an endpoint follow-up still measured 14.4 s. `experiments/inference/worker.py` fingerprinted every weight straight after response one, reading no request for the duration. The command-line controller hid this by waiting for the audit before timing its second request; an HTTP client cannot. The audit now runs after the request loop ends, before the completion marker.
**Impact**: Removing a wait from the caller does not help when the callee is busy. Check the server phase spans on both sides, and keep whole-model work outside the window in which a request can arrive.

### 2026-09-18 - Cache eviction can lose a race, and said nothing about which file
**Finding**: One loader timing block failed in `experiments/inference/cold_cache.py` with "Selected files are still cached", naming no file, so the cause could not be identified; the record is never written when the check raises. Two neighbouring blocks with identical settings passed. Eviction now retries a bounded number of times and names each file with its resident and total page counts.
**Impact**: Retrying makes the cold-cache guarantee stricter, not weaker, because the returned records must still show zero resident pages. Keep the failed slot rather than rerunning it.

### 2026-09-18 - Compiled models cannot be captured, and smaller models snapshot worse
**Finding**: `--compile-mode default` makes CRIU fail in `criu/proc_parse.c:118` with `handle_device_vma plugin failed`, before any image is written: generated kernels are device mappings the pinned plugin does not handle. The compiled model's greedy output also diverged from the uncompiled reference at token twelve. On Qwen2.5-0.5B a snapshot took 17.95 s against a fresh 11.12 s, a worse ratio than the 8B model's.
**Impact**: Do not expect a smaller model to favour restoration; fixed process state is a larger share of a small image. Test a capture-compatibility hypothesis on the small model first, and give any compiled campaign its own prepared reference.

### 2026-09-18 - Keep benchmark bookkeeping and audits out of the request path
**Finding**: The first endpoint trial measured 145.7 s for a fresh activation and 14.1 s for a warm follow-up. `Runtime.ensure_ready()` fingerprinted all 15.26 GiB of model files inside the first request, which also warmed the cache the trial had just evicted, and `generate()` waited for the worker's full weight audit before the second. Server phase records isolated both: `received->ready` 145.7 s, then `ready->first_token` 14.1 s with the model already resident.
**Impact**: Build the comparison key when the server starts, and verify the audit at unload after the response pair, as the plan requires. Start a server before evicting caches, and read the server phase spans before trusting a client number.

### 2026-09-18 - A listener on the port is not a ready endpoint
**Finding**: The first endpoint campaign chose port 8090, already held by an unrelated local service that answered `/healthz` with plain `ok`. `api.py` died on bind, the campaign script's `curl -sf` readiness check passed, and the trial failed inside `json.loads`. `/healthz` now returns a per-process `server_id`, model, and route; `bench_http.ready()` requires them.
**Impact**: Check port ownership before a campaign, and make readiness probes identify the intended server rather than confirm that something is listening. Report a non-JSON reply with its body, not a decoder error.

### 2026-09-18 - Published inference images are currently single-use
**Finding**: `experiments/finetuning/control.py:330` requires the exact published phase; `restore.py:48` changes registration, while validation also rejects old inspection markers and changed captured logs. `experiments/criu/session.py:204` restores the saved PID, and `worker.py:87` retains its captured run path.
**Impact**: Repeated activation uses `control/activation.json` per attempt plus baseline log copies under `snapshot/external/`, never a phase reset; `control.validate()` resets the consumed marker and registration under the lock. Training keeps the single-use contract; see the [activation contract](docs/inference-activation-contract.md).
