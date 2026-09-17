# Experiment results

This page records what was actually observed, separating the EC2 implementation from earlier container probes:

- **EC2 A10G: isolated four-update LoRA continuation passed with CRIU and application restart.** Both dropout configurations and repeated CRIU restoration matched references. Three timing pairs measured capture/sync and restore/update medians of 32.20 s / 4.47 s for CRIU, versus 1.30 s / 7.95 s for application checkpoints. Compatibility remains qualified by sharing ownership and retained warnings. See the [full acceptance and timing results](#four-update-lora-acceptance-and-timing--2026-09-14).
- **Earlier L4 container: CRIU process capture was blocked.** CRIU 3.16.1 failed during startup feature detection before it captured the CPU counter; this was not an EC2 host limitation.
- **Earlier L4 container: NVIDIA GPU-only checkpoint/restore passed.** A 64 MiB tensor survived and another job used the GPU while it was suspended, but the original CPU process stayed alive and no durable process image was created.
- **Earlier L4 container: DMTCP restored a complete PyTorch GPU process after the original exited.** Lifecycle, tensor, and subsequent GPU-compute checks passed. Four anonymous shared-memory warnings leave sharing semantics unresolved, so strict acceptance failed and DMTCP fine-tuning compatibility remains unproven. DMTCP was not benchmarked on EC2.

No experiment changed container security settings or the NVIDIA driver. Curated DMTCP measurements are also available as [JSON](evidence/2026-09-14/dmtcp-summary.json). Raw process images and logs remain under ignored `runs/`; coordinator logs can contain inherited environment values and must not be published wholesale.

## Run the maintained experiments

Use a Linux NVIDIA host for GPU work. Run the host check and experiment under the same user and in the same container or host. Every launcher requires a new run directory and refuses to overwrite an existing one.

```bash
# CPU process capture through CRIU
bash experiments/check-host.sh cpu
bash experiments/cpu/checkpoint.sh cpu "$PWD/runs/cpu-01"

# NVIDIA-only GPU suspension; the CPU process remains alive
bash experiments/gpu/probe-gpu.sh \
  /home/gpu-checkpointing-tools/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint \
  "$PWD/runs/gpu-only-02"

# Local CPU evidence checks
PYTHONPATH=experiments/cpu python3 -m unittest discover -s tests -v
```

The CPU experiment needs Python's standard library, CRIU, and the usual GNU/Linux tools (`timeout`, `sync`, and `setsid`). The NVIDIA probe needs PyTorch with CUDA and a compatible `cuda-checkpoint` executable. DMTCP source is retained as `experiments/dmtcp/probe-dmtcp.py`; its CLI is documented by `--help` because tool and run locations are explicit inputs.

```bash
mkdir -p runs
python3 experiments/dmtcp/probe-dmtcp.py cpu \
  "$PWD/runs/dmtcp-cpu-03" \
  --dmtcp-root /home/gpu-checkpointing-tools/dmtcp-4.2.0
python3 experiments/dmtcp/probe-dmtcp.py gpu \
  "$PWD/runs/dmtcp-gpu-02" \
  --dmtcp-root /home/gpu-checkpointing-tools/dmtcp-4.2.0
```

The DMTCP parent `runs/` directory must already exist, and each run directory must be fresh. The controller uses a restrictive file-creation mask. Its GPU mode rejects unsupported-resource warnings even when lifecycle and numerical checks pass; no new GPU run was performed as part of the repository cleanup.

Generated images contain process memory. Keep `runs/` private and on storage appropriate to the failure being tested. `sync -f` flushes to the current filesystem; it does not prove that local storage survives instance deletion.

## Read the CPU evidence

The CPU experiment first runs an uninterrupted reference from records 1 through 25. A second process reads through record 20, then waits while retaining its random token and open-file position in memory. After process restoration and release, it must continue at record 21 and finish at record 25. The expected saved boundary is step 20 at byte offset 100; the final position is step 25 at byte offset 125.

`experiments/cpu/cpu_counter.py` never loads an application checkpoint or reads the evidence logs. The input file must remain present and unchanged because its restored open descriptor still refers to that file. The launcher verifies that the original process disappeared before restore. CRIU can restore the same numeric PID, so a different post-restore PID is not required proof.

Read `comparison.json`, `lifecycle.txt`, and `process.jsonl` together. Retain the dump and restore logs, `original.pid`, and `restored.pid` with them; matching application JSON by itself does not prove that CRIU performed the lifecycle. If CRIT is installed, inspect the process tree with:

```bash
crit decode -i runs/cpu-01/images/pstree.img --pretty
```

The image directory can also contain `core-*.img`, `mm-*.img`, `pagemap-*.img`, and raw `pages-*.img`; raw page files are not structured CRIT records. `timing-ns.txt` contains monotonic nanosecond timestamps, so subtract a start value from its matching end value and divide by 1e9 for seconds. `image-directory-bytes.txt` is the apparent size of the whole image directory, including logs. Polling, command launch, and timestamp overhead are included, so these values are initial observations rather than precise benchmarks.

## DMTCP native CUDA investigation — 2026-09-14

DMTCP v4.2.0 introduced a native NVIDIA CUDA plugin. The experiments used maintenance commit `b175bb5ccadd2f02d11cf052f586d2d9ac62ad53` from September 7, 2026, without local source patches and with `--disable-prgname-prefix`. This route launches the workload through DMTCP from the beginning; it is not an attach-to-any-process interface. The CUDA driver still stages GPU state, while DMTCP captures and rebuilds the CPU process. [DMTCP v4.2.0 release](https://github.com/dmtcp/dmtcp/releases/tag/v4.2.0), [pinned CUDA plugin](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/plugin/cuda/cuda-ckpt.cpp).

### CPU restore

Run `dmtcp-cpu-02` passed. The original real PID 11118 exited and real PID 11128 restored from a 38,118,736-byte image. Token, file position, and steps 1–25 matched. Image finalization was observed after 0.510 seconds, before the separate filesystem sync.

The controller's completion condition follows the pinned DMTCP write path. DMTCP writes a name containing `.dmtcp.temp`, waits for the checkpoint writer, and then renames the temporary file to its final name. With `--no-gzip`, the default non-forked checkpoint mode, and a fresh run directory, appearance of the final `*.dmtcp` file is an observable image-finalization boundary for this probe. It remains distinct from the later filesystem sync and does not establish durability after host loss. [Temporary-name definition](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/processinfo.h#L114), [writer wait and final rename](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/dmtcpworker.cpp#L490-L504).

An earlier controller attempt checked for the image too soon after the blocking checkpoint command. Correcting the controller to wait for finalization resolved it; that attempt was not evidence of a DMTCP restore failure.

### GPU process restore

Run `dmtcp-gpu-01` produced a 3,535,032,587-byte image (3,535,036,416 allocated filesystem bytes, about 3.29 GiB) for a 67,108,864-byte tensor payload. The original real PID 11208 exited, job B used the GPU, and restored real PID 11262 returned the same full tensor SHA-256 and passed the next addition endpoint checks. DMTCP preserved virtual PID 40000 inside the application. Image finalization was observed after 1.313 seconds, before sync.

```text
tensor SHA-256 before and after:
bcfcc724743f7bf094ad3ecaf64d1d5fcc08e80c5801a5c00d368c99bcf8f709

memory_matches: true
next_gpu_operation_correct: true
job_B_gpu_sum: 1048576.0
```

The strict controller nevertheless rejected unqualified acceptance because four `/dev/zero (deleted)` shared-memory warnings occurred. DMTCP saves the bytes in those regions but restores them as private anonymous memory, so an original sharing or aliasing relationship can be lost. Four writable 2 MiB mappings with distinct backing identities were present; their ownership and driver-side dependencies remain unknown. A separate session-leader warning also appeared. No unsupported `anon_inode:` mapping appeared in the saved pre-checkpoint maps, but later workloads must be checked independently. [Shared-memory serialization](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/writeckpt.cpp#L479-L494), [kernel-backed mapping diagnostics](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/plugin/ipc/file/fileconnlist.cpp#L451-L463).

This establishes restoration of this complete GPU process image and subsequent GPU computation. It does not establish optimizer-state restoration, stochastic continuation, repeated restore, replacement-host restore, a real spot interruption, or LoRA fine-tuning.

### Build and capacity observations

The active PyTorch runtime used CUDA 13.0, while system `nvcc` and `/usr/local/cuda` development headers were CUDA 12.6. Those headers lacked the checkpoint API declarations; CUDA 13 runtime headers alone then lacked `crt/host_defines.h`. The plugin was compiled with isolated `nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-nvcc==13.0.88`, and `nvidia-cuda-crt==13.0.88` headers and linked the installed `libcuda.so.1`. The existing Python environment was not replaced. The include paths identify the headers used more reliably than the Makefile banner, which still reported the 12.6 `nvcc` on `PATH`.

| Resource | Observation |
| --- | --- |
| GPU | NVIDIA L4, 23,034 MiB total |
| Driver | 595.58.03 |
| Container RAM limit | 133,143,986,176 bytes (124 GiB) |
| `/home` free space | 106,499,379,200 bytes at inspection (about 99.2 GiB) |
| Runtime | PyTorch 2.11.0+cu130; native allocator; no allocator override |
| Training packages | Transformers 5.7.0 installed; PEFT absent |

The cgroup RAM limit, rather than host-wide `free`, controls available staging headroom.

## CRIU process capture — 2026-09-14

On Ubuntu 22.04.5 with Linux 6.8.0-110-generic x86_64, both the packaged CRIU 3.16.1 check and a direct dump of the disposable CPU counter returned exit 1:

```text
Unable create a network namespace: Operation not permitted
kerndat_has_nftables_concat failed when initializing kerndat.
Could not initialize kernel features detection.
```

This happens before counter capture. CRIU performs the nftables check in a temporary network namespace; the counter itself does not require networking. The container's effective and bounding capability masks were `a80425fb`, without `CAP_CHECKPOINT_RESTORE`, `CAP_SYS_PTRACE`, or `CAP_SYS_ADMIN`. `Seccomp: 2` shows filtering is active but does not identify the rejected rule. A direct `unshare --user --map-root-user true` also returned `Operation not permitted`, even though visible user-namespace sysctls were enabled. No security settings were changed. [CRIU 3.16.1 feature check](https://github.com/checkpoint-restore/criu/blob/v3.16.1/criu/kerndat.c), [CRIU 4.2.1 permission documentation](https://github.com/checkpoint-restore/criu/blob/v4.2.1/Documentation/criu.txt#L840-L871).

The result is a container-permission blocker for the CRIU route, not a workload requirement and not proof that CRIU cannot work on a suitably permitted host.

## NVIDIA GPU-only suspension — 2026-09-14

The probe used NVIDIA `cuda-checkpoint` commit `00d5cce84c628088d6caa203fc4af40c1538b6f7`, whose x86_64 utility reported version 595.58.03. The CPU process remained alive throughout; no CRIU image or durable process image existed.

| Observation | Before | Suspended | Restored |
| --- | --- | --- | --- |
| CUDA state | running | checkpointed | running |
| `nvidia-smi` process | PID 1029, 252 MiB | absent | PID 1029, 252 MiB |
| Host RSS | 609,740 KiB | 816,396 KiB | 609,832 KiB |

The 64 MiB tensor hash matched before and after. Job B allocated a CUDA tensor and returned the expected sum 1,048,576 while A was suspended; A then performed its next GPU addition correctly. Both processes exited and the final GPU compute-process listing was empty.

Single-run transition observations, including launch and timestamp overhead, were 0.0283 s to lock, 0.2218 s to checkpoint, 0.2069 s to restore, and 0.0296 s to unlock. These are mechanism observations, not training benchmarks. Suspended host RSS rose by about 202 MiB, showing that driver and allocator state exceeded the visible tensor payload.

## EC2 CRIU validation — 2026-09-14

**Decision: use CRIU with its native CUDA plugin for the EC2 POC.** CPU restoration, a complete PyTorch GPU process restore, and a preliminary one-update Qwen LoRA capture/restore all passed. These are new measurements on EC2, separate from the earlier container results. [Curated evidence](evidence/2026-09-14/ec2-criu-summary.json).

### Host and tools

The host runs Ubuntu 22.04.5, kernel `6.8.0-1036-aws`, an NVIDIA A10G with 23,028 MiB, and driver `570.172.08`. Host sudo, user-namespace creation, and root network-namespace creation succeeded. The host execution context reported `Seccomp: 0`. The restricted tool sandbox, in contrast, could not access the GPU and had no effective or bounding capabilities; those sandbox observations do not describe the host's available permissions. Host RAM was about 30 GiB, and the session and ancestor cgroup memory limits were `max`.

CRIU 4.2.1 was built without source patches at `9539417f3e3cfa4eb84c319cd71f4d52f1f08645`, including `cuda_plugin.so`. Build dependencies were downloaded as Ubuntu packages and extracted under ignored `runs/tools/`; no system packages, driver, or security settings were changed. NVIDIA cuda-checkpoint was pinned to `00d5cce84c628088d6caa203fc4af40c1538b6f7` and reported version `570.172.08`. `criu check` passed, with a build warning that libnftables support was absent. All GPU transitions in the restore probes were plugin-managed; no additional manual restore/unlock was issued.

The preliminary probes used the existing host Python 3.10.12, PyTorch 2.8.0+cu128, Transformers 5.0.0, and PEFT 0.18.1 without modifying those packages. This is not yet the plan's isolated, locked training environment. Exact dependency versions, model asset hashes, and local probe source hashes are in the curated evidence. Disposable probe sources and raw artifacts remain under ignored `runs/ec2-check/`.

### Measured restoration

| Probe | Evidence | Observed trial time | Image files |
| --- | --- | --- | --- |
| CPU (`cpu-03`) | Original exit verified; restored token, file offset, and continuation through step 25 matched | 0.59 s | 6,623,420 bytes |
| GPU tensor (`gpu-01`) | Original exit; job B; all 64 MiB of tensor data matched; next addition succeeded | 13.82 s | 645,792,735 bytes |
| Qwen LoRA (`lora-01`) | Original exit; job B; complete recorded boundary state matched; next update matched two uninterrupted references | 63.69 s | 3,899,903,119 bytes |

Times run from workload launch to verification and include diagnostics, filesystem sync, and job B where applicable. Reference creation and initial asset download are excluded. Image sizes sum `.img` files, excluding logs and other run artifacts. These single diagnostic trials are not paired backend benchmarks or the full four-update timing experiment. The LoRA dump command took 10.42 s, post-exit sync took 23.47 s, and the restore phase returned after 7.47 s; storage and state handling already outweigh the tiny amount of training.

The first two CPU controller attempts exposed probe bugs around root-owned PID-file access and CRIU restoring the original numeric PID. After those controller fixes, `cpu-03` passed. They were not CRIU capture/restore incompatibility findings. The controller reaped the original before restoration, adopted/reaped the restored process, and verified every probe PID was absent at the end. The final GPU compute-process listing was empty.

### Preliminary LoRA gate

The model was `Qwen/Qwen2.5-0.5B` at immutable revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, with FP32, eager attention, ordinary LoRA on `q_proj`/`v_proj`, `r=8`, alpha 16, and 540,672 trainable parameters. AdamW and the four-update linear schedule matched the plan. The probe used two small authored examples, batch size 1, answer-only labels, and a 128-token maximum, stopping after update 2.

LoRA dropout was **0.1**. The model entered training mode after adapter construction. Forward hooks verified execution of CUDA dropout in training mode, and CUDA RNG state advanced during each real update. Two uninterrupted references matched exactly. After update 1, initialized Adam state and changed adapter weights were verified before capture. The restored trainer inspected its state before any repair or update 2. Named hashes covered all resident model parameters and buffers, optimizer moments/counters/groups, schedule, cursor, Python/CPU/CUDA RNG, module modes, active adapters, trainable flags, and dropout settings/call counts.

The pre-capture, post-restore, and reference boundary records had SHA-256 `48f33ddc50659302d987fac0e52ee7c2a35d51c17e2c43e432913d72de59cc7e`. The next-update record matched both references with SHA-256 `e5fad952c20e839dfb6d1b387163108ff327d202a12a404e26c1b1cb7ed65e1f`. Losses matched exactly at `2.6493842601776123` and `1.7310601472854614`. The process route did not read an application checkpoint or reference state to restore itself.

### Warnings and limits

CRIU emitted one interrupted-system-call warning in the tensor dump and two in the LoRA dump. In the inspected source, `-ERESTART_RESTARTBLOCK` is restored as `-EINTR`; retain this evidence rather than describing these runs as warning-free. The observed continuation passed. [Pinned register handling](https://github.com/checkpoint-restore/criu/blob/9539417f3e3cfa4eb84c319cd71f4d52f1f08645/compel/arch/x86/src/lib/infect.c#L401-L424).

All seven observed LoRA `/dev/zero (deleted)` ranges remained `rw-s` at the same addresses after restore, backed by new `/memfd:/dev/zero (deleted)` objects. This is stronger mapping evidence than the earlier DMTCP private-memory treatment, but does not independently establish driver-side or external sharing dependencies. No DMTCP-style shared-memory warning appeared in these CRIU logs; that does not resolve the separate historical DMTCP finding.

At this preliminary milestone, the early LoRA continuation gate passed, while the isolated environment, complete application-checkpoint comparison, four-update acceptance, and repeated restoration were still pending. Those later gates passed in the [final acceptance](#four-update-lora-acceptance-and-timing--2026-09-14) below. Replacement-host recovery and spot interruption remain untested. DMTCP was not run on EC2, so these measurements do not establish a performance ranking between backends.

## Isolated LoRA implementation — 2026-09-14

The shared trainer and CRIU controller now repeat the early gate in the locked virtualenv: capture after update 1 with dropout 0.1, original exit, job B, restore inspection, and matching update 2. This passed in **59.21 s**. A four-update zero-dropout trial captured after update 2 and matched updates 3–4 in **62.75 s**. Both have separate pairs of exactly matching uninterrupted references. Eight CPU state/lifecycle tests pass. [Curated milestone evidence](evidence/2026-09-14/isolated-criu-summary.json).

The mapping progression contained zero `/dev/zero (deleted)` ranges after Python, PyTorch import, and CUDA initialization; five after model construction; and seven after Adam initialization. This locates their appearance without proving ownership. Each isolated CRIU dump emitted one interrupted-system-call warning. Numerical and lifecycle checks pass; unqualified compatibility remains false. Full hashes and job B are included in these diagnostic trial times; they are not headline snapshot latencies.

The complete application-checkpoint route also passed at dropout zero: after update 2 it saved 6,705,655 bytes, ended the original, ran job B, rebuilt a fresh trainer, and matched the saved boundary and updates 3–4. The diagnostic trial took 35.44 seconds; nine CPU contract tests pass. [Application evidence](evidence/2026-09-14/application-summary.json).

The full dropout-0.1 application and CRIU comparisons passed, including CUDA dropout execution and RNG advancement on every update. A repeated CRIU run exposed a controller bug: the second restore refused the existing PID filename (`O_EXCL`). Generation-specific PID files fixed it; fresh matching references and the corrected two-capture run passed through update 4 in **104.83 s**. Each capture verified original exit and job B. Ten CPU tests pass, including the PID-file regression. The failed run was cleaned up and its diagnostic evidence retained. [Stochastic and repeated-lifecycle evidence](evidence/2026-09-14/stochastic-summary.json).

## Four-update LoRA acceptance and timing — 2026-09-14

**The bounded experiment passes with qualified compatibility.** Two uninterrupted references match for each dropout setting. Application restart and CRIU restore preserve the full recorded state and continuation; a separate CRIU trial captures after both updates 2 and 3. Thirteen CPU contract tests pass. [Run instructions](finetuning/README.md), [curated evidence](evidence/2026-09-14/finetuning-summary.json).

| Route | Restored boundary | Continuation evidence |
| --- | --- | --- |
| Application, dropout 0 and 0.1 | Update 2 state matches before save and reference | Updates 3–4 losses and full state match |
| CRIU, dropout 0 and 0.1 | Update 2 state matches before capture and reference, before any repair | Updates 3–4 losses and full state match |
| Repeated CRIU, dropout 0.1 | Updates 2 and 3 each match after separate restores | Update 4 matches the reference |

All 48 LoRA dropout modules executed on CUDA in training mode on each stochastic update; the used CUDA generator advanced. Adam state, scheduler, cursor, Python/CPU/CUDA RNG, adapter flags, trainable flags, modes, dropout settings, resident base tensors, adapters, and buffers were compared. Each capture ended the original, verified its exit, observed the GPU, and completed job B before restore. Job B allocated 4 MiB and verified a GPU reduction. The process route read no application checkpoint to restore state.

The final diagnostic trials took **35.66–35.73 s** for application restart, **62.33–62.49 s** for one CRIU capture, and **104.11 s** for two CRIU captures. These include full state hashing, inspection, sync, job B, continuation, and cleanup; reference preparation is excluded. The 2–3-minute warm development-trial target is met. The ordinary zero-dropout reference updates took 0.846, 0.304, 0.289, and 0.292 seconds. More training is unnecessary for this continuation check.

### Paired timing

Three pairs used dropout zero, four updates, capture after two, the same isolated assets, retained warm page cache, and sequential application-then-CRIU execution. Full hashes and inspection waits are excluded from the measured restore path. Asset hashing occurs before launch, and all losses plus the full final state are checked after the next-update endpoint. Each timing run requires matching full correctness evidence. Values below are median (minimum–maximum); file sizes use decimal units.

| Method | Capture request → filesystem sync | Restore request → next completed update | Capture request → GPU observed after exit | Saved bytes |
| --- | --- | --- | --- | --- |
| Application | 1.30 s (1.27–1.33) | 7.95 s (7.82–7.97) | 1.27 s (1.26–1.31) | 6.71 MB |
| Criu | 32.20 s (31.70–32.25) | 4.47 s (4.45–4.57) | 11.49 s (11.10–11.56) | 3.69–3.70 GB |

| Pair | Application capture/sync (s) | Application restore/update (s) | CRIU capture/sync (s) | CRIU restore/update (s) |
| --- | --- | --- | --- | --- |
| 1 | 1.304 | 7.951 | 31.703 | 4.453 |
| 2 | 1.331 | 7.823 | 32.196 | 4.569 |
| 3 | 1.274 | 7.967 | 32.247 | 4.465 |

The GPU observation occurred 32.7–74.3 ms after verified original exit in the timing runs. This is an observation window, not an exact hardware release timestamp. Job B took 2.99–3.09 seconds and is excluded from both headline latency intervals. Application save includes flush/atomic publication, original exit, and the explicit filesystem sync; the save operation alone completed in about 0.15–0.20 seconds. CRIU resumes faster here, but capture and storage costs are much larger than the application checkpoint.

One CRIU timing trial, relative to capture request:

| Event | Controller-clock time |
| --- | --- |
| capture requested | 0.000 s |
| dump completed | 11.061 s |
| original exit verified | 11.062 s |
| gpu observed after exit | 11.098 s |
| filesystem synced | 31.703 s |
| job b completed | 34.735 s |
| restore requested | 34.736 s |
| restore returned | 38.856 s |
| next update completed converted | 39.188 s |

### Clock correction and provenance

The first timing calculation was invalid: the restored trainer's raw monotonic timestamp was compared directly with the controller's clock and produced a negative latency. CRIU's pinned `timens.c` creates a time namespace with an offset based on the captured clock. In the first timing trial the recorded trainer offset was −24.294238681 s; the matrix controller's measured offset was zero. Conversion is `controller timestamp = trainer timestamp − trainer offset + controller offset`. The corrected restore-to-update interval is 4.452699984 s. [Pinned clock restoration](https://github.com/checkpoint-restore/criu/blob/9539417f3e3cfa4eb84c319cd71f4d52f1f08645/criu/timens.c#L52-L120).

The calculation now uses the exact logged offsets and rejects negative latency. A regression test exercises both the invalid mixed-clock calculation and the conversion, including a negative fractional offset. The GPU matrix and raw run files were retained; only postprocessing changed after those runs. The curated report preserves superseded calculations, recorded offsets, original workload source fingerprints, and separate analysis-source fingerprints. The correction was checked against all recorded GPU trials; no new training behavior was introduced.

### Footprint, environment, and limits

During the timing trials, observed trainer RAM high-water marks were **3.395 GiB** for application restart and **3.846–3.858 GiB** for CRIU. Sampling used a 100 ms interval and also recorded the kernel's process high-water counter. Session cgroup peaks were **14.38–14.48 GiB** and **15.84–15.85 GiB**, respectively; those include other processes and file cache and are not attributable solely to the trainer. Minimum host available RAM stayed above **16.95 GiB**. Session/ancestor cgroup limits remained `max`.

Both timing variants peaked at **1.971 GiB allocated VRAM** and **2.088 GiB reserved VRAM**. Diagnostic variants' allocator counters are also retained in the evidence. Tool/trainer process launch is included in request latencies: application launch returned in 0.69–4.44 ms; the full CRIU restore command, including useful restoration work, returned in 4.12–4.22 seconds. These are different phases, not an isolated tool-startup benchmark. Overall trial time excludes controller preflight; timing trials took 23.63–23.84 seconds for application restart and 50.86–51.42 seconds for CRIU.

The isolated environment used Python 3.10.12, PyTorch 2.8.0+cu128, Transformers 5.0.0, PEFT 0.18.1, and safetensors 0.6.2 on the recorded EC2 A10G/driver 570.172.08. Exact packages, immutable Qwen revision, asset hashes, tool commits, and binary hashes are in the evidence. The isolated CRIU build script was also rebuilt successfully. Images were written to ext4 on the root EBS volume; `sync -f` includes work on that filesystem. These measurements do not establish survival of host loss, replacement-host recovery, or spot recovery.

Each CRIU dump retained an interrupted-system-call warning. Seven observed shared `/dev/zero (deleted)` ranges retained their addresses and `rw-s` permissions after restore, and no `anon_inode` mapping was observed in those snapshots. This does not prove external or driver-side sharing ownership; unqualified compatibility remains false. The final GPU compute-process listing was empty, and no experiment trainer, controller, or helper remained.

## Readability refactor validation — 2026-09-17

**The refactored pipelines pass fresh numerical and lifecycle checks.** Reference,
application restart, and CRIU restore now have dedicated files; shared helpers
name the capture, handoff, inspection, and comparison steps. Six agents worked on
implementation, explanations, the teaching plan, and independent review.
[Code walkthrough](finetuning/README.md), [teaching plan](../docs/readability-plan.md),
[curated validation evidence](evidence/2026-09-17/readability-summary.json).

Fourteen CPU checks pass. Review found that cleanup skipped an already-exited
child because splitting its empty command-line bytes produced the nonempty list
`[b'']`. Linux retains an exited child's process record until its parent collects
the exit status; this remaining record is called a **zombie**. Cleanup now checks
the raw bytes and exited state before collecting an unidentified child. A real
child reproduces the regression, and the ownership check also covers a live
child with an empty command line. The fix and regression test share commit
`631ffb6`; the pipeline refactor is in `b373c80`.

Every fine-tuning source file and the process helper are fingerprinted, so earlier
references could not validate edited code. The final sources were held fixed for
all nine fresh cases on the same EC2 A10G/driver 570.172.08 and isolated environment:

| Validation cases | Count | Result |
| --- | --- | --- |
| Independent reference pairs at dropout 0 and 0.1 | 2 | Each pair matched exactly |
| Application restart and CRIU restore, each at dropout 0 and 0.1 | 4 | Boundary state and continuation matched references |
| CRIU captures after updates 2 and 3, with dropout 0.1 | 1 | Both restore inspections and subsequent updates matched |
| One timing run for each restore route | 2 | Final state and all losses matched; timing calculations completed |

The audit checked CUDA RNG advancement on every active-dropout update, the order
of image/save completion, original exit, post-exit GPU observation, filesystem
sync, job B, and restore. Each job B completed its GPU allocation and reduction.
All recorded source hashes matched the final code, all recorded trainer PIDs
were absent afterward, and the GPU compute-process listing was empty.

Diagnostic application trials took 25.84–25.96 seconds; single-capture CRIU trials
took 53.99–54.08 seconds; repeated CRIU capture took 92.51 seconds. These warm trial
durations include diagnostics and handoff. The two single timing checks verify
the refactored measurement paths; they do not replace the September 14 three-pair
benchmark. Individual measurements and their clock offsets remain in the JSON.

All 16 experiment Python files parse, all 65 functions have docstrings, and all
four experiment shell scripts pass syntax checks. Relative document targets were
checked. The short code walkthrough is implemented; deeper OS chapter expansion
and its additional diagrams remain planned. These checks are not a substitute
for that teaching work.

CRIU's interrupted-system-call warnings remain in the evidence. External and
driver-side sharing ownership remains unresolved; every run keeps unqualified
compatibility false. This validation establishes continuation only in the
recorded same-host configuration.

## Environment history

The initial September 12 host observation reported driver 580.126.20 with CUDA 13.0 and neither `criu` nor `cuda-checkpoint` on `PATH`. After the instance resumed on September 14, it reported driver 595.58.03. Workspace persistence therefore did not imply an unchanged driver environment. Each experiment must record its live GPU, driver, runtime, tools, permissions, and cgroup limits.

The local development machine was macOS ARM64 with no NVIDIA GPU tools. It is suitable for authoring and CPU-side checks, but NVIDIA CUDA checkpointing is a Linux feature and was validated only on the Linux L4 host.
