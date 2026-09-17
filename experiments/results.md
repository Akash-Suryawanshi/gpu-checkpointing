# Experiment results

**Same-host LoRA continuation passed on EC2 A10G with CRIU and application
restart.** See the [September 14 benchmark](#four-update-lora-acceptance-and-timing--2026-09-14)
and [September 17 refactor validation](#readability-refactor-validation--2026-09-17).
Sharing ownership and retained warnings still qualify compatibility.

On the earlier **L4 container**, permissions blocked CRIU; NVIDIA GPU-only
suspension passed with the CPU process alive; DMTCP restored a GPU process but
shared-memory warnings prevented strict acceptance. DMTCP was not run on EC2.

Raw images/logs stay in ignored `runs/`: they can expose memory or environment
values. No experiment changed drivers or security settings.

## Run the maintained experiments

Use a Linux NVIDIA host for GPU work, running checks and workloads under the
same user and execution environment. Each run directory must be fresh.
The [LoRA README](finetuning/README.md) gives the isolated training commands.

```bash
# CPU: Python standard library, CRIU, timeout, sync, and setsid required.
bash experiments/check-host.sh cpu
bash experiments/cpu/checkpoint.sh cpu "$PWD/runs/cpu-01"

# GPU-only: PyTorch with CUDA and a compatible NVIDIA helper required.
bash experiments/gpu/probe-gpu.sh \
  /home/gpu-checkpointing-tools/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint \
  "$PWD/runs/gpu-only-02"

# DMTCP: use the actual isolated build path; parent directory must exist.
mkdir -p runs
python3 experiments/dmtcp/probe-dmtcp.py cpu "$PWD/runs/dmtcp-cpu-03" \
  --dmtcp-root /home/gpu-checkpointing-tools/dmtcp-4.2.0
python3 experiments/dmtcp/probe-dmtcp.py gpu "$PWD/runs/dmtcp-gpu-02" \
  --dmtcp-root /home/gpu-checkpointing-tools/dmtcp-4.2.0
```

DMTCP's controller rejects unsupported-resource warnings even when numerical
checks pass. Keep outputs private. `sync -f` flushes the current filesystem;
it does not establish survival of instance deletion.

## Read the CPU evidence

Capture follows record 20; restoration must continue at 21, retaining the random
token and open-file position. Boundary/final positions are **20 / 100 bytes** and
**25 / 125 bytes**, matching the uninterrupted reference. The input must remain
unchanged. The workload reads no checkpoint or evidence log; CRIU may reuse its PID.

Read `comparison.json`, `lifecycle.txt`, `process.jsonl`, dump/restore logs, and
PID files together to establish both continuation and process replacement.
[Mechanism](../docs/01-cpu-checkpointing.md). With CRIT installed:

```bash
crit decode -i runs/cpu-01/images/pstree.img --pretty
```

Raw `pages-*.img` files are not structured CRIT records. In `timing-ns.txt`, divide
matching end-minus-start timestamps by 1e9 for seconds. `image-directory-bytes.txt`
includes logs. These initial observations include polling and command overhead.

## DMTCP native CUDA investigation — 2026-09-14

The L4 probes used unmodified commit `b175bb5ccadd2f02d11cf052f586d2d9ac62ad53`
with `--disable-prgname-prefix`. DMTCP must launch the workload from the start;
its native CUDA plugin was introduced in v4.2.0.
[Release](https://github.com/dmtcp/dmtcp/releases/tag/v4.2.0),
[pinned plugin](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/plugin/cuda/cuda-ckpt.cpp),
[curated evidence](evidence/2026-09-14/dmtcp-summary.json).

### CPU restore

`dmtcp-cpu-02` passed: original PID 11118 exited, restored PID 11128 preserved the
token, file position, and records 1–25. Image size was **38,118,736 bytes**;
finalization was observed after **0.510 s**, before filesystem sync.

For this uncompressed, non-forked, fresh-directory probe, the writer completes
before `.dmtcp.temp` is renamed. Waiting for that rename fixed an earlier
premature controller check. Finalization does not establish durable storage.
[Temporary name](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/processinfo.h#L114),
[finalization](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/dmtcpworker.cpp#L490-L504).

### GPU process restore

`dmtcp-gpu-01` captured a **67,108,864-byte tensor** in a **3,535,032,587-byte image**
(3,535,036,416 allocated filesystem bytes, about 3.29 GiB). Original PID 11208
exited; job B computed **1,048,576.0**; restored PID 11262 matched the full tensor
SHA-256 and passed its next GPU addition. The application retained virtual PID
40000. Finalization was observed after **1.313 s**, before sync.

Four `/dev/zero (deleted)` warnings prevented strict acceptance: DMTCP restores
their bytes as private memory, losing potential sharing. The four writable 2 MiB
mappings' ownership/driver dependencies remain unknown. A session-leader warning
also occurred; no unsupported `anon_inode:` mapping appeared.
[Serialization](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/writeckpt.cpp#L479-L494),
[mapping diagnostics](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/plugin/ipc/file/fileconnlist.cpp#L451-L463).

This establishes tensor-process continuation. LoRA, optimizer restoration,
stochastic/repeated continuation, replacement-host and spot recovery were untested.

### Build and capacity observations

PyTorch used CUDA 13.0; system `nvcc` and development headers were 12.6 and lacked
checkpoint declarations. Runtime headers alone lacked `crt/host_defines.h`.
The plugin used isolated `nvidia-cuda-runtime==13.0.96`,
`nvidia-cuda-nvcc==13.0.88`, and `nvidia-cuda-crt==13.0.88`, linking installed
`libcuda.so.1`. Python packages were unchanged. Include paths, rather than the
Makefile's 12.6 banner, identify the actual headers.

| Resource | Recorded value |
| --- | --- |
| GPU / driver | L4, 23,034 MiB / 595.58.03 |
| Container RAM limit | 133,143,986,176 bytes (124 GiB) |
| `/home` free space | 106,499,379,200 bytes (about 99.2 GiB) |
| Runtime | PyTorch 2.11.0+cu130; native allocator, no override |
| Training packages | Transformers 5.7.0; PEFT absent |

Use the cgroup RAM limit, not host-wide free memory, when sizing staging space.

## CRIU process capture — 2026-09-14

On the earlier Ubuntu 22.04.5 / Linux `6.8.0-110-generic` container, CRIU 3.16.1
check and CPU dump both exited 1 before capture:

```text
Unable create a network namespace: Operation not permitted
kerndat_has_nftables_concat failed when initializing kerndat.
Could not initialize kernel features detection.
```

This was CRIU's startup feature check, not a networking requirement of the
counter. Capability masks were `a80425fb`, lacking `CAP_CHECKPOINT_RESTORE`,
`CAP_SYS_PTRACE`, and `CAP_SYS_ADMIN`; `Seccomp: 2` showed filtering but not its
rejected rule. `unshare --user --map-root-user true` also failed despite enabled
namespace sysctls. No security settings changed. This container failure does not
describe EC2 host permissions.
[Feature check](https://github.com/checkpoint-restore/criu/blob/v3.16.1/criu/kerndat.c),
[permission documentation](https://github.com/checkpoint-restore/criu/blob/v4.2.1/Documentation/criu.txt#L840-L871).

## NVIDIA GPU-only suspension — 2026-09-14

NVIDIA helper commit `00d5cce84c628088d6caa203fc4af40c1538b6f7` reported version
595.58.03. The CPU process remained alive; there was no durable process image.

| Observation | Before | Suspended | Restored |
| --- | --- | --- | --- |
| CUDA state | running | checkpointed | running |
| `nvidia-smi` | PID 1029, 252 MiB | absent | PID 1029, 252 MiB |
| Host RSS | 609,740 KiB | 816,396 KiB | 609,832 KiB |

The 64 MiB tensor hash matched; job B returned **1,048,576** while A was suspended,
and A's next addition passed. Both exited; the final GPU listing was empty.
Lock/checkpoint/restore/unlock took **0.0283 / 0.2218 / 0.2069 / 0.0296 s**,
including command overhead. These are mechanism observations, not training
benchmarks. Suspended RSS rose about 202 MiB, exceeding the tensor payload.

## EC2 CRIU validation — 2026-09-14

**Decision: use CRIU's native CUDA plugin on EC2.** CPU, GPU tensor, and preliminary
LoRA restoration passed. [Curated evidence](evidence/2026-09-14/ec2-criu-summary.json).

### Host and tools

| Component | Recorded configuration |
| --- | --- |
| Host | Ubuntu 22.04.5, `6.8.0-1036-aws`, about 30 GiB RAM |
| GPU / driver | A10G, 23,028 MiB / 570.172.08 |
| Permissions | Host sudo and user/root-network namespaces worked; `Seccomp: 0` |
| Memory limits | Session and ancestor cgroups: `max` |
| CRIU | 4.2.1, unmodified `9539417f3e3cfa4eb84c319cd71f4d52f1f08645`, native CUDA plugin |
| NVIDIA helper | `00d5cce84c628088d6caa203fc4af40c1538b6f7`, version 570.172.08 |
| Preliminary Python | 3.10.12; torch 2.8.0+cu128; Transformers 5.0.0; PEFT 0.18.1 |

The restricted tool sandbox had no GPU access or capabilities; the host did.
Build dependencies were extracted under `runs/tools/`, with no system changes.
`criu check` passed; the build warned that libnftables support was absent. The
CUDA plugin owned all GPU restore/unlock operations. These preliminary probes
used existing host packages; isolated-environment results follow below.

### Measured restoration

| Probe | Verified continuation | Diagnostic trial | `.img` bytes |
| --- | --- | --- | --- |
| CPU `cpu-03` | Original exit; token/file offset/records through 25 matched | 0.59 s | 6,623,420 |
| Tensor `gpu-01` | Original exit; job B; all 64 MiB matched; next addition passed | 13.82 s | 645,792,735 |
| LoRA `lora-01` | Original exit; job B; boundary and next update matched two references | 63.69 s | 3,899,903,119 |

Times include launch, diagnostics, sync, and applicable job B; downloads and
references are excluded. LoRA dump/sync/restore-return phases took
**10.42 / 23.47 / 7.47 s**. These single trials are not backend benchmarks.
Two earlier CPU attempts exposed controller bugs in root-owned PID-file access
and same-PID restoration; both were fixed. All probe PIDs and GPU processes were
absent afterward. Sources and raw artifacts remain in `runs/ec2-check/`.

### Preliminary LoRA gate

Configuration: `Qwen/Qwen2.5-0.5B` revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, FP32, eager attention, ordinary LoRA on
`q_proj`/`v_proj`, rank 8, alpha 16, **540,672 trainable parameters**, AdamW, and a
four-update linear schedule. Two authored examples used batch 1, answer-only
labels, and at most 128 tokens; capture followed update 1, finishing at update 2.

Dropout **0.1** ran on CUDA in training mode and advanced its random generator
on both updates. Adam initialization and changed adapter weights were checked.
Untouched restored state and update 2 matched two references across model,
optimizer, schedule, cursor, random state, and training behavior. Exact record
hashes are in the JSON; losses were **2.6493842601776123** and
**1.7310601472854614**. CRIU read no application checkpoint for restoration.

### Warnings and limits

Tensor/LoRA dumps emitted **one/two interrupted-system-call warnings**. Pinned
register handling maps `-ERESTART_RESTARTBLOCK` to `-EINTR`; continuation passed,
but the warnings remain evidence.
[Source](https://github.com/checkpoint-restore/criu/blob/9539417f3e3cfa4eb84c319cd71f4d52f1f08645/compel/arch/x86/src/lib/infect.c#L401-L424).

Seven LoRA `/dev/zero (deleted)` ranges kept addresses and `rw-s` permissions,
using new `/memfd:/dev/zero (deleted)` backing. This preserves observed shared
mapping flags, not proof of all driver/external sharing relationships. No
DMTCP-style shared-memory warning appeared; DMTCP's separate finding remains.

## Isolated LoRA implementation — 2026-09-14

Each milestone used fresh matching reference pairs in the locked virtualenv:

| Milestone | Diagnostic result | Evidence |
| --- | --- | --- |
| Dropout 0.1, capture 1, continue 2 | 59.21 s; passed | [Isolated](evidence/2026-09-14/isolated-criu-summary.json) |
| Dropout 0, capture 2, continue 3–4 | 62.75 s; passed | Same isolated report; 8 CPU checks |
| Application, dropout 0 | 35.44 s; 6,705,655-byte save; passed | [Application](evidence/2026-09-14/application-summary.json); 9 CPU checks |
| Both routes, dropout 0.1; repeated CRIU at 2 and 3 | Passed; repeated trial 104.83 s | [Stochastic](evidence/2026-09-14/stochastic-summary.json); 10 CPU checks |

Shared `/dev/zero (deleted)` ranges progressed from **0** after Python/PyTorch/CUDA
initialization to **5** after model construction and **7** after Adam. This locates
their appearance, not ownership. Each dump retained one interrupted-call warning.

The first repeated restore failed because CRIU opens its PID file with `O_EXCL`
and rejected the existing filename. Separate filenames per generation fixed it;
the failed trial was cleaned up and its evidence retained. Each successful
capture verified original exit and job B.

## Four-update LoRA acceptance and timing — 2026-09-14

**Acceptance passed with qualified compatibility.** Thirteen CPU checks passed.
[Run instructions](finetuning/README.md),
[curated evidence](evidence/2026-09-14/finetuning-summary.json).

| Route | Exact comparison |
| --- | --- |
| Independent references, dropout 0 and 0.1 | Each pair matched |
| Application and CRIU, each at dropout 0 and 0.1 | Restored update 2 and updates 3–4 matched references |
| Repeated CRIU, dropout 0.1 | Restored updates 2 and 3, then update 4 matched |

Compared state covered resident base/adapters/buffers, Adam, schedule, cursor,
Python/CPU/CUDA random state, modes, adapters, trainable flags, and dropout settings.
All **48** LoRA dropout modules executed on CUDA in training mode for each
stochastic update; the generator advanced. Each original exited before the GPU
observation and job B's 4 MiB allocation/reduction. CRIU used no application save.

Diagnostic durations were **35.66–35.73 s** (application), **62.33–62.49 s** (one
CRIU capture), and **104.11 s** (two captures), including inspection, hashing,
sync, handoff, and cleanup but excluding reference preparation. The 2–3-minute
warm-iteration target was met. Zero-dropout reference updates took
**0.846 / 0.304 / 0.289 / 0.292 s**; longer training was unnecessary for this check.

### Paired timing

Three pairs used dropout zero, four updates/capture 2, identical isolated assets,
warm page cache, and application-then-CRIU order. Matching diagnostic evidence was
required. Asset hashing occurred before launch; restore-path hashing/waits were
omitted, with all losses and final state checked after the next-update endpoint.
Values are median (minimum–maximum); file units are decimal.

| Method | Capture → filesystem sync | Restore → next update | Capture → GPU observed after exit | Saved bytes |
| --- | --- | --- | --- | --- |
| Application | 1.30 s (1.27–1.33) | 7.95 s (7.82–7.97) | 1.27 s (1.26–1.31) | 6.71 MB |
| CRIU | 32.20 s (31.70–32.25) | 4.47 s (4.45–4.57) | 11.49 s (11.10–11.56) | 3.69–3.70 GB |

| Pair | App capture/sync | App restore/update | CRIU capture/sync | CRIU restore/update |
| --- | --- | --- | --- | --- |
| 1 | 1.304 s | 7.951 s | 31.703 s | 4.453 s |
| 2 | 1.331 s | 7.823 s | 32.196 s | 4.569 s |
| 3 | 1.274 s | 7.967 s | 32.247 s | 4.465 s |

GPU observation followed verified exit by **32.7–74.3 ms**; this is observation
uncertainty, not an exact release timestamp. Job B took **2.99–3.09 s**, excluded
from both headline intervals. Application capture includes save publication,
exit, and sync; saving alone took about **0.15–0.20 s**. CRIU restored faster here
but captured much more data at substantially higher cost.

One CRIU trial, relative to capture request:

| Event | Controller-clock time |
| --- | --- |
| Request / dump complete / original exit | 0.000 / 11.061 / 11.062 s |
| GPU observed / filesystem synced | 11.098 / 31.703 s |
| Job B complete / restore requested | 34.735 / 34.736 s |
| Restore returned / next update, clock converted | 38.856 / 39.188 s |

### Clock correction and provenance

The first calculation mixed controller and restored-trainer clocks and produced
negative latency. CRIU's time namespace gives the trainer a clock offset; trial 1
recorded **−24.294238681 s**, while the controller's measured offset was zero:

```text
controller timestamp = trainer timestamp − trainer offset + controller offset
```

The corrected restore/update interval is **4.452699984 s**.
[Pinned clock restoration](https://github.com/checkpoint-restore/criu/blob/9539417f3e3cfa4eb84c319cd71f4d52f1f08645/criu/timens.c#L52-L120).
Conversion uses exact logged offsets and rejects negative results; within-process
update durations need no adjustment. Regression checks include negative fractional
offsets. Only postprocessing changed: all GPU trials were rechecked, and curated
evidence retains raw timestamps, superseded calculations, workload-source hashes,
and separate analysis-source hashes.

### Footprint, environment, and limits

| Timing footprint | Application | CRIU |
| --- | --- | --- |
| Trainer RAM high-water mark | 3.395 GiB | 3.846–3.858 GiB |
| Session cgroup peak, including other processes/cache | 14.38–14.48 GiB | 15.84–15.85 GiB |
| Peak allocated / reserved VRAM | 1.971 / 2.088 GiB | 1.971 / 2.088 GiB |
| Whole trial, excluding controller preflight | 23.63–23.84 s | 50.86–51.42 s |

RAM sampling was every 100 ms, supplemented by the kernel high-water counter.
Host available RAM stayed above **16.95 GiB**; cgroup limits remained `max`.
Application launch returned in **0.69–4.44 ms**; the full CRIU restore command in
**4.12–4.22 s**. These different phases are not an isolated startup comparison.

The isolated environment used Python **3.10.12**, torch **2.8.0+cu128**,
Transformers **5.0.0**, PEFT **0.18.1**, and safetensors **0.6.2**, on the EC2 host
above. Exact packages, asset/tool/binary hashes, and allocator counters are in the
evidence. The isolated tool build also passed. Images used ext4 on root EBS;
sync included that filesystem's work.

Each dump retained an interrupted-call warning. Seven shared mappings kept their
addresses/`rw-s` flags; no `anon_inode` mapping was observed. External/driver
sharing remains unresolved, so unqualified compatibility stays false. No trainer,
controller, helper, or GPU process remained. These trials establish neither
host-loss survival nor replacement-host or spot recovery.

## Readability refactor validation — 2026-09-17

The refactored pipelines passed fresh checks on the same EC2 A10G/driver
570.172.08 and isolated environment. [Walkthrough](finetuning/README.md),
[teaching plan](../docs/readability-plan.md),
[curated evidence](evidence/2026-09-17/readability-summary.json).

Fourteen CPU checks passed. Cleanup had skipped an exited child because splitting
empty command-line bytes produced `[b'']`; it now checks raw bytes and exit state,
preserving live-child ownership checks. A real-child regression accompanies the
fix in `631ffb6`; pipeline refactoring is in `b373c80`.

All nine fresh cases used fixed, fingerprinted sources; old references were not
reused:

| Cases | Count | Result |
| --- | --- | --- |
| Reference pairs at dropout 0 / 0.1 | 2 | Exact matches |
| Application and CRIU, each dropout setting | 4 | Boundary and continuation matched |
| CRIU captures at 2 and 3, dropout 0.1 | 1 | Both restores and continuation matched |
| One timing check per route | 2 | Final state/losses matched; calculations completed |

The audit verified dropout RNG advancement, source hashes, lifecycle ordering,
and job B. All trainer PIDs were absent afterward; the GPU listing was empty.
Diagnostic application/CRIU/repeated-CRIU
durations were **25.84–25.96 / 53.99–54.08 / 92.51 s**. The two single timing checks
validate measurement paths; they do not replace September 14's three-pair benchmark.

All 16 experiment Python files parsed, 65 functions had docstrings, four shell
scripts passed syntax checks, and relative document targets were checked at that
milestone. Those checks do not establish teaching completeness. Interrupted-call
warnings and unresolved sharing ownership still qualify same-host compatibility.

## Environment history

The September 12 L4 host had driver **580.126.20**, CUDA 13.0, and no CRIU/NVIDIA
helper on `PATH`; September 14's resumed host had **595.58.03**. Recheck versions,
permissions, and memory limits even when the workspace survives.

Initial authoring used macOS ARM64 without NVIDIA tools. CUDA checkpointing was
validated on Linux hosts; local CPU checks are not GPU restoration evidence.
