# Experiment results

**Same-host LoRA continuation passed on EC2 A10G with CRIU and application
restart.** See the [September 14 benchmark](#four-update-lora-acceptance-and-timing--2026-09-14)
and [independent lifecycle validation](#independent-lifecycle-validation--2026-09-17).
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

## Code simplification validation — 2026-09-17

The code-only pass (`eeb22a3`) shares restore bookkeeping and hash/JSON utilities,
and removes one single-use helper. Original explanatory comments/docstrings are
retained; see the [pipeline flow](finetuning/README.md#read-one-pipeline-from-start-to-finish).

Fresh references and all nine cases in the matrix above passed again, along with
14 CPU checks. Report output matched the baseline except for analysis-source
hashes; [curated evidence](evidence/2026-09-17/code-simplification-summary.json)
records the exact sources and cleanup audit.

The single timing checks validate execution paths, not a new benchmark.
Same-host compatibility remains qualified.

## Environment history

The September 12 L4 host had driver **580.126.20**, CUDA 13.0, and no CRIU/NVIDIA
helper on `PATH`; September 14's resumed host had **595.58.03**. Recheck versions,
permissions, and memory limits even when the workspace survives.

Initial authoring used macOS ARM64 without NVIDIA tools. CUDA checkpointing was
validated on Linux hosts; local CPU checks are not GPU restoration evidence.

## Independent lifecycle validation — 2026-09-17

**30 CPU checks, all 15 GPU matrix cases, and a separate standalone recovery
passed** on A10G / driver 570.172.08 / pinned CRIU 4.2.1. The published
[source and validation record](evidence/2026-09-17/independent-validation.json)
identifies the exact tested code; later commits only add documentation/evidence.

The matrix covers paired uninterrupted references, dropout 0 and 0.1,
application and CRIU continuation, capture after update 1, two sequential
captures at updates 2 and 3, and three paired timing repetitions. Diagnostic
states and losses match their references; dropout execution and RNG advancement
were checked. Timing runs use the corresponding diagnostic admission.

The separate recovery began after the capture worker **and its launcher** exited.
A fresh restore worker compared untouched state, released training, and exited;
the trainer completed both updates and was reaped. Job B successfully used the
GPU between capture and restore; final GPU process observations were empty.
See the [lifecycle sequence](../docs/independent-lifecycle-plan.md#architecture).

| Warm measurement | Seconds |
| --- | ---: |
| Four-update CRIU diagnostics, one capture (two dropout settings) | 66.04–66.68 |
| Four-update CRIU diagnostic, two captures | 125.63 |
| Application timing: capture to sync, median (range), n=3 | 0.964 (0.915–0.964) |
| CRIU timing: worker start to publication, median (range), n=3 | 44.69 (44.38–48.55) |
| Application timing: restore to next update, median (range), n=3 | 5.01 (4.94–5.04) |
| CRIU timing: restore to next update, median (range), n=3 | 10.20 (10.11–11.91) |

Diagnostic durations are outer CLI wall time, including interpreter startup;
`trial_seconds` begins inside the runner. CRIU publication includes dependency
checks, hashing, payload sync, and publication; its request-to-ready interval
(12.49–12.75 seconds in timing runs) includes trainer initialization and updates.
CRIU restore includes admission and untouched-state inspection; application
timing defers diagnostics. These boundaries differ, so this is not a speed ranking.

[Per-run evidence](evidence/2026-09-17/independent-runs.csv) includes wall times,
artifact sizes, verdicts, and result hashes. [Per-generation evidence](evidence/2026-09-17/independent-generations.csv)
preserves the cost breakdown and measured clock corrections. Each trial verified
unchanged snapshot payload hashes after restore; the final audit checked manifests,
sizes, ordering, source identity, and absence of 20 recorded process identities.

CPU checks cover ownership, worker death, bounded locking, cancellation/expiry,
publication failures at write/sync/rename operations, artifact rejection, and
inspection mismatch preventing continuation. The acknowledged-pause expiry
check specifically prevents a timeout from racing dump arming.

CRIU's interrupted-system-call warnings remain recorded. Success applies to this
bounded workload on the same host/boot with matching dependencies and persistent
local storage; it does not establish arbitrary resource-sharing compatibility.
Ordered persistence is implemented and fault-tested, but power loss, volume
deletion, replacement hosts, and spot recovery were not tested.

## Inference activation and GPU reuse — 2026-09-17

The pinned Qwen3-8B model completed all four routes on one A10G in BF16
(16-bit floating point). [Curated validation](evidence/2026-09-17/inference-8b-validation.json)
records source `6a7ace0`, five diagnostics, twelve scheduled timings, and a
separate capture/restore demonstration; [individual measurements](evidence/2026-09-17/inference-8b-runs.csv)
retain each block.

```text
resident: GPU model stays loaded -------------------------> request
RAM:      GPU bytes -> live worker's RAM -> job B -> GPU --> request
disk:     GPU + CPU state -> files -> original exits
          -> job B -> new process from files -> verify ---> request
```

Arrows describe state movement and sequence. RAM recovery needs the same living
CPU process; disk recovery succeeded after that process and the capture command
had ended. All routes also require the pinned local model files.

| Route | Median first-token seconds | Min–max seconds | Trials |
| --- | ---: | ---: | ---: |
| fresh | 11.1522 | 10.9967–11.4176 | 3 |
| resident | 0.0762 | 0.0737–0.0776 | 3 |
| ram | 4.0378 | 4.0332–4.0568 | 3 |
| disk | 277.6203 | 274.0257–279.3674 | 3 |

Primary time ends at the controller's first-token receipt. Fresh starts before
launch, resident before the request, RAM before GPU restoration, and disk before
its independent restore command, including validation. Full model audits occurred
after response one in timing runs; the second request was a separate health check.

Median RAM-parking host retention was 16.95 GiB, with no reported GPU allocation.
Resident held 15.29 GiB of reserved GPU memory. Disk retained no original worker and
added 16.61 GiB of snapshot files; the local model files remained required.

Both released-state routes let a separate process allocate, touch, and check
12 GiB before waking the model. This exceeded pre-release free GPU memory by
at least 1 GiB; it proves more than the smaller availability probe alone.

Saving took a median 10.92 seconds for RAM and 279.18 seconds for disk, outside the
primary timer. Disk recovery's primary measurement includes file/dependency
validation, CRIU reconstruction, and the pre-request metadata check; these costs
were not subtracted.

Admission validation reads model files before worker launch; their cache residency
was uncontrolled. Five-millisecond request polling and durable
file writes contribute overhead; sampled resource peaks can miss short spikes.

Three timings per route do not establish tail latency, and same-host/boot recovery
does not establish replacement-host or spot recovery. CRIU capture reported
interrupted-system-call warnings, retained in the curated record and raw logs.

The [runbook](inference/README.md) owns commands. Historical exploratory attempts
remain separate: the first preparation lacked the direct loader's cuBLAS setting,
and an earlier campaign was intentionally interrupted to fix helper cancellation.
No failed timing slot was replaced.

Subsequent cancellation and report fixes have
separate regression checks; their container validation uses a new comparison key.

## Cold-cache activation on EBS — 2026-09-18

With the model-file cache verified empty before each timer, the pinned Qwen3-8B
fresh start and the CRIU snapshot restore were both bound by the data volume:
gp3 at 125 MiB/s provisioned throughput. [Curated validation](evidence/2026-09-18/cold-ebs-01-validation.json)
records source `8769fc8`, two diagnostics, and six scheduled timings;
[individual measurements](evidence/2026-09-18/cold-ebs-01-runs.csv) retain each block.

```text
fresh:    evict -> launch -> read 15.26 GiB safetensors once ---------------------> token
snapshot: evict -> hash image 16.6 GiB -> hash model 15.3 GiB -> CRIU reads image 16.6 GiB -> token
          |<-- 134.9 s -->|<-- 125.1 s -->|<-- 139.3 s -->|<- 0.6 s ->|
```

Arrows show the order of disk passes; times are the block medians from the
host-clock profile. Kernel device counters confirm the traffic is physical.

| Route | Median first-token seconds | Min–max seconds | Device reads per activation |
| --- | ---: | ---: | ---: |
| fresh | 131.11 | 129.46–132.22 | 15.26 GiB |
| snapshot (disk) | 401.23 | 401.14–401.39 | 48.49 GiB |

Paired within blocks, the snapshot route took 3.03–3.10 times longer than fresh.
Every snapshot activation hashed the 16.6 GiB image and the 15.3 GiB model files
before CRIU read the image again; each pass ran at the volume's cap. The
[activation contract](../docs/inference-activation-contract.md) does not change this
byte count; the plan's storage and integrity-policy arms address it.

Warm follow-up requests took 0.07–0.08 s on both routes. Each snapshot timing
built its own image (median save 246.8 s) and discarded it after the final hash
check. CRIU logged one interrupted-system-call warning per restore, retained in
the curated record.

Three trials per route do not establish tail latency. Runtime and library caches
were not evicted; device counters include any other reader of the volume, which was
otherwise idle. The historical 11 s fresh result had uncontrolled cache residency and
is not comparable.

## The container reproduces every host result — 2026-09-18

The whole comparison was repeated inside the pinned container: namespace, CPU and
GPU restore probes, asset preparation, diagnostics, a capture published from
inside the container, and six endpoint trials against one reusable image.
[Curated measurements](evidence/2026-09-18/container-nvme-01.json) record source `ff694fa`.

| Measurement | Container | Host |
| --- | ---: | ---: |
| fresh diagnostic, cold cache | 54.72 s | 54.95 s |
| snapshot diagnostic, cold cache | 178.57 s | 178.35 s |
| endpoint fresh, median of three | 54.95 s | 54.30 s |
| endpoint snapshot, median of three | 164.43 s | 164.19 s |

Every trial passed with exact reference output. A container supplies user-space
software while the kernel and GPU driver stay on the host, so this establishes
that the pinned image reproduces the measurements, not that a captured process
survives a different host or driver.

Host and container results are kept separate rather than pooled. The `sudo`
hostname warnings come from the disabled network and are retained in the logs.

## Both volumes are bandwidth-limited, and our loaders already saturate them — 2026-09-18

Every loading and restore path measured on local flash returned between 0.31 and
0.34 GB/s regardless of concurrency, which looked low enough to suspect our own
software. Reading the raw device settles it.
[Measurements](evidence/2026-09-18/storage-bandwidth.txt): 3 GiB per case,
cache bypassed, distinct ranges of one weight shard, two passes.

| Volume | 1 stream | 4 streams | 8 streams |
| --- | ---: | ---: | ---: |
| EBS gp3 | 0.13 GB/s | 0.13 GB/s | 0.13 GB/s |
| instance NVMe | 0.32–0.39 GB/s | 0.32 GB/s | 0.32 GB/s |

Concurrency changes nothing on either volume, which is the signature of a
throughput cap rather than a queue-depth limit. EBS matches its 125 MiB/s exactly.

So our loaders are already running at device speed, and no faster I/O path exists
to be written here. The ten seconds available from overlapping construction with
the first reads remain available, because that cost is not I/O.

A first attempt at this probe reported 1.95 GB/s on NVMe from a 0.27 second
sample, which did not survive a longer read; its highest-concurrency row was
invalid as well, requesting more data than the file held. Both the retracted
figures and the method are kept in the evidence file.

## The endpoint boundary costs nothing measurable — 2026-09-18

A live HTTP service activated the model on demand, either by launching a worker or
by restoring a published image. Clients opened a new connection per request and
timed to the first streamed token. [Curated measurements](evidence/2026-09-18/http-ebs-03.json)
record source `1441343`; both volumes used the strict integrity policy.

| Volume | Route | Endpoint median | Command-line median | Difference |
| --- | --- | ---: | ---: | ---: |
| EBS | fresh | 129.49 s | 131.11 s | −1.62 s |
| EBS | snapshot | 401.21 s | 401.23 s | −0.02 s |
| NVMe | fresh | 54.30 s | 57.12 s | −2.82 s |
| NVMe | snapshot | 164.19 s | 164.10 s | +0.09 s |

```text
client --new TCP connection--> API --> launch worker, or restore published image
  START before connect                        one image, three activations
  STOP at first token event <--ndjson token--<
```

Twelve trials, three pairs per volume, every correctness check passed, headers
arriving in 0.001 s. The request path therefore adds nothing to either route on
either volume, so the command-line findings transfer directly to a
request-triggered deployment. The fresh route reads slightly faster through the
endpoint because the server fingerprints the model once at startup, before the
trial evicts caches, rather than inside the measured activation. The
[NVMe arm](evidence/2026-09-18/http-nvme-03.json) is curated separately.

The snapshot figures come from one published image activated three times in
sequence, which validates the [activation contract](../docs/inference-activation-contract.md)
on the host: an image published by one process, restored on demand by a live
server, served, and released, three times over.

`followup_ttft_seconds` in this campaign is not a warm-request latency. The worker
fingerprinted every weight straight after the first response and read no request
meanwhile, which a command-line controller hid by waiting and an HTTP client
cannot. Fixed after these images were published; warm latency is 0.07 s across
every command-line campaign.

## Packed loaders win the data path and lose it again to serialization — 2026-09-18

The installed Transformers default was compared against the two packed loaders on
a cold file cache, fresh route only. [Curated measurements](evidence/2026-09-18/cold-nvme-loaders.json)
record source `8769fc8`.

| Loader | Median first token | Trials | Construct | Read, verify, copy |
| --- | ---: | ---: | ---: | ---: |
| Transformers default | 54.95 s | 3 | overlapped with loading | — |
| packed, four readers | 63.08 s | 2 | 9.78 s | 48.78 s |
| packed, direct I/O | 63.83 s | 3 | 9.60 s | 48.78 s |

The packed data path is faster than the entire default loader, 48.78 s against
53.29 s, while additionally verifying every 16 MiB chunk with SHA-256, which the
default never computes. It gives the advantage back by building the empty model
serially before reading starts; the installed default overlaps construction with
its own threaded loading.

Bypassing the file cache changed nothing: both packed loaders moved the data in
the same 48.78 s. On a cold cache there are no pages to avoid, and the volume is
the limit either way. Every path measured on this volume, including four
concurrent readers, lands between 0.31 and 0.34 GB/s.

Overlapping construction with the first reads is therefore the available gain,
roughly ten seconds, and it applies to the route that already wins. One pipelined
block failed in cold-cache eviction and is retained rather than replaced.

## A smaller model and a compiled model both fail to help — 2026-09-18

Two hypotheses for making restoration win were tested as single diagnostics on
Qwen2.5-0.5B, not as timing campaigns.
[Curated probes](evidence/2026-09-18/compile-and-small-model-probes.json) record source `bd6defe`.

| Probe | First token | Outcome |
| --- | ---: | --- |
| fresh | 11.12 s | passed |
| snapshot | 17.95 s | passed |
| fresh, compiled | — | failed: output differs from the uncompiled reference |
| snapshot, compiled | — | failed: capture refused the compiled process |

**A smaller model is the worse case.** Its snapshot took 1.61 times a fresh start,
against 1.22 for the 8B model. Part of an image does not shrink with the model:
the interpreter, the libraries, and the device context. The 8B image is 1.09 times
its weights; this one is closer to three times.

**A compiled model cannot be captured by this toolchain.** Compilation was the
remaining way to give an activation state that no weights file holds, but the dump
fails before writing anything:

```text
handle_device_vma plugin failed: No such file or directory
Can't handle non-regular mapping on 2844526's map 73c4ce64c000
Dumping FAILED.
```

Generated kernels are loaded as device mappings the pinned CRIU build and its CUDA
plugin do not recognise. Separately, the compiled model's greedy output diverged
from the prepared reference at token twelve, so a compiled campaign would need a
reference produced the same way; the existing correctness gate caught that unaided.

Published work reducing inference cold start relies on engines whose startup is
dominated by compilation and on checkpoint tooling that supports those mappings.
Reproducing that needs a different serving stack, not another flag here.

## The snapshot route at its floor still trails a fresh start — 2026-09-18

Under `publication-verified-v1` an activation stops rereading the model files and
reads the image exactly once. [Curated validation](evidence/2026-09-18/cold-nvme-policy-01-validation.json)
and [measurements](evidence/2026-09-18/cold-nvme-policy-01-runs.csv) record source
`19f77b7`; publication still hashed model content, and every other check stayed on.

| NVMe campaign | Model check at activation | Snapshot median | Device reads | Snapshot ÷ fresh |
| --- | --- | ---: | ---: | ---: |
| `cold-nvme-01` | contents, 15.26 GiB | 164.10 s | 48.49 GiB | 2.85–3.03 |
| `cold-nvme-flags-01` | contents, reordered | 119.42 s | 31.89 GiB | 2.09 |
| `cold-nvme-policy-01` | identity only | 69.84 s | 16.61 GiB | 1.21–1.29 |

Dependency validation fell from 50.63 s to 0.17 s and the campaign's own fresh
median was 57.24 s. Removing the model pass also removed the cache eviction: with
nothing running between the image hash and CRIU, the default order already leaves
the image resident, and CRIU took 14.48 s instead of 58.32 s. The ordering fix and
the policy fix address the same eviction, so they do not add up.

```text
strict:   hash image 53.3 -> hash model 50.6 -> CRIU rereads from disk 58.3 = 164 s
policy:   hash image 53.3 -> identity 0.2 ---> CRIU reads from memory   14.5 =  70 s
fresh:    read 15.26 GiB of weights ------------------------------------------ = 57 s
```

At this floor the snapshot route reads 16.61 GiB against a fresh start's 15.26 GiB
and remains 21% to 29% slower in every block pair. The remaining cost is the image
hash plus reconstruction, and the image is intrinsically larger than the weights
because it holds 15.29 GiB of GPU memory and 1.39 GiB of process memory. For this
workload a snapshot stores no expensive derived state, so it cannot overtake a
loader reading the same weights from disk.

`publication-verified-v1` is an experiment, not a recommendation, and remains
opt-in. It detects replacement and truncation, not a careful in-place rewrite;
the [contract](../docs/inference-activation-contract.md) states the trade.

## Validation order decides whether CRIU reads the image from cache — 2026-09-18

Hashing the snapshot payload last, immediately before CRIU consumes it, leaves
the image in the kernel's file cache. The same activation then reads it from
memory instead of disk. [Curated validation](evidence/2026-09-18/cold-nvme-flags-01-validation.json)
records source `8769fc8`; every integrity check stayed enabled.

| NVMe campaign | Validation order | Snapshot median | Device reads | CRIU phase |
| --- | --- | ---: | ---: | ---: |
| `cold-nvme-01` | payload first | 164.10 s | 48.49 GiB | 58.32 s |
| `cold-nvme-flags-01` | dependencies first | 119.42 s | 31.89 GiB | 14.02 s |

```text
payload first:      hash image -> hash model 15.3 GiB -> image evicted -> CRIU rereads from disk
dependencies first: hash model -> hash image ---------> still cached  -> CRIU reads from memory
```

The saving is one full image pass, 16.61 GiB. Host memory is 30 GiB and the
image is 16.61 GiB, so the two minutes spent hashing model files in between are
enough to evict it. Time before CRIU was unchanged at 104.8 s against 105.3 s.

The flags campaign applied the setting to its restore route only, so it holds two
comparison keys and `report.py` declines to aggregate it; that refusal is correct.
Its fresh runs carry the identical key to the strict campaign's, which confirms
validation order does not touch the fresh path.

That arm also raised hash workers from one to four. A separate
[order-only campaign](evidence/2026-09-18/cold-nvme-order-01-validation.json) keeps
one worker and reaches a 119.70 s median against the combined arm's 119.42 s, so
the reordering explains the whole gain and the worker count contributes nothing
measurable. Its single comparison key lets `report.py` aggregate it normally.

## What a restored image actually reopens — 2026-09-18

Activation-time validation hashes 15.26 GiB of model files on every restore.
The [file audit](evidence/2026-09-18/restored-image-files.json) shows the restored
process reopens 92 files and none of them is a model file.

| Reopened by the restored process | Count |
| --- | ---: |
| Shared libraries (PyTorch, CUDA runtime, NumPy, tokenizers) | 83 |
| Captured run logs (`updates.jsonl`, `trainer.stderr`) | 2 |
| Interpreter, locales, working directory, `/dev/null` | 7 |
| Model weight files | 0 |

The weights are in the image's restored memory pages, so the 12 hashed model
files are an environment-equivalence check rather than a restore prerequisite.
`experiments/inference/image_files.py` reproduces this from any activated run.

This does not by itself justify skipping the check. Moving it to publication
time would trade a per-activation cost for an assumption that the files stay
unchanged on writable storage. That trade is a separate integrity policy, and a
measurement using it must record a different policy label so results never pool.

## Storage arm: the same comparison on instance-store NVMe — 2026-09-18

Only the storage changed: model files, images, and caches moved to the
g5.2xlarge instance store, with every validation check retained.
[Curated validation](evidence/2026-09-18/cold-nvme-01-validation.json) and
[individual measurements](evidence/2026-09-18/cold-nvme-01-runs.csv) use source `8769fc8`.

| Storage | fresh median (min–max) | snapshot median (min–max) | snapshot ÷ fresh |
| --- | ---: | ---: | ---: |
| EBS gp3, 125 MiB/s | 131.11 s (129.46–132.22) | 401.23 s (401.14–401.39) | 3.03–3.10 |
| instance NVMe | 57.12 s (54.29–57.66) | 164.10 s (164.07–164.23) | 2.85–3.03 |

```text
NVMe snapshot: hash image 53.3 s -> hash model 50.6 s -> CRIU 58.3 s -> 0.6 s
               16.6 GiB           15.3 GiB             16.6 GiB
```

Each pass moved about 300 MB/s on NVMe, for hashing, CRIU, and the Transformers
loader alike, against 125 MiB/s on EBS. Faster storage shortened both routes by
2.3–2.4 times and left the three-pass structure, and its ratio, unchanged.

Whether 300 MB/s is the device or single-threaded hashing is not separated
here; the validation-worker arm tests that. Instance store is lost when the
instance stops, so staging cost and durability remain deployment questions.

## Container compatibility — 2026-09-17

The [pinned image](inference/Dockerfile) passed 47 CPU checks, actual namespace
creation, CPU-process restoration, and a 64 MiB GPU-tensor restoration with a
verified next operation. It then passed two fresh and one resident, RAM, and disk
8B diagnostic; [curated evidence](evidence/2026-09-17/inference-container.json)
records the image, permissions, source hashes, and result digests.

```text
image: Python + snapshot tools -----+
mounts: model + assets + results ---+--> probes -> five 8B diagnostics
host: Linux kernel + GPU driver ---+                -> exact state and outputs
```

A container supplies user-space software, but still uses the host's kernel and
GPU driver. All diagnostics ran inside the same privileged container on the A10G;
this does not establish recovery after container restart or on another host.

Prepared tokens, output, and weight fingerprints matched host preparation.
The final cancellation/report sources used a new compatibility key; container
diagnostic times were not pooled with the host's three timing blocks.

CRIU warned about absent libnftables support (Linux firewall integration) and an
interrupted system call; networking recovery was not tested. The `sudo` hostname
warnings were nonfatal. All warnings remain in the evidence.

The [runbook](inference/README.md#container-validation) gives the tested mounts,
permissions, and probe commands; no smaller model or larger GPU was needed.

## vLLM: a real compile cost, and a snapshot that still loses — 2026-09-18

Qwen2.5-0.5B on the A10G, vLLM 0.19.1, three timing blocks per route with a
verified cold file cache, all eight runs under one comparison key. The
[plan](../docs/vllm-snapshot-plan.md) owns the requirements; the
[runbook](vllm/README.md) owns the commands.

| Route | Median TTFT | Range | Device bytes read |
| --- | --- | --- | --- |
| cold | 27.65 s | 27.49–27.69 | 0.92 GiB |
| snapshot | 30.31 s | 30.24–30.40 | 6.42 GiB |

The engine has the startup cost our own stack never had: weights load in 0.17 s
while `LLM()` takes 22.3 s with a warm compiled-kernel cache and 47.3 s without.
That is compilation and CUDA-graph capture, and no file can supply it.

```text
cold      [read 0.92 GiB][compile + capture graphs ~22 s]   27.65 s
snapshot  [read 6.42 GiB ................................]  30.31 s
```

The snapshot removes the computation and pays for it in reads: 6.42 GiB against
0.92 GiB, at the 0.21 GiB/s these volumes deliver. Restoration is storage-bound
here, so it loses by 2.66 s in every block. The image also does not replace the
weight file — CRIU re-reads the 0.93 GiB mapping as well as the 5.47 GiB image.

Capture worked, and not where the plan expected it to fail. `handle_device_vma
plugin failed` never appeared; the CUDA plugin checkpointed the devices in about
three seconds. Two PyTorch behaviours blocked the dump first: the libuv
distributed store's io_uring ring, and its TCP connection to itself. See the
[knowledge updates](../AGENTS.md#knowledge-updates) for both.

Scope: this disproves the hypothesis **on this hardware with stock CRIU**, not in
general. NVIDIA's Dynamo Snapshot restores a comparable checkpoint in 2.4 s using
patched CRIU (native AIO, parallel memfd, `O_DIRECT`) on striped NVMe, with the
key-value cache unmapped and weights restored outside the image. Our 5.47 GiB
image is already their size; our read rate is roughly ten times slower. That
rate comparison is misleading, and the [H100 arm](#h100-storage-arm-the-disk-stops-binding-and-our-own-reader-takes-over--2026-09-18)
shows why: most of it is our own integrity check, not reading. The
eager control was built and left unrun at the user's request.

## H100 storage arm: the disk stops binding, and our own reader takes over — 2026-09-18

The same comparison on an H100 80GB host whose volume reads 16x faster. Faster
storage did not reverse the A10G result: the gap widened from 2.66 s to 5.96 s.
[Validation](evidence/2026-09-18/vllm-h100-01-validation.json),
[phases](evidence/2026-09-18/vllm-h100-01-phases.json),
[script](evidence/2026-09-18/vllm-h100-01-run-campaign.sh).

Driver 580.126.20, kernel 5.15.0-171. vLLM 0.19.1 and torch 2.10.0+cu128 held
identical to the A10G arm, so only the hardware differs. Eight runs, one key,
`--gpu-fraction 0.063` for 9238 key-value blocks and a 7.37 GiB image.

| Route | Median TTFT | Range | Device reads | A10G |
| --- | ---: | --- | ---: | ---: |
| cold | 15.96 s | 15.95–15.98 | 0.92 GiB | 27.65 s |
| snapshot | 21.92 s | 21.76–22.16 | 8.29 GiB | 30.31 s |

Two things moved against the snapshot. The H100 cut `LLM()` from 22.3 s to
12.1 s, shrinking the prize by 10 s. And matching the block count rather than the
fraction grew the image, because vLLM reserves 3.37 GiB of non-key-value headroom
here against roughly 1.70 GiB there.

### The bottleneck moved off the disk

[Raw reads](evidence/2026-09-18/storage-bandwidth-h100.txt) reach 5.4 GB/s at
eight streams or `bs=64M` on one, against the A10G's flat 0.32 GB/s. At that rate
the image is 1.5 s of reading, so a storage-bound snapshot would have won.

| Phase | Median | Bound by |
| --- | ---: | --- |
| payload validation | 14.24 s | serial 4 MiB read-and-hash |
| dependency validation | 1.67 s | the same |
| CRIU restore | 4.63 s | page cache, then PCIe |
| continuation to first token | 1.30 s | handshake |

```text
A10G      [===== read 6.42 GiB at 0.21 GiB/s =====]  30.31 s  disk-bound
H100      [== hash 7.37 GiB at 0.52 GiB/s ==][CRIU]  21.92 s  reader-bound
                                              ^^^^ all the snapshot really costs
```

`control.inventory()` hashes serially and `prepare.file_hash()` reads 4 MiB at a
time. Measured alone on the real page image: 0.78 GB/s to read, 1.80 GB/s to hash
from RAM, 0.67 GB/s for both — a tenth of the device. The device improved 16x
across hosts; validation improved 2.5x, never having read at device speed.

CRIU itself is cheap because validation just pulled the image into a 196 GiB page
cache: 7.32 GiB of pages in 3.02 s, then 1.3 s of CUDA device resume. The
[validation-order fix](#validation-order-decides-whether-criu-reads-the-image-from-cache--2026-09-18)
is moot here — nothing evicts the image, and counters show one image pass.

### Against the Dynamo figure

The [A10G note](#vllm-a-real-compile-cost-and-a-snapshot-that-still-loses--2026-09-18)
calls NVIDIA's 2.4 s restore ten times faster than our read rate. That compares
unlike quantities: their 2.4 s is a restore path, and nothing in the cited source
says it re-hashes the payload per activation.

| Quantity | Seconds |
| --- | ---: |
| Dynamo Snapshot restore, as cited | 2.4 |
| our intrinsic restore: CRIU pages plus CUDA resume | 4.63 |
| our full `strict-v1` activation | 21.92 |

Against the comparable middle row we are about twice as slow, not ten times, and
they hold the advantages that note lists. Their integrity policy is unknown here.

## Two payload policies: the hash was also a prefetch — 2026-09-18

Two campaigns test the obvious repairs to that 14.24 s: stop hashing at
activation, and hash in parallel. Each is a separate `integrity_policy` with its
own comparison key, eight runs each.
[Skip](evidence/2026-09-18/vllm-h100-skipval-validation.json),
[parallel](evidence/2026-09-18/vllm-h100-parallel-validation.json),
[phases](evidence/2026-09-18/vllm-h100-payload-policies.json),
[script](evidence/2026-09-18/vllm-h100-policy-experiments.sh).

| Policy | Payload | CRIU | Snapshot | Cold | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `strict-v1` | 14.24 s | 4.64 s | 21.92 s | 15.96 s | loses 5.96 s |
| `publication-only-v1` | 0.00 s | 11.20 s | 14.01 s | 15.98 s | wins 1.97 s |
| `parallel-chunked-v1` | 1.54 s | 4.59 s | **8.99 s** | 15.80 s | **wins 6.81 s** |

Cold is the control and does not move. The policy reaches the snapshot path only.

### Skipping the check returns less than half of what it cost

Removing the hash should have saved 14.24 s. It saved 7.91 s, because CRIU's
restore grew from 4.64 s to 11.20 s in the same campaign.

```text
strict    [ hash 7.37 GiB 14.2s ][CRIU 4.6s]   image already in RAM
skip      [                     ][CRIU 11.2s]  CRIU reads it from disk itself
parallel  [1.5s][CRIU 4.6s]                    verified and still in RAM
```

The hash did two jobs: it proved the image unchanged, and it pulled 7.37 GiB into
the page cache so CRIU read from memory at 2.60 GB/s instead of disk at
0.62 GiB/s. Delete the check and the prefetch goes with it.

So neither reader ever approached the device. Validation moved 0.52 GiB/s, CRIU
alone 0.62 GiB/s, against a volume that does 5.4 GB/s.

### Hashing in parallel beats not hashing at all

`parallel-chunked-v1` covers every byte with SHA-256, folding 64 MiB chunk
digests in file order. At 16 workers it verifies 9.2x faster than `strict-v1` and
still warms the cache, keeping CRIU's 4.59 s.

| Workers | 1 | 4 | 8 | 16 | 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GB/s on the 7.32 GiB page image | 0.61 | 2.38 | 4.22 | 5.51 | 6.93 |

Its digest value differs from an ordinary SHA-256, which is why it is a labelled
policy and not a faster `strict-v1`. The guarantee does not differ: a byte flipped
anywhere changes it, as does swapping two chunks that keep every byte, and the
value never depends on the worker count.

The honest repair is therefore not to trust the image less. Skipping trades
tamper detection for 1.97 s; parallel hashing keeps it and wins 6.81 s.
`publication-only-v1` stays a real option where nothing else can write the image,
but here it is the weaker guarantee *and* the weaker result.

### Unmeasured: what this would do on the A10G

**A prediction, not a result.** No A10G was available, so these policies have
never run on slow storage.

Parallel chunking wins only by raising read concurrency, and the A10G volumes are
flat-capped at 0.13 and 0.32 GB/s across 1, 4 and 8 streams
([bandwidth](#both-volumes-are-bandwidth-limited-and-our-loaders-already-saturate-them--2026-09-18)).
Concurrency buys nothing against a throughput cap.

| Host | 1 stream | 8 streams | Serial SHA-256 | Binding cost |
| --- | ---: | ---: | ---: | --- |
| A10G instance NVMe | 0.32 GB/s | 0.32 GB/s | 1.80 GB/s | the device |
| this H100 host | 0.82 GB/s | 5.46 GB/s | 1.80 GB/s | the reader |

Serial hashing already runs above that cap, so hashing there is device-bound and
the policy should save close to nothing, leaving the snapshot near 30.31 s and
still losing. To falsify: run both policies there and compare payload phases.

### What this does and does not settle

The vLLM hypothesis holds on this host once the activation stops reading the
image single-stream. It failed on the A10G because the volume was slow, and here
at first because our verification was.

Unchanged: the image is still 7.37 GiB against cold's 0.92 GiB, and a faster GPU
shrinks the prize. One model, one host, three campaigns that pool neither with
each other nor with the A10G pair. Training keeps `strict-v1`.
