# Four-update LoRA snapshot experiment

Use the EC2 host's GPU access and `sudo`; the restricted tool sandbox has no GPU.
Keep outputs in ignored `runs/`, using a fresh directory for every preparation
and trial. [Measured results](../results.md#four-update-lora-acceptance-and-timing--2026-09-14).

The trainer updates the model; the controller manages capture/restore. These
separate **processes** have their own memory. Read the
[CPU chapter](../../docs/01-cpu-checkpointing.md) for background.
[Independent capture/restore commands](../../docs/independent-lifecycle-plan.md)
remain planned; this page describes the implemented controller.

Create the isolated Python environment and prepare the pinned model and tokens:

```bash
python3 -m venv runs/finetuning/venv
runs/finetuning/venv/bin/pip install -r experiments/finetuning/requirements.lock
runs/finetuning/venv/bin/python experiments/finetuning/prepare.py runs/finetuning/assets
```

Add `--model-dir runs/tools/qwen2.5-0.5b` to reuse the preliminary probe's model.
Preparation records its revision, file hashes, tokens, and environment.

The default `--tools runs/tools` expects CRIU 4.2.1 at `criu-4.2.1/criu/criu`, its
`plugins/cuda/cuda_plugin.so`, libraries in `criu-deps/usr/lib/x86_64-linux-gnu`,
and `cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint`. To build an isolated copy
on Ubuntu 22.04 x86_64 with Git, GCC, Make, and pkg-config:

```bash
bash experiments/criu/build.sh runs/new-tools
```

Pass `--tools runs/new-tools` to every run. Dependencies are extracted locally,
without system installation. Our controller invokes CRIU's CLI; its CUDA plugin
owns NVIDIA transitions. The alternative `pycriu` RPC interface is unused.

Create a matching reference pair, then compare both restore routes:

```bash
runs/finetuning/venv/bin/python experiments/finetuning/run.py reference runs/finetuning/reference-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py criu runs/finetuning/criu-zero \
  --reference runs/finetuning/reference-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py application runs/finetuning/application-zero \
  --reference runs/finetuning/reference-zero
```

The default is four updates, capturing after update 2. Other checks use:

| Check | Arguments |
| --- | --- |
| Active CUDA randomness | `--dropout 0.1` on reference and restore commands |
| Early one-update capture | `--until 2` on reference and CRIU; `--capture 1` on CRIU |
| Repeated CRIU restoration | `--capture 2,3` on CRIU, with the matching four-update reference |

References require matching environment, assets, configuration, tools, and source
hashes; even comment edits require fresh references. Application restart reloads
adapters, buffers, Adam, schedule, cursor, and training settings, restoring random
state last. CRIU receives only process images.

## Read one pipeline from start to finish

[run.py](run.py) checks prerequisites and selects a dedicated pipeline:

| File | Responsibility |
| --- | --- |
| [reference_pipeline.py](reference_pipeline.py) | Run two independent trainers; `run.py` compares them before publishing `verified`. |
| [application_pipeline.py](application_pipeline.py) | Request a save, verify exit, run job B, launch a new trainer, and inspect loaded state. |
| [criu_pipeline.py](criu_pipeline.py) | Capture, verify exit, run job B, restore, and inspect each requested generation. |
| [pipeline.py](pipeline.py) | Shared capture, handoff, comparison, and exit checks; `Trial` tracks the current process for cleanup. |
| [session.py](../criu/session.py) | Linux process operations and CRIU commands. |
| [train.py](train.py), [state.py](state.py) | Training loop, observed state, and application save/load. |
| [job_b.py](job_b.py) | Separate 4 MiB GPU allocation and checked sum. |

Empty marker files announce readiness or permission in a diagnostic CRIU run:

```mermaid
sequenceDiagram
    participant C as Controller
    participant T as Trainer
    participant R as CRIU + CUDA plugin
    T-->>C: ready-2: complete update and state-2.json
    C->>R: dump after comparing reference state
    Note over R,T: Capture process/GPU state; original exits
    Note over C: Verify exit, GPU availability, sync; run job B
    C->>R: restore process image
    C-->>T: inspect-2: observe before changing state
    T-->>C: after-2.json + inspected-2
    C-->>T: continue-2 only if reference / before / after match
    T-->>C: Updates 3–4 state and losses
    C->>C: Compare continuation with reference
```

These arrows show control and observations; the
[GPU chapter](../../docs/02-gpu-checkpointing.md) explains RAM/VRAM transfers.
Generations use separate images, PID files, and markers. Exit verification
includes **reaping**: collecting the child's exit status to remove its remaining
Linux record. Results distinguish numerical, lifecycle, and compatibility verdicts.

## Measure after correctness passes

Timing requires matching full correctness evidence:

```bash
runs/finetuning/venv/bin/python experiments/finetuning/run.py application runs/finetuning/timing-application-1 \
  --reference runs/finetuning/reference-zero --timing --validated-run runs/finetuning/application-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py criu runs/finetuning/timing-criu-1 \
  --reference runs/finetuning/reference-zero --timing --validated-run runs/finetuning/criu-zero
```

Repeat with fresh `-2` and `-3` directories on an otherwise idle GPU, keeping
dropout, capture boundary, assets, and cache conditions unchanged. Timing omits
large state hashes and inspection waits from the restore path. The controller
checks assets before launch; all losses and final state are checked after the
measured next update.

Measure capture request → filesystem sync and restore request → next update.
GPU observation follows verified exit; job B is timed separately. RAM sampling
is every 100 ms; CUDA peaks come from the trainer. Linux control-group (cgroup) memory includes other
processes/cache. Trial duration excludes controller preflight.

CRIU can restore a different clock offset for the trainer. [metrics.py](metrics.py)
converts its update timestamp using the exact restore-log offset before comparing
it with the controller; negative latencies are rejected. Raw timestamps and
offsets remain available. See the measured
[clock correction](../results.md#clock-correction-and-provenance).

Curate matching trials without publishing memory images or raw logs:

```bash
python3 experiments/finetuning/report.py runs/finetuning/reference-zero \
  runs/finetuning/application-zero runs/finetuning/criu-zero \
  runs/finetuning/timing-application-{1,2,3} runs/finetuning/timing-criu-{1,2,3} \
  --output runs/finetuning/report.json
```

The JSON retains individual runs, medians, and ranges. Older runs lacking a
controller offset need `--controller-clock-offset-ns` from measured evidence
(zero for September 14). Reanalysis preserves old calculations and records
analysis-source hashes separately from workload-source hashes.

Run CPU-side contract checks with:

```bash
PYTHONPATH=experiments/cpu:experiments/finetuning:experiments/criu \
  runs/finetuning/venv/bin/python -m unittest discover -s tests -v
```

This establishes same-host continuation. `sync -f` flushes the local filesystem;
it does not establish host-loss, replacement-host, or spot recovery. Shared-memory
ownership and retained CRIU warnings qualify compatibility even when state matches.
