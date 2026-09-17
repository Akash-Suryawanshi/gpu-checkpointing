# Four-update LoRA snapshot experiment

Use the EC2 host's GPU access and `sudo`; the restricted tool sandbox has no GPU.
Use fresh private output directories on persistent storage with sufficient space;
keep large images outside the checkout when its volume is nearly full. [Measured results](../results.md#independent-lifecycle-validation--2026-09-17).

The trainer updates the model; independent capture and restore workers manage
its snapshots. These separate **processes** have their own memory. Read the
[CPU chapter](../../docs/01-cpu-checkpointing.md) for background.
The [lifecycle contract](../../docs/independent-lifecycle-details.md) defines
publication and ownership. Recovery does not require the experiment harness.

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
without system installation. Workers invoke CRIU's CLI; its CUDA plugin
owns NVIDIA transitions. The alternative `pycriu` RPC interface is unused.

## Independent commands

Run these from the repository root in separate shells. Use a new absolute `JOB`
path and `SNAPSHOT` path on persistent storage; keep assets and external job files
available. The foreground launch shell collects the original trainer's exit.

```bash
PYTHON="$PWD/runs/finetuning/venv/bin/python"
TOOLS="$PWD/runs/tools"
ASSETS="$PWD/runs/finetuning/assets"
JOB="$PWD/runs/finetuning/manual-job"
SNAPSHOT="$PWD/runs/finetuning/manual-snapshot"
mkdir -m 700 "$JOB"
```

Repeat the variable assignments in a second shell and start capture there first;
it waits for trainer registration. Then launch the trainer in the first shell:

```bash
env -i PATH="$PATH" HOME="$HOME" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 "$PYTHON" experiments/finetuning/train.py \
  --assets "$ASSETS" --run-dir "$JOB" --external-control --until 4 \
  < /dev/null > "$JOB/updates.jsonl" 2> "$JOB/trainer.stderr"
```

The second shell uses the command below. `--at 1` requests the earliest boundary;
the acknowledgement records the actual update. A request after completion fails within `--timeout` (300 s by
default); the automated harness below avoids human timing races.

```bash
"$PYTHON" experiments/finetuning/checkpoint.py \
  --run-dir "$JOB" --snapshot "$SNAPSHOT" --tools "$TOOLS" --at 1
# Capture has exited; run job B separately before independent restoration.
"$PYTHON" experiments/finetuning/job_b.py
"$PYTHON" experiments/finetuning/restore.py --snapshot "$SNAPSHOT" --tools "$TOOLS"
```

Restore checks publication, content hashes, source/environment/assets, external
logs, process identity, and control state before CRIU runs. It compares untouched
state before release, then exits while the trainer continues under the host or
ancestor reaper (the process that collects an exited child's status).

Use only capture A → restore A → capture B → restore B. Old, consumed, partial,
or ambiguous snapshots are rejected; there is no automatic fallback/revalidation
command. After dump may have started, failure never automatically resumes an
unknown CPU/CUDA state; the original may already be gone.

## Automated comparisons

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
| [criu_pipeline.py](criu_pipeline.py) | Launch independent workers, reap owned trainers, run job B, and compare continuation. |
| [checkpoint.py](checkpoint.py), [restore.py](restore.py) | Independent capture/publication and admission/restore/inspection/release. |
| [control.py](control.py) | Job identity, external pause requests, operation lock, and durable records. |
| [pipeline.py](pipeline.py) | Shared capture, handoff, comparison, and exit checks; `Trial` tracks the current process for cleanup. |
| [session.py](../criu/session.py) | Linux process operations and CRIU commands. |
| [train.py](train.py), [state.py](state.py) | Training loop, observed state, and application save/load. |
| [job_b.py](job_b.py) | Separate 4 MiB GPU allocation and checked sum. |

Independent workers use identity-bearing JSON records, not shared Python objects:

```mermaid
sequenceDiagram
    participant H as Optional experiment harness
    participant T as Trainer
    participant C as Capture worker
    participant R as Restore worker
    C-->>T: Pause request with job/capture identity
    T-->>C: Completed-update acknowledgement and before evidence
    C->>C: CRIU dump; observe original removed and GPU empty
    Note over H: Launch parent reaps original; capture cannot reap it
    C->>C: Hash, sync payload, publish COMPLETE + phase; exit
    H->>H: Run independent job B
    R->>R: Validate artifacts; CRIU restore
    R-->>T: Inspect this restore attempt
    T-->>R: Untouched state evidence
    R-->>T: Continue only after equality; worker exits
    T-->>H: Subsequent states and losses match reference
```

Arrows between processes carry control requests and observations. The
[GPU chapter](../../docs/02-gpu-checkpointing.md) explains memory transfers.
Mutable attempts/logs are separate from immutable image payload; every restore
has a fresh PID filename. Numerical, lifecycle, and compatibility verdicts remain
separate.

## Measure after correctness passes

Timing requires matching full correctness evidence:

```bash
runs/finetuning/venv/bin/python experiments/finetuning/run.py application runs/finetuning/timing-application-1 \
  --reference runs/finetuning/reference-zero --timing --validated-run runs/finetuning/application-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py criu runs/finetuning/timing-criu-1 \
  --reference runs/finetuning/reference-zero --timing --validated-run runs/finetuning/criu-zero
```

Repeat with fresh `-2` and `-3` directories on an otherwise idle GPU, keeping
dropout, capture boundary, assets, and cache conditions unchanged. Application timing omits large state hashes before the next update, following
matching diagnostics. Independent CRIU always checks untouched state before
release, including timing repetitions; report this inspection cost explicitly.

For independent CRIU, measure capture-worker launch → publication and restore-worker
launch → next update, including artifact validation and state inspection. Application
measurements retain their save-request and reconstruction boundaries.
GPU observation follows verified exit; job B is timed separately. RAM sampling
is every 100 ms; CUDA peaks come from the trainer. Linux control-group (cgroup) memory includes other
processes/cache. Trial duration includes harness admission; outer CLI wall time also includes
interpreter/module startup. Request/ready, dump, hash, sync, publication, inspection,
and the gap between workers have separate events.
The harness queues the target boundary when it launches the trainer, so the
request-to-ready interval includes initialization and updates before that boundary.
Separate this waiting time from ready-to-publication cost; do not present the
whole queued request as capture overhead for an already-running workload.

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

This establishes same-host continuation. Independent capture fsyncs payloads,
directories, manifest, completion, and terminal phase in order; the application
comparison retains its `sync -f` barrier. Neither establishes abrupt-host-loss,
volume-deletion, replacement-host, or spot recovery. Shared-memory
ownership and retained CRIU warnings qualify compatibility even when state matches.
