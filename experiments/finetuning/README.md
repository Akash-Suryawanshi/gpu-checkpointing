# Four-update LoRA snapshot experiment

Use the EC2 host with GPU access and working `sudo`, which lets this experiment
invoke CRIU with Linux administrator privileges. The tool sandbox is a restricted
execution environment inside the host; it does not expose the GPU. Keep all
assets, environments, and run output under ignored `runs/`.

The **trainer** and **controller** are separate processes: running programs with
their own memory. The trainer performs the updates; the controller manages when
it may continue and asks CRIU to capture or reconstruct it. If these concepts are
new, start with the [machine overview](../../README.md#start-with-the-machine)
and the [CPU chapter](../../docs/01-cpu-checkpointing.md).

Create an isolated environment, then prepare the pinned local model and tokens:

```bash
python3 -m venv runs/finetuning/venv
runs/finetuning/venv/bin/pip install -r experiments/finetuning/requirements.lock
runs/finetuning/venv/bin/python experiments/finetuning/prepare.py runs/finetuning/assets
```

To reuse the model from the preliminary probe, add
`--model-dir runs/tools/qwen2.5-0.5b`. Preparation refuses to overwrite existing
output. It records the model revision, asset hashes, tokens, and environment.

Create two matching references, then run one process-image trial:

```bash
runs/finetuning/venv/bin/python experiments/finetuning/run.py reference runs/finetuning/reference-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py criu runs/finetuning/criu-zero \
  --reference runs/finetuning/reference-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py application runs/finetuning/application-zero \
  --reference runs/finetuning/reference-zero
```

Each run directory must be new. Add `--dropout 0.1` to the reference, CRIU, and
application commands to compare CUDA randomness. The early gate uses `--until 2`
for the reference and CRIU commands, plus `--capture 1` for the CRIU trial.
The default is four updates, with capture after two.
Use `--capture 2,3` for the repeated CRIU lifecycle. Images, PID files, and
handshake markers all use separate generation numbers.

The application route saves adapters, buffers, Adam, schedule, data position, RNG,
and training behavior. It flushes and atomically publishes the save, verifies the
original process has exited, runs job B, then builds a fresh trainer from the pinned
base model and the save. It restores RNG last and compares state before update 3.
The process-image route restores solely from CRIU images.

`run.py` uses the isolated interpreter that launched it. `session.py` invokes CRIU's
CLI and handles waiting, exit verification, and cleanup; it does not implement
checkpointing. CRIU also offers `pycriu` RPC bindings, but the CLI keeps the single
privileged call separate from the trainer without introducing a worker/service
interface. NVIDIA transitions belong entirely to CRIU's CUDA plugin.

The default `--tools runs/tools` expects `criu-4.2.1/criu/criu`, its
`plugins/cuda/cuda_plugin.so`, libraries under `criu-deps/usr/lib/x86_64-linux-gnu`,
and `cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint`. This matches the isolated
EC2 tool build recorded in the project results. No host Python packages are used.
To reproduce the tool build on Ubuntu 22.04 x86_64 with Git, GCC, Make, and
pkg-config available, run `bash experiments/criu/build.sh runs/new-tools`, then
pass `--tools runs/new-tools`. The script extracts downloaded development packages
locally and installs no system packages.

Results contain separate lifecycle/numerical verdicts and compatibility warnings.
References are reused only when their environment, assets, settings, and code
fingerprints match. Full state evidence is read only by the verifier. The restored
trainer waits until the controller compares it before allowing another update.
Even a comments-only source edit changes the source fingerprint, so create a new
reference pair after editing code. Old results remain historical evidence.

## Read one pipeline from start to finish

The command-line interface stays in [run.py](run.py). It checks that assets,
configuration, tools, and reference evidence agree, then selects one file:

| File | Follow this sequence |
| --- | --- |
| [reference_pipeline.py](reference_pipeline.py) | Launch a trainer, collect its exit, then repeat in a fresh process. `run.py` compares both runs before marking the reference usable. |
| [application_pipeline.py](application_pipeline.py) | Request an explicit training-state save, verify exit, run job B, launch a fresh trainer, and inspect its loaded state. |
| [criu_pipeline.py](criu_pipeline.py) | Capture the waiting trainer, verify exit, run job B, restore its process image, and inspect it before continuing. Repeat for each requested capture. |

Shared phases live in [pipeline.py](pipeline.py). Its `Trial` record holds the
current process identity so memory sampling and failure cleanup can find it even
when a pipeline raises an exception. It contains no model or optimizer.
[session.py](../criu/session.py) handles Linux process operations and the CRIU
command; [job_b.py](job_b.py) is the separate 4 MiB GPU allocation and sum.

For a diagnostic CRIU run, follow these named operations:

1. `train.main()` finishes update 2, clears gradients, synchronizes CUDA, writes
   `state-2.json`, and creates `ready-2`. A **marker** is a file whose existence
   communicates an event between processes; it contains no model state.
2. `pipeline.capture_boundary()` waits for that marker and compares the saved
   observation with the reference. `session.criu("dump", ...)` then captures the
   process into `images-2/` through the CUDA plugin.
3. `pipeline.handoff()` uses `session.reap()` to collect the original process's
   exit status, checks GPU availability, syncs the filesystem, and runs job B.
   **Reaping** removes the exited child's remaining Linux process record; it is
   distinct from merely noticing that training stopped.
4. `session.criu("restore", ...)` reconstructs the process. Creating `inspect-2`
   releases its saved wait, so `train.inspect_restored()` can observe state before
   any further training. It writes `after-2.json` and `inspected-2`, then waits.
5. `session.release_verified()` compares reference, pre-capture, and restored
   observations. Only a match permits `continue-2`; the trainer then performs
   updates 3–4. `pipeline.compare_run()` compares their state and losses too.

```mermaid
sequenceDiagram
    participant C as Controller
    participant T as Trainer
    participant R as CRIU + CUDA plugin
    T-->>C: ready-2 marker: update 2 is complete
    C->>R: dump request
    Note over R,T: Capture process/GPU state; original exits
    Note over C: Verify exit, GPU availability, sync; run job B
    C->>R: restore request
    Note over R,T: Reconstruct saved trainer
    C-->>T: inspect-2 marker: observe restored state
    T-->>C: after-2 evidence + inspected-2 marker
    C-->>T: continue-2 marker, only after comparison passes
    T->>T: Complete updates 3–4
```

The arrows above are control requests and observations, not RAM/VRAM transfers.
The [GPU chapter](../../docs/02-gpu-checkpointing.md) explains that data movement.
The application route instead uses `state.save_application()` and
`state.load_application()` to reconstruct explicitly saved values in a new
trainer. The CRIU route never supplies `--load` or an application checkpoint.

## Measure after correctness passes

After the full correctness run passes, measure a separate trial:

```bash
runs/finetuning/venv/bin/python experiments/finetuning/run.py application runs/finetuning/timing-application-1 \
  --reference runs/finetuning/reference-zero --timing --validated-run runs/finetuning/application-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py criu runs/finetuning/timing-criu-1 \
  --reference runs/finetuning/reference-zero --timing --validated-run runs/finetuning/criu-zero
```

Repeat the pair with fresh `-2` and `-3` directories, keeping dropout, capture
boundary, assets, and cache conditions the same. Full hashes and inspection waits
are absent from the timing restore path. Asset hashes are checked by the controller
before launch; losses and the full final state are checked after the measured
next-update endpoint. Results identify this narrower verification scope.

The report records capture request to filesystem sync, restore request to the next
completed update, verified original exit, and an immediate GPU observation after
exit. The GPU observation window bounds polling uncertainty; job B remains a
separate functional check. Trainer RAM is sampled every 100 ms, and CUDA allocator
peaks come from the trainer. Cgroup usage includes other processes and file cache.
Tool/trainer launch is included in request latencies; the overall trial timer runs
from trainer launch to verification, excluding controller preflight.

CRIU restores a time namespace, so the trainer's monotonic clock can differ from
the controller's. The measurement code reads the pinned CRIU restore log's exact
namespace offset and converts the completed-update timestamp to the controller's
clock. Within-process update durations need no adjustment. Negative latencies are
rejected. Raw timestamps and namespace offsets remain in the evidence.

Curate the selected matching trials without publishing their images or raw logs:

```bash
python3 experiments/finetuning/report.py runs/finetuning/reference-zero \
  runs/finetuning/application-zero runs/finetuning/criu-zero \
  runs/finetuning/timing-application-{1,2,3} runs/finetuning/timing-criu-{1,2,3} \
  --output runs/finetuning/report.json
```

The JSON retains individual runs and gives timing medians and ranges. Run against
an otherwise idle GPU. Warm page cache is retained; `sync -f` covers this filesystem
and does not establish replacement-host or host-loss recovery.
For older runs without a recorded controller clock offset, reporting requires
`--controller-clock-offset-ns` from measured namespace evidence. The September 14
matrix used zero. Reanalysis retains the superseded calculation and fingerprints
the analysis source separately from the code that executed the workload.

Run the small CPU-side contract tests with:

```bash
PYTHONPATH=experiments/cpu:experiments/finetuning:experiments/criu \
  runs/finetuning/venv/bin/python -m unittest discover -s tests -v
```

This is a same-host correctness experiment; model quality and spot recovery are
outside its acceptance claim. Shared-memory ownership and CRIU warnings remain
explicit qualifications even when numerical continuation passes.
