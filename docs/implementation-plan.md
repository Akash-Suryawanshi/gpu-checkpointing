# Single-GPU Fine-Tuning Checkpoint Implementation Plan

**Implemented, with qualified same-host compatibility.** This document defines the
current experiment's contracts. The proposed [independent lifecycle
plan](independent-lifecycle-plan.md) awaits approval and belongs on a later branch.
Use the [runbook](../experiments/finetuning/README.md) for commands and
[results](../experiments/results.md) for dated measurements.

The experiment ends a LoRA trainer after update 2, restores its CPU/GPU process
image, and compares state and updates 3–4 with uninterrupted training and a complete
application checkpoint. One trainer loop serves all routes; the controller owns
capture and verification. CRIU's CUDA plugin alone owns NVIDIA transitions.

## Implementation milestones

Environment setup, one-update restoration, application restart, active dropout,
repeated restoration, and paired timing all passed on EC2. The September 17
readability refactor passed fourteen CPU checks and nine fresh GPU cases.
[Validation record](../experiments/results.md#readability-refactor-validation--2026-09-17).
Sharing ownership and interrupted-system-call warnings still qualify compatibility.

## Global constraints

- Same host/GPU, matching packages, model/data, precision, seeds, and behavior.
- Process restore uses its images; application saves and diagnostic records are
  never hidden training-state inputs.
- No paged optimizer/managed memory, CUDA IPC, data-loader subprocesses,
  `torch.compile`, additional attention kernels, or remote tracking.
- Keep private run directories, minimal child environments, bounded waits, and
  cleanup restricted to identified experiment processes. Images/raw logs stay in
  ignored `runs/`; commit curated evidence only.
- Keep lifecycle, numerical, and compatibility verdicts separate. Do not suppress
  unsupported-resource warnings or change host security/driver settings to pass.

## 1. Chosen workload

| Setting | Contract |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B`; immutable model/tokenizer revision and file hashes |
| LoRA | Rank 8, alpha 16, `q_proj`/`v_proj`, `bias="none"`; no trainable base modules |
| Precision | FP32; no autocast or gradient scaler |
| AdamW | Trainable parameters only; LR 0.001, betas (0.9, 0.999), epsilon 1e-8, decay 0, `foreach=False`, `fused=False` |
| Schedule | Linear decay over four completed updates; preserve counter and next LR |
| Data | 32 authored prompt/answer pairs; batch 1, at most 128 tokens, fixed order |
| Loss | Answer tokens only; prompt/padding labels −100; reject empty answer labels |
| Execution | Eager attention, `use_cache=False`, no accumulation; clear gradients with `set_to_none=True` |
| Randomness | Seed 2026; separate dropout 0 and 0.1 comparisons |
| Lifecycle | Four updates, capture after two; repeated case also captures after three |

Assert finite losses/gradients, initialized Adam state, and changed adapter weights.
The configured model has 540,672 trainable parameters: about 2.16 MB of FP32
adapters and 4.33 MB of Adam moments. A process image also contains the resident
base/runtime. These settings test restoration, not model quality.
[Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).

### Fast development cycle

Target **2–3 minutes per warm, offline process trial**, including capture/sync,
original exit, job B, restore, inspection, continuation, and cleanup. Exclude
installation, downloads, reference generation, and the full acceptance matrix.
Count optimizer updates, not epochs: two epochs over 32 examples would take 64
updates at batch 1.

Start with capture after update 1 and comparison of update 2, using the same
four-update schedule. Use four updates for routine trials and the full matrix at
milestones. Reuse references only when source, assets, packages, GPU/driver, and
training settings match their manifest; otherwise regenerate both references.

## 2. Files and responsibilities

| Files in `experiments/finetuning/` | Responsibility |
| --- | --- |
| `prepare.py`, configuration/data/lock files | Pin environment/assets and tokenize examples |
| `train.py`, `state.py` | Training, observable state, application save/load |
| `run.py` | Validate inputs, select experiment, write verdict |
| `reference_pipeline.py` | Two independently initialized references |
| `application_pipeline.py` | Save state, then start a fresh trainer |
| `criu_pipeline.py` | Capture and restore the existing process |
| `pipeline.py` | Shared lifecycle steps and process ownership |
| `job_b.py`, `metrics.py`, `report.py` | GPU handoff check, measurement, curated reports |

`experiments/criu/session.py` owns privileged CRIU/process operations; `build.sh`
pins the tools. The current controller launches and collects the trainer's exit
status. Independent workers require the different ownership protocol in the
[proposed contract](independent-lifecycle-details.md#control-and-process-ownership).

### Implementation style

Keep one explicit training loop and a readable execution sequence per pipeline.
Extract substantive shared operations; avoid generic backend interfaces or class
hierarchies. Explain non-obvious blocks with docstrings/grouped comments, linking
the [learning chapters](../README.md) for OS fundamentals. Preserve necessary
synchronization and verification even when reducing code.

## 3. State contract

Record `schema_version`, `update`, `next_example_index`, `example_order`,
`model_identity`, `config`, `adapters`, `optimizer`, `schedule`, `rng`, and
`training_behavior`.

- **Identity/tensors:** pinned assets and hashes, LoRA/data configuration, and
  resident frozen-base verification. Hash tensor dtype, shape, and contiguous
  bytes in stable name order.
- **Optimizer/schedule:** state by parameter name, moments, step counters,
  parameter-group order/hyperparameters, schedule counter/settings, and next LR.
- **RNG:** Python, PyTorch CPU, and actually used CUDA generators. No unused
  NumPy state is claimed.
- **Behavior:** named module modes, active/enabled adapters, trainable flags, and
  dropout probabilities. Call `model.train()` after adapter construction; inspect
  restored values **before any repair**.

### Completed-update boundary (review clarification)

```text
forward/backward → optimizer → scheduler → clear gradients/temporary outputs
 → commit update + next-example cursor → synchronize CUDA → publish ready → wait
```

All gradients must be `None`; readiness permits no next-input work or RNG use.
The current trainer pauses at configured update numbers. External requests that
wait for the next boundary belong to the proposed independent lifecycle change.

Application saves contain changing state and fixed-base identity. Rebuild the
pinned model/optimizer/scheduler, load state, then restore RNG **last**, after setup
that could consume randomness. An adapter export alone is insufficient.
[PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).

Exact equality concerns named state and associated records, not whole image-file
bytes. Follow it with matching real updates. Diagnostics must not consume training
RNG or retain tensor clones inside the image: hash in bounded chunks, release
copies, and record remaining allocation/cache overhead.

## 4. Experiment sequence and gates

### Task 1: Freeze the environment and inspect compatibility

Pin the isolated packages, model/tokenizer/data, CRIU plugin, and NVIDIA helper.
Record live GPU/driver, CUDA/Python versions, host/cgroup RAM limits, and disk
capacity; run jobs offline. Inspect mappings after Python, PyTorch, CUDA, model,
and Adam initialization. See [setup](../experiments/finetuning/README.md) and the
[recorded environment](../experiments/results.md#footprint-environment-and-limits).

**Gate:** real finite adapter updates with measured headroom. Required sharing
must be demonstrated or fixed before unqualified acceptance; unknown ownership
remains qualified, and an unsupported required kernel object blocks the workload.

### Task 1a: First GPU acceptance milestone — one-update LoRA restore

Use the isolated trainer and two matching references. After update 1 initializes
Adam, capture, verify original exit, sync, run job B, and restore from images only.
Compare untouched restored state and update 2. **Gate:** matching continuation;
retain separate compatibility qualifications before expanding the experiment.

### Task 2: Build and verify ordinary training

Run four updates twice with deterministic-algorithm checks and identical CUDA
configuration. Compare initial/boundary/final state and continuation losses.
**Gate:** exact reference agreement, changed adapters, populated Adam moments,
and the correct next example. Diagnose disagreement before comparing restore;
do not choose a tolerance after failure.
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

### Task 3: Implement the complete application-checkpoint comparison

At update 2, write a new candidate containing the complete state contract, flush
and atomically publish it without destroying an older valid save. End/reap the
original, start a fresh trainer with cached assets, and inspect before update 3.
**Gate:** updates 3–4 match the reference after a real save/restart.

### Task 4: Implement full CRIU fine-tuning restoration

```text
Trainer:    ready-2 → wait for inspect-2
Controller: terminating dump → verify exit → observe GPU → sync → run job B
            → restore → publish inspect-2
Trainer:    write after-2 → inspected-2 → wait for continue-2
Controller: compare before/after/reference → publish continue-2 only on pass
Trainer:    updates 3–4
```

Require successful CRIU completion, image inventory, and CUDA plugin evidence
for dump and restore. Use regular-file output, disconnected stdin, explicit tool
paths, and a fresh image directory. Do not use `--leave-running`/`--leave-stopped`
or append manual NVIDIA restore/unlock. File visibility alone is not completion.
The current `sync -f` barrier acknowledges filesystem writeback; it does not prove
survival of volume deletion. The proposed [publication protocol](independent-lifecycle-details.md#exact-local-storage-protocol)
adds independent snapshot completion records.

Collect the original's exit status before restore; CRIU may reuse its numeric
PID. Initial failure deadlines are 300 s for trainer preparation, 120 s per
capture/restore, 30 s for exit, and 60 s for job B—not expected runtimes. On
failure retain evidence, withhold continuation, and clean up only owned processes.

**Gate:** verified original exit, successful job B, image-only restore, untouched
boundary equality, and matching updates 3–4. Warnings retain their own verdict.

### Task 5: Exercise randomness and repeated restoration

Repeat references/application/CRIU with dropout 0.1. Prove the LoRA dropout path
executes in training mode and its used CUDA generator advances during actual
updates, including continuation; unrelated random draws do not count.

Capture again after update 3 in a separate trial. Use fresh image directories,
PID filenames, and readiness/inspection/release markers for each generation;
CRIU refuses to overwrite a PID file. **Gate:** both cycles match through update
4, with clean exit and no surviving experiment GPU work/helpers. Retain older
images until replacements are verified.

### Task 6: Measure and report

Keep detailed correctness hashing/inspection outside headline timings. Measure:

- Capture request → completed filesystem sync, for both methods.
- Restore request → next completed update, including application reconstruction.
- Separate image completion, original exit, and post-exit GPU availability; record
  polling uncertainty. Earlier GPU release may be temporary; job B proves use.
- Full warm diagnostic duration, image bytes, peak RAM/cgroup memory, allocated/
  reserved VRAM, baseline update time, and controller/tool launch costs.

Run three paired repetitions after correctness; publish individual values,
median/range, cache conditions, warnings, and storage limits. Report job B
separately. Convert restored trainer timestamps using CRIU's recorded time-namespace
clock offset before comparing them with the controller; reject negative latency.
**Gate:** reproducible measurements, without assuming lower cost or spot recovery.

## 5. Tests worth keeping

Keep focused CPU checks for actual Adam-state round-trip and the next update,
field-specific corruption detection, blocked continuation on mismatch, generation
isolation, and failed save/capture/exit handling. They complement real GPU runs;
do not add checks for every constant or wrapper. Test docstrings identify their
acceptance, critical-contract, regression, or non-obvious-correctness purpose.

## 6. Scope and review decision

CRIU was selected because the EC2 CPU, tensor, and LoRA gates passed and its
terminating dump fits this experiment. Historical DMTCP results do not establish
an EC2 performance ranking. See [backend evidence](../experiments/results.md#ec2-criu-validation--2026-09-14).

Actual spot termination, replacement-host restore, multi-GPU, QLoRA/paged
optimizers, compression/incremental GPU capture, scheduling, and billing remain
outside this implementation. Later recovery work must independently establish
storage survival, compatible replacement capacity, and the warning deadline.

## Incorporated review decisions

The retained requirements are active dropout/RNG evidence (state contract/Task 5), post-exit
GPU handoff (Tasks 4/6), an early one-update gate (Task 1a), and short four-update
trials. Numerical success does not resolve shared-memory compatibility.
