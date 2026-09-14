# Single-GPU Fine-Tuning Checkpoint Implementation Plan

**Status:** Implementation plan. A preliminary one-update LoRA restore passed on EC2; the complete isolated four-update comparison is not yet implemented or validated.

**Goal:** End a real LoRA training process after update 2, reconstruct it from a CPU/GPU process image, and verify that its state and updates 3–4 match an uninterrupted run. Compare the cost with a complete application checkpoint.

**Architecture:** One explicit training loop supports reference, application restart, and CRIU restore. The trainer owns training state and observable waiting points; an external controller owns process capture, restoration, lifecycle evidence, and comparison. CRIU's pinned CUDA plugin owns NVIDIA transitions.

**Backend status:** Select CRIU after the September 14 EC2 probes passed CPU restore, full GPU tensor restore, and one-update LoRA restore with matching stochastic continuation. The LoRA trial took 63.69 seconds including sync, job B, and diagnostics. Warnings and the preliminary environment remain qualified. [EC2 results](../experiments/results.md#ec2-criu-validation--2026-09-14).

**Tech Stack:** Ubuntu/Linux on EC2 with NVIDIA A10G; Python/PyTorch; Transformers and PEFT; CRIU 4.2.1 at `9539417f3e3cfa4eb84c319cd71f4d52f1f08645` with its CUDA plugin; NVIDIA cuda-checkpoint at `00d5cce84c628088d6caa203fc4af40c1538b6f7`; JSON evidence and local files. Verify live versions at setup; the probe used driver 570.172.08. No distributed trainer or new checkpoint framework.

**References:** [Fine-tuning research](../research/single-gpu-finetuning.md), [measured mechanism results](../experiments/results.md), and the review decisions in this document. Repository baseline: `aecdf2cbd72bb60e7216e31639aaac3dd904d1c6`.

## Implementation milestones

- Environment/assets: isolated virtualenv installed from exact package pins; all 32 authored examples tokenized; a real two-update LoRA check passed. Safetensors 0.6.2 is pinned instead of the host's release candidate. Progressive mapping inspection and the shared-controller restore gate follow in the next milestone.

## Global constraints

- This phase is same-host, same-GPU restoration. Real spot interruption and replacement-host recovery require a later experiment.
- Complete means all state required for this LoRA configuration, not updating all base-model parameters.
- No application checkpoint is an input to the process-restore route. Diagnostic evidence is never a hidden restore source.
- Preserve matching software, model revision, data, seeds, precision, and training behavior across routes.
- No managed-memory/paged optimizer, CUDA IPC, data-loader subprocesses, remote tracking, `torch.compile`, or additional attention kernel packages in the baseline.
- Keep unresolved shared-memory and unsupported-resource diagnostics visible. Numerical success and compatibility acceptance are separate verdicts.
- Use fresh private run directories, minimal child environments, targeted process cleanup, and images/logs under ignored `runs/`.
- Do not change container security policy, host driver, or provider lifecycle as part of this plan.
- Keep implementation code minimal, clean, and understandable. Readability and the required evidence take priority over a line-count target.

## 1. Chosen workload

| Item | Initial setting and reason |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B`; resolve and record an immutable model/tokenizer commit before downloading |
| Adaptation | Ordinary LoRA; `r=8`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj"]`, `bias="none"`, no extra trainable base modules |
| Precision | FP32 weights/computation; no autocast or gradient scaler in this configuration |
| Optimizer | AdamW over trainable parameters only; learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`, `foreach=False`, `fused=False` |
| Schedule | Linear learning-rate decay over 4 completed updates; preserve its counter, settings, and effective next learning rate |
| Data | 32 small project-authored prompt/answer pairs; batch size 1; maximum 128 tokens; fixed recorded order |
| Loss | Causal language-model loss on answer tokens; prompt and padding labels are `-100`; reject examples with no answer labels left after truncation |
| Execution | Eager attention; `use_cache=False`; no gradient accumulation; clear gradients with `set_to_none=True` after each update |
| Randomness | Fixed seed 2026; dropout zero for first comparison; LoRA dropout 0.1 in a required follow-up |
| Lifecycle | 4 updates; capture after 2; a repeated-lifecycle run also captures after 3 |

These are correctness-demo settings, not a model-quality recipe. Assert that adapter weights actually change, optimizer states exist, and losses/gradients remain finite.

The published model has about 0.49 billion parameters: FP32 base weights are roughly 1.96 GB before working memory. For its published dimensions, the specified LoRA targets imply 540,672 trainable parameters: 2.16 MB of FP32 adapter weights plus 4.33 MB for Adam's two moments. Verify actual counts after construction. A process snapshot also includes the resident base and runtime; those 6.49 MB are not its expected size. [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json), [PEFT LoRA configuration](https://huggingface.co/docs/peft/package_reference/lora).

### Fast development cycle

Target **2–3 minutes for one warm, offline process-snapshot trial**, including launch, training, capture/sync, verified original exit, job B, restore, state inspection, continuation, and cleanup. Installation, initial asset download, reference creation, and the full acceptance matrix are outside this target. Measure the first trial before treating the target as achievable: loading, hashing, image I/O, and restore may dominate. Keep the existing 0.5B model, FP32, batch size 1, and 128-token maximum initially.

| Stage | Workload |
| --- | --- |
| First GPU acceptance milestone | Complete update 1 → capture → verify original exit → job B → restore → compare boundary state and update 2 |
| Normal development trial | Complete update 2 → capture → verify original exit → job B → restore → compare boundary state and updates 3–4 |
| Repeated restoration | Capture after updates 2 and 3 using separate generations; finish update 4 |
| Milestone acceptance | Both dropout configurations, application restart comparison, repeated restoration, and timing repetitions |

Count optimizer updates, not epochs: two full epochs over 32 examples at batch size 1 would require 64 updates. Four updates bound the snapshot-continuation test; they do not establish model quality or long-run compatibility. Use the same four-update learning-rate schedule across all routes, including the early probe that stops after update 2.

Reuse a verified uninterrupted reference only when relevant code, model/tokenizer/data hashes, package versions, GPU/driver, seeds, and numerical/training configuration match its manifest. Keep separate references for dropout zero and 0.1. Relevant changes invalidate reuse and require matching uninterrupted runs again. Run one process-restore case per development iteration; run the complete acceptance matrix at milestones. Reference evidence is an external comparison input, never a source for process restoration.

## 2. Files and responsibilities

```text
experiments/finetuning/
  prepare.py             Freeze assets and environment; tokenize the fixture
  train.py               One training loop and the checkpoint waiting points
  state.py               Named state records, fingerprints, application save/load
  run.py                 Run the comparisons and write the combined verdict
  config.json            Explicit workload settings from section 1
  examples.jsonl         The 32 small prompt/answer examples
  requirements.lock      Exact validated Python package versions, generated at setup
experiments/criu/
  session.py             Bounded CRIU dump/restore and process lifecycle operations
tests/
  test_finetuning_state.py
```

Modify `experiments/results.md` with new evidence after execution; add one README link when the experiment is available. Keep the two concept chapters focused on explanation. Downloaded model files, tokenized data, full state evidence, images, and raw logs remain in `runs/`; commit only curated summaries.

Reuse the small lifecycle operations established by the existing CPU checkpoint script and the EC2 CRIU probe. Keep normal trainer launch, explicit interpreter/plugin paths, minimal environments, successful dump completion, verified process exit, and warning collection. Use host privileges only where CRIU requires them. Reap the original before restore: CRIU can reuse the same numeric PID, so compare lifecycle generations rather than requiring different PID numbers. Account for root-owned CRIU output files and reap the detached restored process. Keep the historical DMTCP probe separate. No generic backend interface is needed.

### Implementation style

- Use one explicit training loop and small, plainly named functions for state inspection and process lifecycle. Keep the update and capture/restore sequence readable in execution order.
- Keep modules focused on the responsibilities above. Extract helpers for actual reuse or a clear responsibility; avoid speculative abstractions, class hierarchies, and a generic checkpoint-backend interface.
- Implement the pinned CRIU route and the required application-checkpoint comparison. Add configuration only for the planned experiments; do not build automatic backend selection, a dashboard, or a training framework.
- Use explicit inputs, straightforward control flow, and errors that identify the failed phase or state field. Comments should explain checkpoint invariants and tool-specific behavior.
- Preserve synchronization, bounded waits, image completion, verified exit, restore inspection, and targeted cleanup. These checks are necessary evidence even in a small POC. Review each milestone for unnecessary code before extending it.

## 3. State contract

`state.py` produces a named record with `schema_version`, `update`, `next_example_index`, `example_order`, `model_identity`, `config`, `adapters`, `optimizer`, `schedule`, `rng`, and `training_behavior`. Record optimizer state by stable parameter name, including step counters, moment tensors, parameter-group order, and hyperparameters. Hash tensor dtype, shape, and canonical contiguous bytes in deterministic name order.

- `model_identity`: model/tokenizer revision, downloaded-file hashes, LoRA settings, and data/tokenization hashes. Verify frozen resident base tensors at validation boundaries as well as the source files.
- `rng`: Python random state, PyTorch CPU state, and the CUDA generator states actually used. No unused NumPy state is claimed.
- `schedule`: completed count, schedule settings, and effective next learning rate.
- `training_behavior`: effective named module modes, active/enabled adapters, parameter trainable flags, and dropout probabilities. Explicitly call `model.train()` after adapter construction. Inspect and compare these values after process restore before any calls that could repair them. Application restart must reconstruct and verify them as part of loading.
- At the completed-update boundary, all parameter gradients must be `None`; no partially accumulated gradients or scaler state exist in this FP32 baseline.

### Completed-update boundary (review clarification)

The checkpoint boundary comes **after the whole training update**, before any work for the next example. For update 2, finish forward and backward computation, `optimizer.step()`, `scheduler.step()`, gradient clearing, and temporary-output cleanup; then commit `update = 2` and the next-example cursor. Synchronize CUDA before publishing checkpoint readiness. The saved adapter weights, optimizer history, schedule, random states, and CPU counters must all describe this same boundary.

A pause request arriving during an update only sets a pending request. The trainer completes the update and reaches the cooperative waiting point before the controller captures it. No next-example fetching or random-number consumption is allowed between readiness and capture. If gradient accumulation or mixed precision is introduced later, this boundary must additionally include all accumulated microbatches and scaler bookkeeping; neither feature is enabled in the baseline.

```text
Forward → backward → optimizer → scheduler → clear gradients / temporaries
        → commit completed count and next-example cursor → synchronize GPU
        → publish ready and wait → capture CPU/GPU process
```

An application save contains the changing state and fixed-base identity, not another copy of frozen base weights. Loading constructs the pinned model, adapters, optimizer, and scheduler, then loads saved training state and restores RNG states **last**, after setup that could consume random numbers. Verify state before update 3. An adapter export by itself is insufficient. [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).

Byte equality means named training-state tensor dtype, shape, and bytes plus associated counters, RNG, and behavior records, not identical whole process-image files. Matching boundary bytes must be followed by matching actual optimizer updates.

Diagnostic collection must not consume training RNG or retain GPU/CPU tensor clones inside the process being captured. Hash tensors in bounded chunks, release temporary copies, and retain only small summaries. Full diagnostic records live outside the process image and are read by the verifier, never by process restoration. Record the observer's remaining allocation/cache footprint; do not assume deleting a Python reference returns memory to the OS.

## 4. Experiment sequence and gates

### Task 1: Freeze the environment and inspect compatibility

**Files:** `prepare.py`, `config.json`, `examples.jsonl`, `requirements.lock`; experiment manifest under `runs/`.

- [ ] Recheck live GPU, driver, Python, PyTorch/CUDA, CRIU SHA/plugin, NVIDIA helper version, cgroup RAM limits, available disk, and process mappings. The EC2 probe observed A10G 23,028 MiB and about 30 GiB host RAM with no cgroup memory cap; recheck instead of carrying over either these values or the earlier L4 host values.
- [ ] Create an isolated training environment. Start from the EC2 probe combination of PyTorch 2.8.0+cu128, Transformers 5.0.0, and PEFT 0.18.1; validate imports and an actual forward/backward/update, then lock exact packages before comparative runs. Do not upgrade the working server environment in place.
- [ ] Download the resolved model/tokenizer revision once, prepare and hash local tokens, and run measured jobs offline against those same assets. Record the resolved package/model versions in the manifest rather than using moving `main` revisions at runtime.
- [ ] Inspect mappings in progressively initialized processes: Python, PyTorch import, CUDA initialization, loaded model, and initialized Adam state. Relate `/dev/zero (deleted)` warnings to the stage that introduces them and inspect aliases/backing identities.
- [ ] If sharing is required, demonstrate preservation with a focused reproducer or fix the configuration/tool support and revalidate. An absence of visible aliases in one process does not prove the driver needs none. Unknown ownership remains an explicit qualification; do not suppress the warning. Record an unsupported kernel-object mapping as a blocker for that workload.

**Gate:** A pinned environment performs real adapter updates with adequate measured memory headroom. Compatibility warnings have a documented status. Qualified diagnostic work can proceed, but unqualified acceptance cannot pass while required sharing remains unresolved.

### Task 1a: First GPU acceptance milestone — one-update LoRA restore

**Preliminary evidence:** This gate passed with the host packages and two authored examples. Repeat it in the isolated locked environment and shared implementation below; do not mark the full plan complete from the exploratory probe.

**Files:** Minimal `train.py`, `state.py`, `run.py`, and shared `experiments/criu/session.py` operations needed for this probe.

- [ ] Build the real LoRA update, state inspection, and bounded lifecycle needed for this gate before the complete comparison/reporting machinery. Use the pinned isolated environment, shared training-update function, and four-update schedule.
- [ ] Establish matching two-update uninterrupted references. In the process route, complete update 1 with initialized Adam moments/counters, cleared gradients, and CUDA synchronization. Capture using Task 4's successful-dump and generation-specific handshake rules; reap and verify original exit, sync, and run job B.
- [ ] Restore from the process image alone. Compare the full state contract before changing restored settings or allowing update 2; compare update 2 loss and state against the reference. Retain mapping diagnostics and separate lifecycle, numerical, and compatibility verdicts.

**Gate:** The real LoRA process survives original exit and performs the matching next optimizer update. Investigate failure before building the complete comparison. Numerical success permits qualified diagnostic work; unresolved required sharing still prevents unqualified compatibility acceptance.

### Task 2: Build and verify ordinary training

**Files:** `train.py`, `state.py`, `prepare.py`.

- [ ] Implement the exact setup and state contract above. Put all three execution modes through the same training-update function.
- [ ] Run 4 updates twice without interruption. Compare initial state, update-2 state, losses and state for updates 3–4, and final state.
- [ ] Enable deterministic algorithm checks, use the same numerical settings for all routes, and record any required CUDA determinism configuration in every child environment. A fixed seed alone is insufficient. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
- [ ] If references disagree, identify the operation or configuration before interpreting any restore difference. Do not choose a tolerance after seeing a failure; the first baseline targets exact equality.

**Gate:** Two ordinary runs match, adapters changed, Adam moments were populated, and the data cursor identifies the next example consistently.

### Task 3: Implement the complete application-checkpoint comparison

**Files:** `state.py`, `train.py`, `run.py`.

- [ ] At update 2, complete the optimizer and scheduler, clear gradients, release temporary outputs, synchronize CUDA, and record state.
- [ ] Serialize adapters plus the full state contract to a new candidate, flush it, and atomically publish the completed checkpoint on the selected filesystem. Preserve the previous completed save if a later save fails.
- [ ] End the original process and verify its real PID is gone. Start a fresh process with the same local assets and explicit interpreter; load state, compare it before update 3, then continue to 4.
- [ ] Compare every continuation update with the reference. Use cached fixed-base assets for both application restart and process restore.

**Gate:** Correct continuation from a real application save/restart. This is the baseline a process snapshot must be compared with.

### Task 4: Implement full CRIU fine-tuning restoration

**Files:** `experiments/criu/session.py`, `train.py`, `run.py`.

- [ ] Put the tested CRIU lifecycle operations in `session.py`. Launch the fine-tuning interpreter normally, with regular-file output and disconnected stdin, then capture its PID using pinned CRIU with the explicit CUDA plugin path. Keep privileged CRIU invocation separate from the trainer.
- [ ] Use the following generation-specific handshake for capture at update 2. A future update-3 capture uses separate `3` markers so earlier release files cannot bypass its wait.

```text
Trainer: complete update 2 → synchronize → write before-2 evidence
         → publish ready-2 → wait for inspect-2
Controller: default CRIU dump → require successful completion
            → reap original / verify PID absent → sync filesystem
            → observe GPU availability → run and finish job B
            → CRIU restore → record restored PID → publish inspect-2
Restored trainer: wait for CUDA restoration → write after-2 evidence
                  → publish inspected-2 → wait for continue-2
Controller: compare before/after/reference → publish continue-2 only on pass
Trainer: perform update 3, then continue through 4
```

- [ ] Let the CRIU CUDA plugin own lock/checkpoint/restore/unlock. Require evidence that it actually ran in dump and restore. Its restore hook coordinates the NVIDIA helper thread before application continuation; keep the trainer synchronization and inspection gate. Do not append manual NVIDIA restore/unlock commands.
- [ ] Use a fresh image directory and default terminating `criu dump`, without `--leave-running` or `--leave-stopped`. Require successful command completion and image inventory before accepting capture; file appearance alone is insufficient. Verify original exit and filesystem sync separately. Reject failed or partial dumps. Do not reuse DMTCP final-name completion logic.
- [ ] Give each phase a bounded deadline. Initial defaults: 300 seconds to prepare the trainer, 120 seconds for capture or restore, 30 seconds for exit, and 60 seconds for job B. These are failure deadlines, not expected durations or a guarantee of the 2–3-minute target. On timeout or mismatch, retain evidence, reject the run, and stop only identified experiment processes and helpers. Never release the next training update after failed verification.
- [ ] Persist lifecycle, numerical, and compatibility verdicts separately even when the final acceptance is false. Do not lose successful tensor/state evidence merely because the later warning gate fails.

**Gate:** Original process ended; GPU was available to job B; the process image alone reconstructed the trainer; state matched before update 3; updates 3–4 matched the reference. Remaining warnings keep the result qualified.

### Task 5: Exercise randomness and repeated restoration

- [ ] Repeat the reference/application/process comparison with LoRA dropout 0.1. Establish matching uninterrupted references for this configuration first. Require evidence that the enabled LoRA dropout path executes in training mode and that the used CUDA generator state changes across an actual update, including restored continuation. RNG advancement from an unrelated operation is insufficient. Compare effective behavior and RNG states before any post-restore modification; do not add diagnostic random draws.
- [ ] Run a separate process-restoration case with captures after updates 2 and 3. Use unique checkpoint directories and handshake markers, verify original exit each time, and compare through update 4.
- [ ] Require clean process exit and no remaining experiment GPU work or helper process after the lifecycle tests. Retain old completed images until their replacements are verified.

**Gate:** Active dropout and CUDA RNG consumption are demonstrated, stochastic continuation matches, and a second checkpoint cycle passes. State clearly which configurations passed and whether compatibility qualifications remain.

### Task 6: Measure and report

**Files:** `run.py`, `experiments/results.md`, curated evidence JSON.

- [ ] Separate correctness runs from timing runs. Detailed hashes and inspection waits are excluded from headline latency; use a timing variant with the same model/data/boundary and smaller evidence after the measured endpoint. Measure each variant's memory footprint.
- [ ] For both save methods, time from capture request to completed filesystem sync and from restore request to the next completed update. Include model/runtime rebuilding for application restart, including necessary CPU-to-GPU copies.
- [ ] Record separate monotonic timestamps for capture request, successful dump completion, verified original-process exit, completed filesystem sync, and observed GPU availability after exit. Report request-to-post-exit GPU availability with polling uncertainty. Any earlier disappearance is a staging event, not completed handoff. The default CRIU dump terminates the original; still require verified exit and a subsequent GPU observation. Retain successful job B as functional evidence. Include interpreter and checkpoint-tool launch in end-to-end costs where required; record job B separately rather than charging its runtime as snapshot overhead.
- [ ] Report full warm development-trial wall time against the 2–3-minute target separately from headline snapshot latencies. Include correctness inspection and job B in this trial duration; identify loading, training, hashing, capture/sync, and restore costs before changing the workload to improve turnaround.
- [ ] Record image/file size, peak process RAM, cgroup memory, allocated/reserved VRAM, baseline update time, and controller/tool launch overhead. Do three paired timing repetitions after correctness, report individual values plus median/range, and state cache conditions and local-storage durability limits.
- [ ] Publish a short state-continuity table, lifecycle timeline, cost comparison, pinned environment, and any unresolved blockers. No raw process environment or process image is committed.

**Gate:** The report supports a narrow, reproducible conclusion; neither lower cost nor real spot recovery is assumed.

## 5. Tests worth keeping

Use the repository's existing `unittest` approach for the new state/lifecycle contracts:

1. **Acceptance:** a local small tensor model with actual Adam state round-trips the application record, then performs the same next update; compare RNG, cursor, schedule, and moments as well as weights.
2. **Non-obvious correctness:** a corruption matrix changes an optimizer moment, cursor, RNG, or schedule while leaving weights intact; each discrepancy is identified by field name.
3. **Critical contract:** the controller cannot publish continuation before post-restore evidence matches, and capture generations cannot reuse old release markers. Include failed image completion/original-exit paths when extracting the session code.

These tests complement, rather than replace, the real EC2 GPU acceptance runs. Retain all three existing CPU tests. Do not create a test for each constant, JSON key, or trivial wrapper.

## 6. Scope and review decision

Use **CRIU with its native CUDA plugin** on this EC2 host. Host privilege probes, CPU restoration, full GPU tensor restoration, and preliminary LoRA continuation passed. Its normal-process launch and default terminating dump fit the bounded handoff experiment. The earlier container restriction is historical; the tool sandbox's restrictions must also be distinguished from the host execution context. [EC2 evidence](../experiments/results.md#ec2-criu-validation--2026-09-14).

DMTCP remains a separate historical route with unresolved shared-memory warnings. It was not benchmarked on EC2, so backend selection is based on demonstrated suitability and the required lifecycle, not a measured speed ranking. The CRIU probe preserved the observed shared mapping flags/ranges but emitted interrupted-system-call warnings; retain those qualifications and do not infer untested external sharing guarantees. Implement one process-checkpoint backend and the required application-checkpoint comparison.

Out of this implementation: actual spot termination, replacement-host restore, multi-GPU, QLoRA/paged optimizers, compression/incremental GPU capture, a scheduler, and billing integration. A later spot-recovery phase must independently prove image/assets survival on durable storage, compatible replacement capacity, and the warning deadline.

Review the workload, state contract, waiting sequence, and acceptance gates before implementation. After agreement, implement and review task by task; synchronize tested milestones through the existing GitHub/SSH workflow.

## Incorporated review decisions

- Training behavior is part of the state contract; Task 5 requires active dropout and actual CUDA RNG consumption, inspected before any restore repair.
- Task 6 distinguishes temporary capture-time GPU release from verified post-exit availability and retains job B as functional evidence.
- Task 1a makes one-update LoRA restoration the first GPU acceptance milestone. Shared-memory warnings remain a separate compatibility qualification.
- The default is four optimizer updates with capture after two; the repeated case captures after two and three. The 2–3-minute target applies to a warm development trial, not the complete acceptance matrix or a promised runtime.
