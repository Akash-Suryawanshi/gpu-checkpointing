# Single-GPU Fine-Tuning Checkpoint Implementation Plan

**Status:** Review draft. The fine-tuning experiment has not been implemented or validated.

**Goal:** End a real LoRA training process after update 20, reconstruct it from a CPU/GPU process image, and verify that its state and updates 21–25 match an uninterrupted run. Compare the cost with a complete application checkpoint.

**Architecture:** One explicit training loop supports reference, application restart, and DMTCP restore. The trainer owns training state and observable waiting points; an external controller owns process capture, restoration, lifecycle evidence, and comparison. DMTCP's pinned CUDA plugin owns NVIDIA transitions.

**Tech Stack:** Linux/NVIDIA L4; Python/PyTorch; Transformers and PEFT; native DMTCP CUDA plugin at `b175bb5ccadd2f02d11cf052f586d2d9ac62ad53`; JSON evidence and local files. No distributed trainer or new checkpoint framework.

**References:** [Fine-tuning research](../research/single-gpu-finetuning.md), [measured mechanism results](../experiments/results.md), and the review decisions in this document. Repository baseline: `aecdf2cbd72bb60e7216e31639aaac3dd904d1c6`.

## Global constraints

- This phase is same-host, same-GPU restoration. Real spot interruption and replacement-host recovery require a later experiment.
- Complete means all state required for this LoRA configuration, not updating all base-model parameters.
- No application checkpoint is an input to the process-restore route. Diagnostic evidence is never a hidden restore source.
- Preserve matching software, model revision, data, seeds, precision, and training behavior across routes.
- No managed-memory/paged optimizer, CUDA IPC, data-loader subprocesses, remote tracking, `torch.compile`, or additional attention kernel packages in the baseline.
- Keep unresolved shared-memory and unsupported-resource diagnostics visible. Numerical success and compatibility acceptance are separate verdicts.
- Use fresh private run directories, minimal child environments, targeted process cleanup, and images/logs under ignored `runs/`.
- Do not change container security policy, host driver, or provider lifecycle as part of this plan.

## 1. Chosen workload

| Item | Initial setting and reason |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B`; resolve and record an immutable model/tokenizer commit before downloading |
| Adaptation | Ordinary LoRA; `r=8`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj"]`, `bias="none"`, no extra trainable base modules |
| Precision | FP32 weights/computation; no autocast or gradient scaler in this configuration |
| Optimizer | AdamW over trainable parameters only; learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`, `foreach=False`, `fused=False` |
| Schedule | Linear learning-rate decay over 25 completed updates; preserve its counter, settings, and effective next learning rate |
| Data | 32 small project-authored prompt/answer pairs; batch size 1; maximum 128 tokens; fixed recorded order |
| Loss | Causal language-model loss on answer tokens; prompt and padding labels are `-100`; reject examples with no answer labels left after truncation |
| Execution | Eager attention; `use_cache=False`; no gradient accumulation; clear gradients with `set_to_none=True` after each update |
| Randomness | Fixed seed 2026; dropout zero for first comparison; LoRA dropout 0.1 in a required follow-up |
| Lifecycle | 25 updates; capture after 20; a repeated-lifecycle run also captures after 22 |

These are correctness-demo settings, not a model-quality recipe. Assert that adapter weights actually change, optimizer states exist, and losses/gradients remain finite.

The published model has about 0.49 billion parameters: FP32 base weights are roughly 1.96 GB before working memory. For its published dimensions, the specified LoRA targets imply 540,672 trainable parameters: 2.16 MB of FP32 adapter weights plus 4.33 MB for Adam's two moments. Verify actual counts after construction. A process snapshot also includes the resident base and runtime; those 6.49 MB are not its expected size. [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json), [PEFT LoRA configuration](https://huggingface.co/docs/peft/package_reference/lora).

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
experiments/dmtcp/
  session.py             Shared launch/checkpoint/restart/lifecycle operations
  probe-dmtcp.py         Existing tensor/CPU probe, using those shared operations
tests/
  test_finetuning_state.py
```

Modify `experiments/results.md` with new evidence after execution; add one README link when the experiment is available. Keep the two concept chapters focused on explanation. Downloaded model files, tokenized data, full state evidence, images, and raw logs remain in `runs/`; commit only curated summaries.

Extract session operations from the existing controller rather than copying its process lifecycle into a second controller. Preserve its fresh coordinator ports, real-PID checks, minimal environment, final-image observation, and warning detection. The shared module must accept the exact Python executable, plugin path, target arguments, and explicit environment so it cannot silently fall back to the server's system Python.

## 3. State contract

`state.py` produces a named record with `schema_version`, `update`, `next_example_index`, `example_order`, `model_identity`, `config`, `adapters`, `optimizer`, `schedule`, and `rng`. Record optimizer state by stable parameter name, including step counters, moment tensors, parameter-group order, and hyperparameters. Hash tensor dtype, shape, and canonical contiguous bytes in deterministic name order.

- `model_identity`: model/tokenizer revision, downloaded-file hashes, LoRA settings, and data/tokenization hashes. Verify frozen resident base tensors at validation boundaries as well as the source files.
- `rng`: Python random state, PyTorch CPU state, and the CUDA generator states actually used. No unused NumPy state is claimed.
- `schedule`: completed count, schedule settings, and effective next learning rate.
- At the completed-update boundary, all parameter gradients must be `None`; no partially accumulated gradients or scaler state exist in this FP32 baseline.

### Completed-update boundary (review clarification)

The checkpoint boundary comes **after the whole training update**, before any work for the next example. For update 20, finish forward and backward computation, `optimizer.step()`, `scheduler.step()`, gradient clearing, and temporary-output cleanup; then commit `update = 20` and the next-example cursor. Synchronize CUDA before publishing checkpoint readiness. The saved adapter weights, optimizer history, schedule, random states, and CPU counters must all describe this same boundary.

A pause request arriving during an update only sets a pending request. The trainer completes the update and reaches the cooperative waiting point before the controller captures it. No next-example fetching or random-number consumption is allowed between readiness and capture. If gradient accumulation or mixed precision is introduced later, this boundary must additionally include all accumulated microbatches and scaler bookkeeping; neither feature is enabled in the baseline.

```text
Forward → backward → optimizer → scheduler → clear gradients / temporaries
        → commit completed count and next-example cursor → synchronize GPU
        → publish ready and wait → capture CPU/GPU process
```

An application save contains the changing state and fixed-base identity, not another copy of frozen base weights. Loading constructs the pinned model, adapters, optimizer, and scheduler, then loads saved training state and restores RNG states **last**, after setup that could consume random numbers. Verify state before update 21. An adapter export by itself is insufficient. [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).

Diagnostic collection must not consume training RNG or retain GPU/CPU tensor clones inside the process being captured. Hash tensors in bounded chunks, release temporary copies, and retain only small summaries. Full diagnostic records live outside the process image and are read by the verifier, never by process restoration. Record the observer's remaining allocation/cache footprint; do not assume deleting a Python reference returns memory to the OS.

## 4. Experiment sequence and gates

### Task 1: Freeze the environment and inspect compatibility

**Files:** `prepare.py`, `config.json`, `examples.jsonl`, `requirements.lock`; experiment manifest under `runs/`.

- [ ] Recheck live GPU, driver, Python, PyTorch/CUDA, DMTCP SHA/plugin, container RAM limit, available disk, and process mappings. September 14 observations were L4 23,034 MiB and 124 GiB RAM; those are historical inputs, not a substitute for checking.
- [ ] Create an isolated training environment. Start from the measured PyTorch 2.11.0+cu130 and Transformers 5.7.0 combination; resolve a compatible PEFT release there, validate imports and an actual forward/backward/update, then lock exact packages before comparative runs. Do not upgrade the working server environment in place.
- [ ] Download the resolved model/tokenizer revision once, prepare and hash local tokens, and run measured jobs offline against those same assets. Record the resolved package/model versions in the manifest rather than using moving `main` revisions at runtime.
- [ ] Inspect mappings in progressively initialized processes: Python, PyTorch import, CUDA initialization, loaded model, and initialized Adam state. Relate `/dev/zero (deleted)` warnings to the stage that introduces them and inspect aliases/backing identities.
- [ ] If sharing is required, demonstrate preservation with a focused reproducer or fix the configuration/tool support and revalidate. An absence of visible aliases in one process does not prove the driver needs none. Unknown ownership remains an explicit qualification; do not suppress the warning. Record an unsupported kernel-object mapping as a blocker for that workload.

**Gate:** A pinned environment performs real adapter updates with adequate measured memory headroom. Compatibility warnings have a documented status. Qualified diagnostic work can proceed, but unqualified acceptance cannot pass while required sharing remains unresolved.

### Task 2: Build and verify ordinary training

**Files:** `train.py`, `state.py`, `prepare.py`.

- [ ] Implement the exact setup and state contract above. Put all three execution modes through the same training-update function.
- [ ] Run 25 updates twice without interruption. Compare initial state, update-20 state, losses and state for updates 21–25, and final state.
- [ ] Enable deterministic algorithm checks, use the same numerical settings for all routes, and record any required CUDA determinism configuration in every child environment. A fixed seed alone is insufficient. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
- [ ] If references disagree, identify the operation or configuration before interpreting any restore difference. Do not choose a tolerance after seeing a failure; the first baseline targets exact equality.

**Gate:** Two ordinary runs match, adapters changed, Adam moments were populated, and the data cursor identifies the next example consistently.

### Task 3: Implement the complete application-checkpoint comparison

**Files:** `state.py`, `train.py`, `run.py`.

- [ ] At update 20, complete the optimizer and scheduler, clear gradients, release temporary outputs, synchronize CUDA, and record state.
- [ ] Serialize adapters plus the full state contract to a new candidate, flush it, and atomically publish the completed checkpoint on the selected filesystem. Preserve the previous completed save if a later save fails.
- [ ] End the original process and verify its real PID is gone. Start a fresh process with the same local assets and explicit interpreter; load state, compare it before update 21, then continue to 25.
- [ ] Compare every continuation update with the reference. Use cached fixed-base assets for both application restart and process restore.

**Gate:** Correct continuation from a real application save/restart. This is the baseline a process snapshot must be compared with.

### Task 4: Implement full DMTCP fine-tuning restoration

**Files:** `experiments/dmtcp/session.py`, `probe-dmtcp.py`, `train.py`, `run.py`.

- [ ] Extract the existing controller's lifecycle primitives into `session.py`, retaining CPU/tensor-probe behavior. Launch the fine-tuning interpreter through pinned DMTCP and its CUDA plugin from the beginning.
- [ ] Use the following generation-specific handshake for capture at update 20. A future update-22 capture uses separate `22` markers so earlier release files cannot bypass its wait.

```text
Trainer: complete update 20 → synchronize → write before-20 evidence
         → publish ready-20 → wait for inspect-20
Controller: capture → wait for finalized image → sync filesystem
            → end original → verify real PID absent → run and finish job B
            → restart image → publish inspect-20
Restored trainer: wait for CUDA restoration → write after-20 evidence
                  → publish inspected-20 → wait for continue-20
Controller: compare before/after/reference → publish continue-20 only on pass
Trainer: perform update 21, then continue through 25
```

- [ ] Let the plugin own GPU restoration. DMTCP resumes CPU threads as part of its GPU helper coordination; the restored trainer's CUDA synchronization blocks until GPU state is ready. Do not wait for a trainer readiness record before permitting the CUDA call needed to produce that record, and do not append manual NVIDIA restore/unlock commands.
- [ ] Keep `--no-gzip`, the default non-forked mode, and one process image. The pinned DMTCP writes `.temp` and renames after its write barrier; final-name appearance is the established completion observation for this bounded configuration. Sync is separate.
- [ ] Give each phase a bounded deadline. Initial defaults: 300 seconds to prepare the trainer, 120 seconds for capture or restore, 30 seconds for exit, and 60 seconds for job B. On timeout or mismatch, retain evidence, reject the run, and stop only identified experiment processes/coordinators. Never release the next training update after failed verification.
- [ ] Persist lifecycle, numerical, and compatibility verdicts separately even when the final acceptance is false. Do not lose successful tensor/state evidence merely because the later warning gate fails.

**Gate:** Original process ended; GPU was available to job B; the process image alone reconstructed the trainer; state matched before update 21; updates 21–25 matched the reference. Remaining warnings keep the result qualified.

### Task 5: Exercise randomness and repeated restoration

- [ ] Repeat the reference/application/process comparison with LoRA dropout 0.1. This exercises CUDA randomness in subsequent training, rather than checking unused RNG bytes only. Establish matching uninterrupted references for this configuration first.
- [ ] Run a separate process-restoration case with captures after updates 20 and 22. Use unique checkpoint directories and handshake markers, verify original exit each time, and compare through update 25.
- [ ] Require clean process exit and no remaining experiment GPU work/coordinator after the lifecycle tests. Retain old completed images until their replacements are verified.

**Gate:** Stochastic continuation and a second checkpoint cycle pass. State clearly which configurations passed and whether compatibility qualifications remain.

### Task 6: Measure and report

**Files:** `run.py`, `experiments/results.md`, curated evidence JSON.

- [ ] Separate correctness runs from timing runs. Detailed hashes and inspection waits are excluded from headline latency; use a timing variant with the same model/data/boundary and smaller evidence after the measured endpoint. Measure each variant's memory footprint.
- [ ] For both save methods, time from capture request to completed filesystem sync and from restore request to the next completed update. Include model/runtime rebuilding for application restart, including necessary CPU-to-GPU copies.
- [ ] Observe request-to-GPU-release separately. Polling gives an interval, not an exact driver timestamp. Include interpreter/coordinator launch in the end-to-end costs where required; record job B separately rather than charging its runtime as snapshot overhead.
- [ ] Record image/file size, peak process RAM, cgroup memory, allocated/reserved VRAM, baseline update time, and DMTCP launch overhead. Do three paired timing repetitions after correctness, report individual values plus median/range, and state cache conditions and local-storage durability limits.
- [ ] Publish a short state-continuity table, lifecycle timeline, cost comparison, pinned environment, and any unresolved blockers. No raw coordinator environment or process image is committed.

**Gate:** The report supports a narrow, reproducible conclusion; neither lower cost nor real spot recovery is assumed.

## 5. Tests worth keeping

Invoke the writing-tests skill before implementation. Extend coverage only for the new state/lifecycle contracts:

1. **Acceptance:** a local small tensor model with actual Adam state round-trips the application record, then performs the same next update; compare RNG, cursor, schedule, and moments as well as weights.
2. **Non-obvious correctness:** a corruption matrix changes an optimizer moment, cursor, RNG, or schedule while leaving weights intact; each discrepancy is identified by field name.
3. **Critical contract:** the controller cannot publish continuation before post-restore evidence matches, and capture generations cannot reuse old release markers. Include failed image completion/original-exit paths when extracting the session code.

These tests complement, rather than replace, the real L4 acceptance runs. Retain all three existing CPU tests. Do not create a test for each constant, JSON key, or trivial wrapper.

## 6. Scope and review decision

The recommended route is DMTCP on the existing permitted environment. CRIU remains a separate route on an appropriately permitted host, not an attempt to evade the current container restrictions. Full application checkpointing is a required comparison, not a replacement for process restoration.

Out of this implementation: actual spot termination, replacement-host restore, multi-GPU, QLoRA/paged optimizers, compression/incremental GPU capture, a scheduler, and billing integration. A later spot-recovery phase must independently prove image/assets survival on durable storage, compatible replacement capacity, and the warning deadline.

Review the workload, state contract, waiting sequence, and acceptance gates before implementation. After agreement, implement and review task by task; synchronize tested milestones through the existing GitHub/SSH workflow.

## Review follow-ups before execution

- Explicitly enter training mode when constructing the trainer, and record effective module modes, adapter activation, trainable flags, and dropout settings. In the dropout follow-up, require the used CUDA generator state to advance across a real update. Inspect the restored values before changing them, so a repair cannot hide an incorrect restoration.
- Distinguish temporary GPU release during capture from the completed handoff. The pinned DMTCP plugin can restore GPU resources when the original resumes after saving. Record image completion, original-process exit, and subsequent GPU availability separately; retain job B as functional evidence.
- Run a one-update LoRA capture/restore after Adam state initializes in the pinned environment, before building the complete comparison. Verify the next update. This early check reduces integration risk, but unresolved required sharing still prevents unqualified compatibility acceptance.
