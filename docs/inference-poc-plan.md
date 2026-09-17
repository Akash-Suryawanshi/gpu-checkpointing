# Inference POC: implementation and testing plan

**Goal:** show how quickly a model can answer again, which resources it keeps
while idle, and which saved states survive the original process ending.
Use this A10G and Qwen3-8B in BF16, a format that uses two bytes per model weight.

This is an implementation plan, not new performance evidence. The 8B files are
downloaded, but preparation was stopped before its completion record was written.
The scripts in `experiments/inference/` are unfinished; RAM parking is missing.

Keep the [validated small-model experiment](../experiments/results.md#independent-lifecycle-validation--2026-09-17)
as the fallback. If 8B cannot fit or restore reliably, run the same comparison on
the smaller model and label its size. Do not mix model sizes in a speedup ratio.

## 1. What we will compare

A **worker** is a running Python program that owns the model. A **token** is one
piece of generated text; first-token time ends when the observer receives that
piece, rather than when the model merely reports that it is ready.

| Route | State kept while idle | Timer starts |
| --- | --- | --- |
| Fresh | Local model files; no running model worker | Before launching the worker |
| Resident | Live worker with model on GPU | Before sending a new request |
| RAM parked | Live worker; saved GPU state in the machine's ordinary memory | Before asking NVIDIA's helper to restore GPU state |
| Disk saved | Completed process image and its required files; original worker gone | Before launching the independent restore worker |

Every timer ends at the observer's first-token receipt. Preparation, parking,
and saving have separate timers. Only the fresh and disk routes reconstruct a
worker; RAM wake-up requires the old worker and machine to remain alive.

```text
                  save GPU state in RAM -> worker stays alive -> wake GPU
loaded, idle model
                  save process to disk -> worker exits -> restore process
                                                              |
                           send prompt -> observe first token <-+
```

The branches are separate trials. Arrows show operations in execution order.

## 2. Step 1 — Finish preparation and check capacity

**Files:** `experiments/inference/prepare_assets.py`; existing environment checks.

```text
local model files -> verify contents -> publish asset record -> load once -> check memory
```

1. Reuse the isolated Python installation and pinned NVIDIA/CRIU tools. Record
   Python, library, kernel, driver, GPU, and model revision details.
2. Verify every required model shard exists. Hash the actual files and save their
   sizes; a hash is a fingerprint used to detect changed bytes.
3. Finish `manifest.json`, the asset record, only after all checks succeed.
   An incomplete preparation directory must never count as ready.
4. Keep model files and snapshots on the data volume. Measure free disk space,
   available host RAM, and free GPU memory before starting a trial.
5. Load 8B and answer one short prompt. Check that every model weight is on the
   GPU and that no automatic placement on CPU or disk changed the experiment.
6. Before parking or capture, allow RAM for staged GPU bytes, the live Python
   program, other processes, and a stated reserve. Check actual memory limits,
   not just the GPU's advertised capacity. Stop the trial if headroom is unsafe.

**Pass:** preparation has a complete asset record, a correct first response, and
recorded capacity checks. If it fails, retain the reason and select the fallback
before spending time on new precision formats, models, or package upgrades.

## 3. Step 2 — Complete fresh and resident inference

**Files:** `experiments/inference/worker.py`, `run.py`, and `measure.py`.

```text
fresh:    start timer -> launch/load -> request -> first token -> stop timer
resident: start timer ---------------> request -> first token -> stop timer
```

Each time ends at its first token, before the full response finishes.

1. Keep one model, one short test prompt, one request at a time, BF16, and at most
   16 output tokens. Disable thinking mode and choose the highest-scoring next
   token each time; stop on the model's end token. Save these settings.
2. Use a separate short warmup prompt for RAM/disk preparation. Delete the
   attention cache, which stores values from earlier input tokens, afterwards.
   Wait for queued GPU operations to finish before announcing an idle worker.
3. Give each run and request a unique identifier. Every ready, first-token,
   response, and completion record must identify its run/request as applicable.
   An old file must not satisfy a new wait.
4. Record fresh time before process launch. Include imports, loading, GPU setup,
   prompt processing, and first-token delivery. Do not warm the fresh route first.
5. After the first response, send the same prompt again to that live worker.
   Start the resident timer before sending it. Rebuild its attention cache from
   the prompt rather than reusing the previous answer's cache.
6. Save all generated token IDs and the decoded text. Verify that the separately
   reported first token is the first ID in the completed response.

**Pass:** two independent fresh runs return identical token IDs, and each worker's
resident response matches. This becomes the reference for both recovery routes.
A result is not accepted merely because two responses from one restored worker match.

## 4. Step 3 — Add RAM parking and wake-up

**Files:** add `experiments/inference/park.py`; connect it through `run.py` and
`worker.py`. Reuse the operations in `experiments/gpu/probe-gpu.sh`.

```text
finish warmup -> worker waits on CPU -> lock -> checkpoint GPU into RAM
 -> observe memory -> run job B -> job B exits -> restore -> unlock
 -> verify saved state -> send prompt -> observe first token
```

The worker must stay alive and issue no GPU work while parked.

1. Record the worker's process identity, including its start time, before acting.
   A reused numeric process ID must never identify the worker accidentally.
2. Hash model state before parking. Use NVIDIA's installed helper to lock new GPU
   calls and stage GPU state in RAM. Bound every command and wait with a deadline.
3. Require the helper to report `checkpointed`. Record GPU memory released and
   host RAM retained; a log message alone does not prove the GPU became usable.
4. Run the existing independent job B, which allocates GPU memory and checks a
   calculation. Record its success, then confirm it exits before waking the model.
5. Start the wake timer, restore GPU state, and unlock CUDA operations. Compare
   untouched model state before sending the test request; keep this check inside
   the reported wake time. Then compare output with the fresh reference.
6. On failure, record the phase and stop. Do not guess whether another restore
   or unlock is safe. Clean up only the identified worker and job B, using
   `session.kill_identified()` and parent-owned exit collection.

**Pass:** the same CPU process survives, GPU memory is reclaimed, job B works,
state and output match, and the worker can handle a second request.
This does not prove recovery from process or machine loss.

## 5. Step 4 — Complete independent disk recovery

**Files:** finish `experiments/inference/run.py` and `worker.py`; reuse
`experiments/finetuning/checkpoint.py`, `restore.py`, `control.py`, and
`experiments/criu/session.py`. Do not build a second persistence protocol.

```text
idle worker -> capture command -> persist files + completion record
 -> verify original removed and capture worker exited -> job B -> job B exits
 -> fresh restore command -> compare untouched state -> release requests
 -> restore worker exits -> model answers -> observer verifies output
```

1. Use a fresh snapshot and request directory for every attempt. Include all
   inference source files in dependency checks so edited code cannot silently
   restore an image made by a different implementation.
2. Capture only after warmup and attention-cache removal. Reuse the existing
   ordered file/directory writes and completion rules from the
   [publication contract](independent-lifecycle-details.md#exact-local-storage-protocol).
3. The launch parent must collect the original worker's exit status. The capture
   worker observes its removal; it cannot collect an unrelated process's status.
4. Require complete publication and both exits before beginning restore. Run
   job B between them to prove the released GPU is usable.
5. Start the timer before launching `restore.py`. Preserve asset/image checks,
   reconstruction, untouched-state comparison, release, and request delivery
   inside that timer. CRIU's plugin owns GPU restore; do not call manual unlock.
6. Ensure the model remains alive after the restore worker exits. Send the prompt,
   compare with the fresh reference, and require a second valid response.
7. Verify image hashes remain unchanged after restore. Do not reuse a consumed
   snapshot for timing repetitions; create a new capture for each one.

**Pass:** publication, process independence, state equality, output equality,
and unchanged image checks all succeed. This proves same-host recovery with
matching dependencies, not recovery after losing the machine or its disk.

## 6. Step 5 — Make results trustworthy

**Files:** extend `measure.py`; add a small `experiments/inference/report.py`.

```text
raw events + outputs -> reject invalid runs -> compare routes -> table and plots
```

1. Take start/end times from one observer's monotonic clock, which measures
   elapsed time without wall-clock adjustments. Do not subtract restored worker
   timestamps from observer timestamps; CRIU can restore a different clock offset.
2. Reject missing/duplicate request records, reversed times, first-token mismatch,
   and any route whose full response differs from the fresh reference.
3. Record the file-message polling interval and disk-write overhead. The draft
   polls every 50 ms; it cannot support claims of finer latency differences.
   Use the same observation method across routes and state its limits.
4. Record readiness and first-token times separately. Sample GPU and host memory
   before parking, while parked, during capture/restore, and after completion.
   Report peaks as sampled peaks, along with the sampling interval.
5. Keep local model files identical. State whether Linux may already have their
   contents in RAM. Use a documented preparation order and rotate route order;
   do not clear the whole machine's cache or call an uncontrolled cache “cold.”
6. Aim for three complete comparisons, each including fresh, resident, RAM, and
   disk results. If a route has fewer successes, show its actual count and failures.
   Never hide failed runs or present three runs as a reliable worst-case estimate.
7. Save raw events, configuration, source hashes, responses, memory observations,
   snapshot sizes, warnings, and pass/fail reasons. The report reads these files;
   it must not contain manually entered benchmark numbers.

**Pass:** the report rejects incompatible model/settings or missing evidence,
shows every individual timing plus median/range, and clearly marks unmeasured
routes. A positive speedup is not a pass condition.

## 7. Tests that are necessary

Read the `writing-tests` skill before editing tests. Each test's docstring must
state its purpose: `Critical:`, `AC...`, `REGRESSION (...)`, or
`Non-obvious correctness:`. Reuse existing ownership/publication tests.

| Test | Required check | Purpose |
| --- | --- | --- |
| Timing and response validation | Wrong request, wrong first token, output drift, or reversed time produces no accepted timing. | Critical contract |
| Safe waiting | Stale ready files and expired waits cannot release a new request. | Critical contract |
| RAM failure cleanup | Fail before/after each helper action; never serve early or signal an unrelated process. | Critical contract |
| Disk refusal | Missing completion, changed source/image, or state mismatch prevents release. | Existing acceptance checks; extend only for new inference behavior |
| Full GPU comparison | Fresh, resident, parked, and restored token IDs match; job B and second requests succeed. | Acceptance criterion |
| Report admission | Different model/settings or absent evidence cannot become a claimed speedup. | Critical contract |

Use small CPU-only records and real child processes for control/error tests.
Fake helper failures may test cleanup, but cannot establish GPU compatibility.
The full comparison must use the real model, helper, GPU, and saved images.

Use the existing [CPU test command](../experiments/finetuning/README.md), including
`experiments/cpu` in `PYTHONPATH`. The previous draft test run failed collection
because `verify_cpu` was not on that path; it is not evidence of a passing suite.

## 8. Completion and presentation

Write runnable commands in `experiments/inference/README.md` once implemented.
Put curated measurements in `experiments/results.md` and its evidence directory;
large images and raw logs stay private on the data volume.

Deliver one live park/use/wake sequence, a disk-recovery example, a latency table,
a resource-use plot, and a recording. Every displayed result identifies the model
and says measured, estimated, failed, or not run.

```text
request -> resident worker available? -> serve
        -> parked worker available?  -> wake, verify, serve
        -> eligible disk image?      -> restore, verify, serve
        -> otherwise                 -> load normally, serve
```

This is a proposed production decision flow, not a deployed scheduler. A failed
recovery must first clean up its own attempt before a normal-load fallback.
Freeing GPU memory creates reusable capacity; the running VM still costs money.

Commit preparation/baselines, RAM parking, disk integration, and reporting as
separate reviewable changes with their necessary tests. Keep each step usable.
If a route blocks progress, retain its failure and demonstrate the working routes.
Reserve time for repeated runs and rehearsal rather than adding new features.

## 9. Supporting context, not more implementation

Reuse [existing LoRA evidence](../experiments/results.md#independent-lifecycle-validation--2026-09-17):
about 6.7 MB of application state versus 3.7 GB process images for the validated
0.5B workload. Frozen weights explain the size difference. No new rank sweep,
70B run, quantization stack, compression, distributed restore, or full dashboard.

- [NVIDIA](https://github.com/NVIDIA/cuda-checkpoint): reuse installed RAM staging; CPU threads remain alive.
- [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/): explain its similar resource choice; do not install another serving stack.
- [Modal](https://modal.com/blog/truly-serverless-gpus): relate initialized-worker snapshots to serving; do not borrow its performance claims.
- [GCR](https://www.usenix.org/system/files/fast26-zeng.pdf): explain copying costs and avoiding unchanged data after measuring our bottleneck.
- [PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) and [compression research](https://radostin.io/files/stoyanov-euromlsys-2026.pdf): leave these integrations for later.
