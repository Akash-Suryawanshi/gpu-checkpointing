# Inference snapshots: implementation and testing plan

**Goal:** measure time to the first answer, resources retained while idle, and
whether inference survives the original process ending. Start on one A10G;
complete the pipeline with a smaller model if 8B exceeds GPU or host-memory limits.

The local implementation passed with Qwen3-8B on A10G: five diagnostics, twelve
scheduled timings, and separate disk capture/restore commands. This document keeps
the acceptance contract. A pinned container also passed CPU/GPU restore probes
and five 8B diagnostics on the same host; [results](../experiments/results-detail.md#inference-activation-and-gpu-reuse--2026-09-17)
own measurements and the [runbook](../experiments/inference/README.md) owns commands.

```text
check capacity -> prepare model -> validate four routes -> measure and report
                       ^                  |
                       +-- smaller model if capacity blocks progress
                                          |
                               complete local pipeline
                                          v
                             containerize -> validate new host
```

A model change starts a separate comparison. Container packaging follows local
validation; moving an image between machines is a separate compatibility question.

## 1. Scope and comparison

A worker is a Python process holding the model. A token is a piece of generated
text; all primary timers end when the controller receives the first token.

| Route | State retained while idle | Primary timer starts |
| --- | --- | --- |
| `fresh` | Local model files | Before worker launch |
| `resident` | Independent live worker with model on GPU | Before request publication |
| `ram` | Live worker with GPU state staged in host RAM | Before GPU restore request |
| `disk` | Published process image and required external files; original gone | Before launching the restore command |

Use Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`,
BF16 (two bytes per weight), PyTorch's built-in attention (SDPA), one GPU, and one
request at a time. Choose the highest-scoring next token (greedy decoding),
with thinking disabled where supported and at most 16 new tokens;
honor the end-of-sequence token and save the exact rendered prompt and token IDs.

Fallback: prepare the existing Qwen2.5-0.5B files at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` for inference with the same checks.
Its [training evidence](../experiments/results-detail.md#independent-lifecycle-validation--2026-09-17)
is not inference validation; keep its assets, diagnostics, timings, and report separate.

Fresh has no warmup. Resident, RAM, and disk each start an independent worker,
run the same short nonbenchmark warmup, remove the attention cache (stored values
from earlier tokens), and finish queued GPU work before reaching the idle boundary.

```text
fresh:     launch/load ---------------------------------> request -> first token
resident:  load/warmup -> idle ---------------------------> request -> first token
RAM:       load/warmup -> park -> job B -> restore/verify -> request -> first token
disk:      load/warmup -> save/exit -> job B -> restore/verify -> request -> first token
```

Preparation and saving are outside the primary wake timers and reported separately.
A second request checks continued operation; its latency is
`health_request_to_first_token`, never the resident baseline.

## 2. Files and interfaces

Keep one controller, one model worker, and small functions for parking and result
validation. Reuse existing process ownership, durable file writes, and snapshot
publication; comments explain identity checks, clock handling, and ordering.

| File | Work and responsibility |
| --- | --- |
| `experiments/inference/prepare_assets.py` | Finish draft: validate local files, pin inputs/environment, run smoke inference, publish complete assets last. |
| `experiments/inference/worker.py` | Finish draft: shared load/generate/inspect functions; request handling; idle capture boundary. Preparation reuses these functions. |
| `experiments/inference/run.py` | Finish draft: supervise children, capacity checks, four routes, diagnostics gate, timing, failure records, cleanup. |
| `experiments/inference/measure.py` | Finish draft: pure record validation, compatibility key, durations and reference comparisons. |
| `experiments/inference/park.py` | Create: bounded NVIDIA helper calls and explicit RAM state transitions; no extra supervising process. |
| `experiments/inference/report.py` | Create: read-only summary, tables, and simple SVG charts using the standard library. |
| `experiments/inference/README.md` | Maintain the intended CLI and live runbook; mark commands ready only after acceptance. |
| `experiments/finetuning/control.py` | Extend dependency hashes to inference sources; optional polling interval, retaining training defaults. |
| `experiments/criu/session.py` | Reuse launch/identity/reap/CRIU helpers; optional polling and explicit cache-variable allowlist. |
| `experiments/finetuning/checkpoint.py` | Reuse capture; include pre-staging memory observations in the published manifest for restore admission. |
| `experiments/finetuning/restore.py` | Reuse validation, identity registration, inspection, and release; no second identity file or manual GPU restore. |
| `experiments/finetuning/job_b.py` | Add `--mib`; preserve the existing small default. Touch the allocation and verify a calculation. |
| `tests/test_inference_measure.py` | Finish draft: result admission and reporting contracts. |
| `tests/test_inference_lifecycle.py` | Create: necessary RAM transitions, deadlines, capacity refusal, and restored-child cleanup checks. |
| `experiments/results.md` | Add measured findings only after runs, linking curated evidence. |

Public commands and defaults are in the [runbook](../experiments/inference/README.md#command-contract).
Internal worker arguments are `--route`, `--kind`, `--assets`, `--run-dir`, and
`--run-id`; the controller supplies them and creates a fresh run directory.

Use these narrow function contracts; names can follow existing local conventions:

- `inspect(kind)` returns full untouched-state evidence for diagnostics and cheap
  metadata for timing: dtype, evaluation mode, tensor shapes/devices, and absent cache.
- `park(identity, deadline)` and `wake(identity, deadline)` require a matching
  process identity and record every helper transition.
- `validate_run(records, reference, diagnostic)` rejects incomplete or incompatible
  evidence before producing accepted durations; reporting calls the same validator.

## 3. Preparation and admission

1. Reuse the isolated Python environment and pinned tools. Record versions,
   source hashes, GPU/driver/kernel, model revision, decoder settings, and file hashes.
2. Check shard completeness and tokenizer inputs; save benchmark and warmup inputs.
   Run smoke inference, verify every weight is on the GPU, and save reference tokens.
3. Write `manifest.json` only after success. A complete asset directory may be
   revalidated without rewriting it; refuse partial directories and use a new path.
4. Measure the actual warmed worker's resident host memory (RSS), GPU allocation,
   GPU reservation, and free space. Model parameter count alone does not establish fit.
5. Record admission inputs and refusal reason before any park, capture, or restore.
   If 8B cannot complete safely, run the smaller model end to end before scaling up.

| Resource | Admission requirement |
| --- | --- |
| GPU | Model's observed reservation plus 1 GiB free reserve; reject foreign compute processes. |
| Available host RAM | Minimum of `MemAvailable` and remaining finite limits of all ancestor memory cgroups (Linux process-group limits). |
| RAM park / disk capture | Additional staged GPU bytes plus 4 GiB reserve; the live worker's RSS is already accounted for in available RAM. |
| Disk restore | Recorded pre-staging RSS plus staged GPU bytes plus 4 GiB reserve, checked while the original is absent. |
| Snapshot storage | Preserve existing `2 * (RSS + reserved GPU bytes)` admission floor; it is not a predicted image size. |
| Root and data volumes | Record both; put model files, images, caches, and temporary writes on the data volume. |

Use reserved GPU bytes as the conservative staging estimate and record observed
host-memory growth. Do not count a staged copy twice or replace the fixed reserve
with RSS; retain the pre-staging observation in the hashed snapshot manifest.

Shell exports alone are insufficient: `session.child_environment()` currently
constructs a fixed environment. Explicitly copy only `TMPDIR`, `HF_HOME`,
`XDG_CACHE_HOME`, `TORCH_HOME`, and optional `CUDA_CACHE_PATH` when set, including
into checkpoint and restore subprocesses; do not inherit the entire shell environment.

## 4. Correctness before speed

Run two independent fresh diagnostics and one per other route. All must match
preparation's reference token IDs and first-token record; differences fail validation.
Use evaluation mode throughout and never repair model state from comparison records.

```text
diagnostic: full fingerprint -> park/save -> restore -> full comparison -> request
timing:     full fingerprint -> park/save -> restore -> metadata check  -> request
                                                                        |
                                                    response 1 -> full audit -> accept
```

A fingerprint hashes model weights and persistent buffers. Both modes save
`audit-before.json` before parking/capture; timing writes `audit-after.json` after
response 1 and requires equality before admitting its measured latency.

Fresh must not hash the loaded model before its first request: that would add
preparatory GPU work. Its post-response audit compares with preparation's full
model fingerprint; resident takes its before-audit outside the request timer.

For disk, pass `inspect(kind)` through existing `control.boundary()` and
`restore.release()` so the saved `before.json` and restored `after.json` compare
the same contract. Diagnostics compare full untouched state before release;
timing compares metadata there and verifies full immutable state after response 1.

Do not assert that random-number state remains unchanged after generation.
Record any diagnostic RNG checks separately; full model equality and exact output
remain mandatory. Artifact hashes and durable publication checks remain enabled.

Timing requires `--validated-run` pointing to a passed diagnostic for that route.
Before launch, match a key covering model/assets, prompts/decoder, source hashes,
libraries/tools, GPU/driver/kernel, and inspection policy; exclude route, kind,
request IDs, and block number from the common comparison key.

Do not subtract inspection time from an observed result: inspection also changes
memory/cache state. Disk may additionally report activation from completion of
validation to first token, but the primary measurement includes validation from launch.

## 5. RAM park, reuse, wake

```text
running -> lock -> locked -> checkpoint -> checkpointed
  -> job B allocates/computes/exits -> restore -> locked -> unlock -> running
  -> inspect -> request -> first token -> full audit -> health request
```

1. The idle worker waits on CPU and issues no GPU operations while parked.
   Verify boot ID, PID, start ticks, and owner before each helper operation.
2. Use one deadline across commands and state waits; require the helper's expected
   state after each action. An unknown state fails without a guessed restore retry.
3. Record host RAM retained and GPU memory released. Run job B in its own process,
   verify its calculation, and collect its exit before starting the wake timer.
4. When possible, allocate `min(12288, post-release-free-MiB - 2048)` MiB for job B;
   require this to exceed pre-release free memory by at least 1024 MiB to claim
   an allocation made possible by parking.
5. If that condition cannot hold, use a 256 MiB availability probe only when it
   leaves 2048 MiB free. Label it as an availability check, not proof of otherwise
   impossible allocation; this is expected to be useful for the smaller model.
6. Wake the same identified CPU process, inspect, request, audit, and check the
   second response. No surviving CPU worker means RAM recovery is unavailable.

## 6. Disk capture, recovery, and ownership

```text
supervisor launches worker -> idle boundary -> capture command publishes image
 -> supervisor reaps original -> capture command exits -> job B exits
 -> independent restore command registers identity -> compares -> releases -> exits
 -> supervisor owns restored worker -> requests/audits -> cleanup/reap
```

The supervisor stays alive during each complete trial and acts as a subreaper:
Linux gives it orphaned descendants so it can collect their exits. `park.py` is
a library, not another supervisor; SIGINT and failures enter bounded cleanup.

1. Register the worker before model initialization with `until=2`; after warmup
   acknowledge boundary 1 and capture with `--at 1`. Reuse `session.launch()`
   so external log files required by the snapshot protocol exist.
2. Use fresh request, snapshot, and attempt directories. Preserve the existing
   [publication contract](independent-lifecycle-details.md#exact-local-storage-protocol),
   absolute external-file paths, dependency hashes, and completion rules.
3. Reap the original before restore, allowing CRIU to reuse its numeric PID.
   Require publication and capture-command exit; job B runs only after release.
4. Start the primary timer before the independent restore command. The CRIU CUDA
   plugin owns GPU restore/unlock; the controller must not repeat these operations.
5. Wait for restore-command exit, then require matching reference output, full
   audit, health response, and unchanged image hashes. Restore success alone is insufficient.
6. Cleanup must read refreshed `job.json` even if the final restore result is
   missing: `restore.py` registers host-observed identity before inspection/release.
   Validate run/job, capture/attempt phase, attempt PID file, adopted-parent ownership,
   and `session.matches()` before signaling; registration is not permission to serve.
7. Preserve the existing early-error fallback using attempt PID file, adopted child,
   and matching run argument for failures before registration. Collect exited
   children without confusing an empty command line with a live unrelated process.
8. Staged `capture` may exit after saving; a later `restore` invocation uses that
   same run directory and adopts the reconstructed worker. Capture writes
   `capture-result.json`, never a successful inference `result.json`.

Every timing trial creates its own image. `--discard-image-after-success` may
remove only that accepted disk timing's image payload after final hash checks;
retain manifests, hashes, logs, outputs, resource observations, and failed images.

## 7. Measurement and report

Timestamp primary start and first-token receipt in the same controller's monotonic
clock. On receipt, validate run/request identity and immediately record time;
later check that the completed response starts with that token.

Use the same durable file-message transport across routes. Configure inference
request/marker polling to 5 ms, leaving the existing training default at 50 ms;
record message-write and observation overhead. Two polls contribute about 10 ms
nominally, with additional filesystem/scheduling delay; this is not a hard bound.

Preserve raw events. Never subtract a restored worker clock from the controller
clock without measured namespace offsets; report helper-local phase durations
separately when cross-process alignment is unavailable.

| Artifact | Required content |
| --- | --- |
| `run.json` | UUID, route/kind/block, compatibility key, diagnostic path/hash, source/config/environment, cache paths. |
| `events.jsonl` | Raw observer times, request IDs, phase changes, first-token receipt, helper outcomes. |
| `memory.json`, resource samples | Pre-staging RSS/reservation and admission inputs; sampled host/GPU peaks with sampling interval (100 ms default). |
| `audit-before.json`, `audit-after.json` | Immutable model fingerprints and comparison result. |
| Request/response records | Run/request IDs, prompt IDs, first token, complete tokens/text, separate health request. |
| `result.json` | `passed` or `failed`, phase/reason, cleanup status, evidence paths, accepted durations only if all checks pass. |
| `capture-result.json` | Staged capture/publication and original-exit evidence; no inference timing claim. |

Publish the intended twelve-run schedule as `schedule.json` before starting
measurements; the report reads it to retain attempts missing a final result.
Use three blocks in this fixed order; do not replace failed runs:

| Block | Order |
| --- | --- |
| 1 | fresh, resident, ram, disk |
| 2 | resident, ram, disk, fresh |
| 3 | disk, fresh, resident, ram |

Report each time plus median and min–max. Produce paired fresh/route speedups only
when all three block pairs pass and match; otherwise show individual accepted
measurements, failures, and `not_run` rows without an aggregate speedup.

The report must resolve the diagnostic supplied by each timing run even when it
lives outside `--runs`, and recheck its saved hash/key. Hashes changed by code or
environment updates require new diagnostics and a new campaign.

Generate `summary.json`, `runs.csv`, `latency.svg`, `resources.svg`, and `index.html`
in a new output directory. Keep report generation read-only against run evidence;
show failures and missing routes rather than silently dropping them.

Label local file-cache conditions as uncontrolled unless measured: repeated local
loads are not guaranteed cold storage reads. Three trials do not establish tail
latency; positive speedup is not a correctness condition.

## 8. Testing and completion gates

Read the user-scope `writing-tests` skill before editing tests. Each test declares
Regression, Critical contract, Acceptance criterion, or Non-obvious correctness
in its docstring; reuse current ownership/publication tests instead of duplicating them.

| Gate | Necessary checks |
| --- | --- |
| CPU measurement contracts | Reject stale/wrong/duplicate IDs, reversed times, first/full-token mismatch, output drift, missing diagnostics, changed keys/hashes; keep health latency separate. |
| CPU lifecycle contracts | Fail around each RAM helper transition; deadlines do not release early; capacity refusal starts no helper; matching cleanup never signals an unrelated PID. |
| Restored-child regression | Inject restore failure after release but before final result; refreshed registration still permits the owning supervisor to clean up and reap. Retain early-registration failure coverage. |
| Report acceptance | Keep missing/failed schedule rows; reject mixed model/host/settings, resolve external diagnostics, and omit aggregate speedup for incomplete pairs. |
| Real GPU diagnostics | Two fresh references and all other routes agree; cache is absent at idle, full state survives, job B succeeds, second request works, all children are reaped. |
| Real GPU timings | Three planned blocks, fresh snapshot each disk trial, all accepted runs fully audited; failures retained. |

Check generation stops at EOS and never exceeds 16 tokens during acceptance;
a full-length response alone does not prove early-stop handling. Mocks establish
control behavior only; real GPU runs establish CUDA/helper compatibility.

Commit preparation/baselines, RAM parking, disk integration, and reporting as
separate reviewable changes with their necessary tests. Show one live park/use/wake
sequence, one saved-process recovery, and the generated latency/resource report.

## 9. Container packaging and later scaling

Only after the local pipeline passes end to end, add a pinned container definition
and launch instructions. Mount model/assets/results externally, keep the host GPU
driver on the host, and document the actual runtime permissions needed by CRIU.

Create `experiments/inference/Dockerfile` and a small build-context ignore file;
extend the runbook with build/run commands and required mounts. Pin the validated
Python packages and snapshot tools, and keep images and model weights out of the build.

```text
validated local pipeline -> pinned container -> CPU/GPU compatibility probes
 -> same-host four-route validation -> larger-model campaign if needed
```

On each target host, record GPU/driver/kernel, storage, host RAM and container
limits; repeat namespace, CPU restore, and GPU restore probes before the model
matrix. A container does not make a captured process portable across drivers/hosts.

Keep 70B as an optional illustrative transfer projection: assume
70,000,000,000 weight bytes for one byte per parameter, excluding quantization
metadata, runtime buffers, and process state. Divide stated bytes by separately
identified transfer/read/write rates; label cache assumptions and estimates.

This is neither measured 70B readiness nor a rigorous lower bound. Retain the
[existing fine-tuning research](../research/single-gpu-finetuning.md) as context;
no new rank sweep, quantization stack, scheduler, or cross-host recovery claim is
needed to explain the measured inference trade-offs.
