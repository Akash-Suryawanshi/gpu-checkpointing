# Plan: request-triggered 8B cold starts

**Proposed; implementation paused.** Measure whether restoring an initialized
8B process reduces client-observed time to first token (TTFT). Confirm the cold
baseline, attack the known validation/storage cost, and repeat through HTTP.

This plan owns the new cold-cache and endpoint comparison. The
[original snapshot plan](inference-snapshot-plan.md) retains its completed
four-route CLI contract; historical results are not endpoint baselines.

## Known evidence and hypothesis

Historical EBS disk trials spent 134–138 s before CRIU, about 139.45 s inside it,
and 0.57–2.10 s afterward. Validation reads 15.26 GiB of model files and 16.61 GiB
of payload; CRIU reads the image again. These are roughly 48.5 GiB of logical
reads, not measured physical disk traffic: cache hits matter.

Hypothesis: repeated reads and observed ~125 MiB/s image I/O dominate restoration.
Measure physical read bytes and cache residency before attributing a byte ratio
or claiming an unavoidable full image reread on this memory-constrained host.

No accepted cold Transformers-default or cold disk baseline exists yet. The lone
passing cold loader diagnostic used instance NVMe; historical EBS cache residency
was uncontrolled. First task after approval: cold Transformers-default on EBS.

## Boundary

```text
client POST /generate -> live API -> activation controller
        START                         |-- fresh load --|
                                      |-- restore ----|-> worker
        STOP <------- token event <-----------------------|
```

Arrows show request dispatch and token delivery. The API/controller stays alive;
only the model worker is captured. HTTP sockets never enter its snapshot.

Scope: **model cold, deployment warm**. Machine, API, and eventual container are
running; files are local; model process is absent. TTFT includes request handling,
admission, activation-time validation, loading/restoration, tokenization, first
forward pass, and delivery to the client.

Container creation, downloads, artifact staging, and snapshot construction are
separate costs. If a later deployment starts containers on demand, include that
startup in its request timer and report it as a different comparison.

## Endpoints and lifecycle

| Interface | Contract |
| --- | --- |
| `POST /generate` | `{model, prompt, max_new_tokens: 16, temperature: 0}`. Reject unsupported settings, including nonzero temperature; activate if absent and stream `token`, `done`, or `error`. |
| Token event | `{request_id, activation_id, index, token_id, text}`. TTFT ends at the first generated-token event, never headers/readiness messages. |
| `GET /healthz` | API liveness; never loads the model. |
| `GET /models/{id}` | Model state, activation ID, last error. |
| `POST /admin/models/{id}/unload` | Local trial reset: reject while busy; otherwise terminate worker and verify GPU release. Keeps the API alive between cold trials. |
| Existing capture CLI | Publish an idle, warmed snapshot before trials; record build time and size. |

Fresh versus snapshot is server configuration; clients send identical requests.
Later requests reuse the ready model.

```text
ABSENT --request--> ACTIVATING --verified--> READY --unload--> ABSENT
                         |--failure--> FAILED --cleanup--> ABSENT
READY --request--> generate --done--> READY
```

One active request/activation; reject concurrent requests with `429`, without a
queue. Initially, deadline/disconnect cancellation stops and reaps the owned
worker/helper; it does not preserve a warm model. Failed cleanup blocks reuse;
restore failure never silently loads fresh weights. Cooperative cancellation
between tokens and queueing are follow-ups.

## Files and interfaces

Paths are under `experiments/inference/` unless qualified. Interfaces below are
proposed; existing CLI behavior remains available.

| File | Responsibility / reuse |
| --- | --- |
| **New `api.py`** | Thin HTTP adapter; call `Runtime.generate(request, deadline)` and flush token events immediately. No CUDA state. |
| **New `runtime.py`** | `ensure_ready`, `generate`, `unload`, `status`. Extract activation/admission/ownership from `run.py`; API and CLI share it. |
| `worker.py` | Reuse `load`, `render`, `generate`, `inspect`. Replace two-request loop with request loop and per-token callback; tokenize HTTP prompts and clear per-request KV cache. |
| `lifecycle.py`, `park.py` | Reuse capture/restore, process identities, cleanup, GPU admission, and resource sampling. |
| `../finetuning/{control,checkpoint,restore}.py`, `lifecycle.restored_candidate()` | Add the versioned [reusable image contract](inference-activation-contract.md): immutable publication, per-attempt state, host registration, and cleanup. Preserve training's single-use protocol. |
| `../criu/session.py` | Reuse CRIU, clock handling, fresh PID files, and process ownership; surface saved-PID conflicts without killing unrelated processes. |
| **New `bench_http.py`** | Reset state, verify cache condition, send requests, timestamp token receipt, check output and follow-up request. Reuse prepared references and `cold_cache.py`. |
| `measure.py`, `report.py`, `disk_profile.py` | Add `transport`, boundary/schema version, and integrity policy to the comparison key; validate HTTP timestamps and physical-I/O evidence. Reuse the existing fresh/disk, blocks 1–3 schedule. |
| `stream_weights.py` | Optional candidate, enabled only if profiling justifies it. |
| `Dockerfile`, `README.md` | Package the same service and document endpoint commands after host validation. |

Reuse file messages; measure polling/write overhead. Capture idle with no request
or KV cache. Repeated restore requires `activation.json` before CRIU, fresh
attempt paths, log handling, and matching acknowledgement before serving; see the
linked contract. Limit: sequential activations, saved PID free, same host/boot.

## Measurements and conditions

| Measure | Boundary / evidence |
| --- | --- |
| Primary TTFT | Client monotonic clock: before opening a new TCP connection to first token received; no keep-alive or automatic retries. |
| Request phases | Receive, admission, activation, tokenization, first forward pass, token flush. Correlate IDs; do not subtract timestamps from different clock domains. |
| Fresh activation | Import/construction, disk reads, tensor setup, host-to-GPU copies, CUDA initialization. Detailed tracing only in diagnostics. |
| Snapshot activation | Dependency/payload validation, CRIU reconstruction, post-restore readiness; reuse host-observer events. |
| I/O | Logical bytes hashed/read separately from kernel-accounted storage reads; collect helper/child counters and device deltas on an idle disk, noting attribution limits. |
| Correctness / costs | Exact tokens and weights, later request success, errors/timeouts, sampled CPU/GPU peaks, retained memory, snapshot size/build time, staging cost. |

Show overlapping reads/copies as timeline spans. Exclusive phases plus an
unattributed remainder reconcile server time; compare client/server durations
without assuming synchronized clocks. All request-path checks remain timed.

- **Fixed workload:** pinned Qwen3-8B, BF16, A10G, batch one, greedy decoding.
  One fixed timing prompt; a fixed prompt set for diagnostics. Record software,
  source/artifact hashes, storage, settings, and timeout policy.
- **Primary pair:** fresh versus snapshot on EBS `/mnt/data`; NVMe is a separate
  storage arm. Confirm the backing volume and provisioned throughput; a device
  name or observed rate alone does not establish gp3 settings. No model
  process/GPU allocation; evict relevant weight/image pages after preparation and
  verify zero residency before the request. No untimed rereads afterward.
- **Cache label:** cold model-file cache; other caches uncontrolled. Eviction is
  benchmark setup, never an operation in the public generation endpoint.
- **Repetitions:** matching diagnostic, then three alternating timing pairs.
  Report every attempt, median, range, failures, and follow-up warm TTFT. Run
  full weight audits after the response pair, outside its timings.
- **p95:** leave blank until a larger campaign, estimator, and uncertainty are
  agreed. Keep models, disks, cache policies, host/container, and diagnostics separate.
- **Later controls:** explicit warm-file-cache and RAM-retained cases. RAM parking
  preserves a living process; it is not a freshly deserialized CPU-only model.
  Add a warm-interpreter/cold-weights control: fresh currently pays Python/import
  startup and is not claimed to be the fastest possible non-snapshot server.

Storage: ~68.7 GiB free on EBS at review; recheck the existing
`2 × (RSS + reserved VRAM)` capture-admission floor. Budget one new ~16.61 GiB
image, no baseline bundle. Before reuse support, capture/discard sequentially;
afterward retain one image across pairs. Preserve historical artifacts.

## Improve one measured bottleneck

| Dominant cost | Intervention | Controlled comparison |
| --- | --- | --- |
| Disk bandwidth | First performance arm: existing instance NVMe. | Move both routes; change only storage and retain strict validation. Report staging cost and ephemeral lifetime. |
| Restore-time model hashing | Conditional policy arm: validate at publication and avoid rereading proven-unused model files. | First audit CRIU mappings/open files and test independence. Require an enforceable artifact-lifetime/immutability contract; record a new integrity policy. File identity alone is not content verification. |
| Validation cache locality | Benchmark existing order/worker flags only if useful. | They reorder/parallelize checks, not remove logical bytes; change one flag at a time. |
| Inefficient weight loading | ServerlessLLM-inspired bulk reads, pinned buffers, overlap, direct I/O. | Add one step at a time against the installed Transformers default. |
| Initialization/restore work | Refine the idle snapshot boundary to retain reusable initialized state. | Count capture cost; require identical later responses and empty request state. |

Reuse exploratory code only as candidates. Optimize transport only if measured
overhead matters. [Published methods](../research/inference-cold-start.md) guide
the choice; a snapshot win remains a hypothesis.

## Milestones

1. **CLI baseline and storage arm:** cold Transformers-default first, then strict
   disk baseline and matched NVMe pairs. Use existing commands; do not wait for
   HTTP refactoring. CLI results diagnose mechanisms, not endpoint latency.
2. **Reusable image contract:** versioned publication/attempt handling, then
   worker handshake/loop in separate commits. Test two sequential activations,
   PID conflicts, changed logs, stale records, failures, and cleanup.
3. **Refactor only:** extract `runtime.py` with CLI behavior preserved and tests
   green. Split ownership/admission from request dispatch if needed.
4. **Endpoint and client:** separate thin API and benchmark commits. Verify
   immediate streaming, strict settings, busy rejection, and cancellation cleanup;
   rerun baseline-versus-best through HTTP. This is the headline result.
5. **Container replay:** repeat diagnostics and the HTTP comparison using the
   validated container profile; keep host/container evidence separate.

Use the writing-tests skill before test changes. Aim for 100–200 changed lines
per review unit, including tests/docs; split anything over 400. These milestones
are groups of commits, not single oversized PRs. The
user reviews and merges. Implementation resumes after this plan is reviewed.
