# Plan: request-triggered 8B cold starts

**Proposed; implementation paused.** Measure whether restoring an initialized
8B process reduces client-observed time to first token (TTFT). Profile first,
improve one bottleneck, and repeat with unchanged correctness requirements.

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
queueing, activation-time validation, loading/restoration, tokenization, first
forward pass, and delivery to the client.

Container creation, downloads, artifact staging, and snapshot construction are
separate costs. If a later deployment starts containers on demand, include that
startup in its request timer and report it as a different comparison.

## Endpoints and lifecycle

| Interface | Contract |
| --- | --- |
| `POST /generate` | `{model, prompt, max_new_tokens: 16, temperature: 0}`. Activates if absent; streams `token`, `done`, or `error` events. |
| Token event | `{request_id, activation_id, index, token_id, text}`. TTFT ends at the first generated-token event, never headers/readiness messages. |
| `GET /healthz` | API liveness; never loads the model. |
| `GET /models/{id}` | Model state, activation ID, last error. |
| `POST /admin/models/{id}/unload` | Local admin control: drain requests, terminate worker, verify GPU release. |
| Existing capture CLI | Publish an idle, warmed snapshot before trials; record build time and size. |

Fresh versus snapshot is server configuration; clients send identical requests.
Later requests reuse the ready model.

```text
ABSENT --request--> ACTIVATING --verified--> READY --unload--> ABSENT
                         |--failure--> FAILED --cleanup--> ABSENT
READY --request--> generate --done--> READY
```

One activation per model; one generation at a time; bounded waiting queue.
Queue time counts; overflow is rejected. Deadlines/disconnections cancel owned
work safely. Failed cleanup blocks reuse; restore failure never silently loads
fresh weights. Concurrent arrivals are tested separately from latency trials.

## Files and interfaces

Paths are under `experiments/inference/` unless qualified. Interfaces below are
proposed; existing CLI behavior remains available.

| File | Responsibility / reuse |
| --- | --- |
| **New `api.py`** | Thin HTTP adapter; call `Runtime.generate(request, deadline)` and flush token events immediately. No CUDA state. |
| **New `runtime.py`** | `ensure_ready`, `generate`, `unload`, `status`. Extract activation/admission/ownership from `run.py`; API and CLI share it. |
| `worker.py` | Reuse `load`, `render`, `generate`, `inspect`. Replace two-request loop with request loop and per-token callback; tokenize HTTP prompts and clear per-request KV cache. |
| `lifecycle.py`, `park.py` | Reuse capture/restore, process identities, cleanup, GPU admission, and resource sampling. |
| `../finetuning/{control,checkpoint,restore}.py`, `../criu/session.py` | Reuse integrity, publication, CRIU, clocks, and process ownership; no second restore implementation. |
| **New `bench_http.py`** | Reset state, verify cache condition, send requests, timestamp token receipt, check output and follow-up request. Reuse prepared references and `cold_cache.py`. |
| `measure.py`, `report.py`, `disk_profile.py` | HTTP evidence schema, stage spans, failures, paired comparisons. Reject pooling HTTP and historical CLI timings. |
| `stream_weights.py` | Optional candidate, enabled only if profiling justifies it. |
| `Dockerfile`, `README.md` | Package the same service and document endpoint commands after host validation. |

Initially reuse file messages between controller and worker; measure their
polling/write overhead. Add activation/request identities, fresh response paths,
and a restore handshake to reject saved messages. Capture only at idle, with no
request or KV cache; keep the HTTP connection outside the restored process.

## Measurements and conditions

| Measure | Boundary / evidence |
| --- | --- |
| Primary TTFT | Client monotonic clock: before connection/request to first token received. Fixed connection policy; no automatic retries. |
| Request phases | Receive, queue/admission, activation, tokenization, first forward pass, token flush. Correlate IDs; do not subtract timestamps from different clock domains. |
| Fresh activation | Import/construction, disk reads, tensor setup, host-to-GPU copies, CUDA initialization. Detailed tracing only in diagnostics. |
| Snapshot activation | Dependency/payload validation, CRIU reconstruction, post-restore readiness; reuse host-observer events. |
| Correctness / costs | Exact tokens and weights, later request success, errors/timeouts, sampled CPU/GPU peaks, retained memory, snapshot size/build time, staging cost. |

Show overlapping reads/copies as timeline spans. Exclusive phases plus an
unattributed remainder reconcile server time; compare client/server durations
without assuming synchronized clocks. All request-path checks remain timed.

- **Fixed workload:** pinned Qwen3-8B, BF16, A10G, batch one, greedy decoding.
  One fixed timing prompt; a fixed prompt set for diagnostics. Record software,
  source/artifact hashes, storage, settings, and timeout policy.
- **Primary pair:** fresh versus snapshot on the same existing EBS disk. No model
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

## Improve one measured bottleneck

| Dominant cost | Intervention | Controlled comparison |
| --- | --- | --- |
| Disk reads | Stage on local instance NVMe. | Move both routes; change only storage. Report staging cost and ephemeral lifetime. |
| Validation reads/cache locality | Dependency-first validation, then optional concurrency. | Preserve every hash/check; change order and worker count separately. |
| Inefficient weight loading | ServerlessLLM-inspired bulk reads, pinned buffers, overlap, direct I/O. | Add one step at a time against the installed Transformers default. |
| Initialization/restore work | Refine the idle snapshot boundary to retain reusable initialized state. | Count capture cost; require identical later responses and empty request state. |

Reuse exploratory code only as candidates. Optimize transport only if measured
overhead matters. [Published methods](../research/inference-cold-start.md) guide
the choice; a snapshot win remains a hypothesis.

## Milestones

1. **Endpoint boundary:** shared runtime, API, worker loop. Test streaming,
   stale-message rejection, single activation, timeout/disconnect cleanup, and
   restore failure without serving invalid state.
2. **Baseline:** HTTP evidence, cache controls, correctness gate, three timing
   pairs. Identify the bottleneck before selecting an optimization.
3. **One improvement:** isolated change and matched repeats; publish TTFT,
   correctness, costs, and limitations even if the result is negative.
4. **Container replay:** reuse the validated container profile; repeat diagnostics
   and baseline-versus-best HTTP measurements. Do not pool host/container results.

Use the writing-tests skill before test changes; commit small milestones. The
user reviews and merges. Implementation resumes after this plan is reviewed.
