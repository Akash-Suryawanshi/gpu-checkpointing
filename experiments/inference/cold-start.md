# Learning disk-to-first-token latency

The [measurement plan](../../docs/inference-cold-start-plan.md) defines fair
comparisons; the [research note](../../research/inference-cold-start.md) connects
the implementation to existing methods.

The [original comparison](../results.md#inference-activation-and-gpu-reuse--2026-09-17)
left file-cache residency uncontrolled. These experiments explicitly evict the
selected weight and image files before starting the activation timer; the
[cold EBS baseline](../results.md#cold-cache-activation-on-ebs--2026-09-18) records the result.

```text
disk weights -> 4 readers + SHA-256 -> 4 pinned buffers -> GPU tensor views
                   read next chunks <---- overlap ----> copy earlier chunks
new Python process -> construct empty model -> attach tensors -> first token
```

Arrows show data movement and startup sequence. Pinned memory stays at a stable
host address during GPU transfer; a CUDA completion event permits buffer reuse.
Four 16 MiB buffers bound the pipeline to 64 MiB of pinned host memory.

## What is being compared

| Loader | State retained before launch | Technique |
| --- | --- | --- |
| `transformers` | Local safetensors files | Installed Transformers default, including its asynchronous loading |
| `packed` | Flat BF16 file + chunk manifest | One reader, pageable buffer, blocking GPU copies |
| `pipelined` | Same flat file + manifest | Four readers, pinned buffers, overlapping reads/hashes/copies |
| `direct` | Same flat file + manifest | Same pipeline with aligned `O_DIRECT` reads that bypass the OS file cache |
| `disk` route | Published CRIU image | Full-process restore; optional dependency-first validation |

The packed loaders are a bounded adaptation of
[ServerlessLLM §4 (OSDI 2024)](https://www.usenix.org/system/files/osdi24-fu.pdf):
bulk layout, memory pooling, concurrent reads, and a loading pipeline. This is
not a reproduction of its scheduler, model manager, or migration. Direct reads
use aligned requests, accept a verified short EOF tail, and fail explicitly
when the buffers or filesystem are unsuitable.

The process starts anew for all four loaders. The packed format preserves BF16
weights exactly; each chunk must pass SHA-256 before its GPU copy. Full model
fingerprints and both generated responses must match the prepared reference.

## Commands

Use the [runbook](README.md#prepare) environment, an idle GPU, and a disk-backed
`BASE` on the storage being measured. Prepare assets for that exact model path;
the packed manifest is bound to the prepared asset manifest.

```bash
"$PY" experiments/inference/stream_weights.py \
  --assets "$ASSETS" --output "$BASE/packed-v1"

# Run each loader's diagnostic before three timing trials. Keep separate runs.
"$PY" experiments/inference/run.py trial --route fresh --kind diagnostic \
  --loader pipelined --bundle "$BASE/packed-v1" --data-cache cold \
  --assets "$ASSETS" --output "$BASE/pipelined/diagnostics/fresh" \
  --tools "$TOOLS" --timeout 900
"$PY" experiments/inference/run.py trial --route fresh --kind timing \
  --loader pipelined --bundle "$BASE/packed-v1" --data-cache cold \
  --assets "$ASSETS" --output "$BASE/pipelined/timing/b1-fresh" \
  --validated-run "$BASE/pipelined/diagnostics/fresh" --block 1 \
  --tools "$TOOLS" --timeout 900
```

Repeat with `--loader packed` and `--loader direct`; for `transformers`, omit `--bundle`.
Use three blocks with rotated loader order and the schedule/report convention in
the runbook. Do not reuse diagnostics across loader, storage, cache, or code changes.

For CRIU, use the existing `--route disk` commands with `--data-cache cold`.
Compare defaults against `--validation-order dependencies-first
--validation-workers 4`; all dependency and payload checks remain enabled.
Run `"$PY" experiments/inference/disk_profile.py RUN` to locate time before,
inside, and after CRIU using host-observer events.

## Measurement boundary

```text
prepare/copy files + validate assets -> evict selected files -> START
  fresh: launch + import + construct + load + request -------> first token
  disk:  validate snapshot/dependencies + CRIU + request ----> first token
```

Eviction uses per-file `POSIX_FADV_DONTNEED`; `mincore` must report zero resident
pages. It does not drop the machine's global caches. Runtime/library caches,
filesystem metadata, and device-internal caches are not controlled.

`io.json` records kernel-accounted device reads between activation start and the
first token, next to the logical bytes hashed. Device counters include any other
reader, so measure on an otherwise idle disk; hashing a cached file adds logical
bytes without device reads.

The EBS data volume is gp3 with 125 MiB/s provisioned throughput and 3,000 IOPS
(AWS `describe-volumes`, 2026-09-18); the instance store is a g5.2xlarge local NVMe.

Preparation, deployment to local storage, and eviction are outside activation.
Fresh retains its original preflight boundary; the packed loader additionally
checks chunks during activation, while CRIU revalidates dependencies and images.
Report these integrity costs, rather than subtracting them from TTFT.

Local instance storage is a deployment cache: it is lost when an EC2 instance
stops or terminates. Keep durable source artifacts elsewhere and count staging
when measuring replacement-host readiness; see
[AWS persistence semantics](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-store-lifetime.html).
