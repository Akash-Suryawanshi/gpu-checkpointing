# Plan: vLLM snapshot versus cold start

**Implemented; gates not yet run.** Test one claim: for an engine whose startup
compiles kernels and captures CUDA graphs, restoring a saved process beats
starting cold. Our own stack could not show this, because its startup work is free.

This plan owns the vLLM comparison. The
[cold-start plan](inference-cold-start-plan.md) owns the Transformers routes and
the cold-cache method; [results](../experiments/results.md) own the evidence.

## Why this engine, when ours showed nothing

Our loader spends its activation reading weights, and a snapshot must read at
least as many bytes back. A serving engine adds startup work that produces no
file at all: compiled kernels and captured CUDA graphs. That derived state is
what an image can carry and a fresh start must recompute.

```text
our stack     [ read weights 53 s ][build ~0]        <- nothing to win
vLLM cold     [ load weights     ][ compile + capture graphs ]
vLLM snapshot [ read image, rebuild process ]
                                  ^^^^^^^^^^^^^^^^^^^^ removed by the snapshot
```

Bars are activation time. The bet is that the compile-and-capture segment is
worth more than the extra bytes a larger image costs to read back.

## Routes

| Route | Activation | What it isolates |
| --- | --- | --- |
| `cold` | launch vLLM, load weights, compile, capture graphs | the real production cold start |
| `snapshot` | restore a published image of a warmed worker | the same endpoint with initialization already done |
| `eager` | launch vLLM with graph capture disabled | how much of the cold start is the compile step |

The `eager` control makes the result explanatory rather than a single number. If
`cold - eager` is large, the compile step is the cost and the snapshot should
recover roughly that much. If it is small, vLLM behaves like our stack, and the
negative result extends to production engines, which is also worth knowing.

## Acceptance

**Proved if** the snapshot median is below the cold median in all three blocks,
with identical output tokens, and the gap is close to the measured
compile-and-capture time.

**Disproved if** the snapshot is at or above cold. The image will exceed the
weights and both volumes cap near 0.32 GB/s, so this is a live outcome and the
byte accounting has to explain it.

**Blocked if** capture refuses vLLM. Stock CRIU already rejects compiled kernels
with `handle_device_vma plugin failed`; that is a clean finding about stock
tooling, not a failed experiment.

## Gates

| Gate | Passes when | Stop if | Budget |
| --- | --- | --- | --- |
| 1 | vLLM installs isolated and serves one token on this A10G | no build works against driver 570 / CUDA 12.8 | 45 min |
| 2 | a warmed worker is captured and restored once, output identical | capture refuses the device mappings | 60 min |
| 3 | cold, eager and snapshot each pass one diagnostic | output differs between routes | 45 min |
| 4 | three timing blocks, cold cache, per route | — | 90 min |

Gate 2 decides the project and runs on Qwen2.5-0.5B first, where a cycle is
minutes; the headline uses Qwen3-8B. Report at each gate, not at the end.

## Conditions held fixed

- One A10G, 23 GiB. Greedy decoding, fixed prompt, 16 new tokens, with the
  reference tokens produced by vLLM itself.
- `--gpu-memory-utilization` low, so the KV cache does not inflate the image. It
  enters the comparison key. NVIDIA's release-and-remap trick is out of scope.
- Cold model-file cache verified by `mincore` before every timer, as in the
  [cold-start plan](inference-cold-start-plan.md).
- Local flash only: EBS has 19 GiB free, which will not hold another image.
- Three blocks, alternating order, every attempt reported including failures.

## Files

New `experiments/vllm/` rather than an extension of `experiments/inference/`,
because vLLM needs its own Python environment and mixing them would drag a
second engine into every existing comparison key.

| File | Role |
| --- | --- |
| `experiments/vllm/engine.py` | the worker: start an engine, warm it, publish an idle marker, serve file requests |
| `experiments/vllm/trial.py` | the controller: one route per run, eviction, capture or launch, timing, result |
| `experiments/vllm/assets.py` | pin the revision, render one prompt, record vLLM's own reference tokens |
| `experiments/inference/measure.py` | `validate_vllm()` admits these records |
| `tests/test_vllm_trial.py` | no GPU: route settings reach the engine, KV release precedes capture, reference mismatch fails |

Capture, restore, process ownership, cold-cache eviction and device counters are
reused unchanged from `experiments/inference/`. Preparation is `assets.py`, not
`prepare.py`: `prepare.py` and `prepare_assets.py` already exist in the other
experiment directories, and a bare import would silently select the wrong one.

Two conditions are recorded per campaign rather than assumed. `--data-cache`
evicts the model files as the cold-start work does. `--compile-cache` says
whether vLLM's compiled kernels are already on disk: `warm` matches a production
restart and is refused when the cache is empty, `cold` makes each launch compile
its own.
