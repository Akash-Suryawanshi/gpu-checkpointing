# Learning 8B disk cold starts

Goal: measure disk-to-first-token time, identify its dominant cost, and test an
intervention without changing the model's output. A snapshot win is a hypothesis;
a slower restore with an explained bottleneck is also a useful result.

```text
cold-file baseline -> locate cost -> change one factor -> repeat -> verify
                            |                             |
                    loader or restore              tokens + weights
```

The sequence separates diagnosis from comparison. The
[research note](../research/inference-cold-start.md) owns method attribution;
the [runbook](../experiments/inference/cold-start.md) owns commands and boundaries.

## Comparison contract

Start the host observer's timer immediately before process launch or the
independent restore helper. End it when the observer receives the first token;
include validation performed in that interval and never subtract inspection costs.

Call the controlled baseline **cold model-file cache**, not fully cold: the
machine, driver, libraries, filesystem metadata, and device caches can be warm.
Container creation, image pulls, model downloads, and deployment copies require
a separate measurement if they are part of the intended startup contract.

| Case | Required initial state | Purpose |
| --- | --- | --- |
| Disk-cold fresh | No model process; selected weight pages absent from OS cache | Primary baseline |
| Disk-cold snapshot | Original process ended; selected image and weight pages absent | Primary restore comparison |
| Warm file cache | No model process; explicitly verified cached weight pages | Storage control |
| Host-memory retained | Define whether weights or an entire suspended GPU process are retained | Transfer/control-plane reference |
| GPU resident | Initialized model and warmup state retained on GPU | Warm-request reference |

The existing RAM-parking route retains a living process and staged GPU state.
It is not a freshly deserialized CPU-only model. Historical uncontrolled-cache
runs cannot be relabeled as an explicitly warm-cache control.

## Attribution and correctness

Use a timeline for launch/import, file reads, tensor construction, GPU copies,
CUDA initialization, and first forward pass. Reads and copies can overlap;
exclusive wall-clock intervals plus unattributed gaps should reconstruct TTFT,
whereas summing overlapping spans can exceed it.

Detailed tracing belongs in separate diagnostic runs. Timing runs retain light
events and validate complete weight fingerprints and deterministic responses;
the health request must succeed after the first response.

For restore, report validation, CRIU, and post-CRIU time using host-observer
timestamps. Do not subtract timestamps from a restored process's different clock
domain or equate CUDA call submission with completed GPU work.

Compare the same pinned 8B revision, BF16 precision, prompt tokens, generation
settings, GPU, storage, software, and integrity policy. Quantization changes the
accuracy contract and needs a separate cohort; faster storage and loader changes
also need separate controls before attributing their individual effects.

## Evidence and interpretation

Require a matching diagnostic before each timing configuration. Rotate execution
order, keep failed attempts visible, and report individual runs, count, median,
and range for the initial three-block experiment.

Leave p95 unreported at this sample size. If a tail-latency claim is needed,
predeclare a larger campaign (for example 100 runs per finalist), the percentile
estimator, uncertainty, and failure treatment before collecting it.

Report snapshot creation time, image size, staging cost, retained CPU/GPU memory,
and storage lifetime alongside TTFT. The primary question is whether disk-cold
snapshot restoration beats disk-cold fresh loading on the same storage; the
streaming loader is an independent candidate for reducing that baseline.

Status: implementation and diagnostics are underway. Results belong in
[measured evidence](../experiments/results.md); unmeasured cases remain unmeasured.
