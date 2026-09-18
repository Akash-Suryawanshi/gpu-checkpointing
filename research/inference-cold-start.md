# Existing approaches to disk-to-first-token latency

Time to first token (TTFT) includes loading weights and making the model ready
to execute. A fast restore API alone does not establish a fast cold start.

```text
local disk -> read bytes -> host memory -> GPU -> initialized model -> token
              streaming / pipelining       ^
saved process -> validation -> restore ----+  skips some initialization
```

Arrows show data movement and execution order. Keeping bytes in host or GPU
memory changes the starting state; compare that separately from a disk start.

## Methods closest to this experiment

[ServerlessLLM (OSDI 2024), §4](https://www.usenix.org/system/files/osdi24-fu.pdf)
combines a loading-oriented format, large reads, direct I/O, concurrent readers,
pinned memory, and a pipeline across storage tiers. Pinned memory cannot be
moved by the operating system while the GPU reads it; reusable buffers avoid
allocating host memory for the entire model.

Our [local adaptation](../experiments/inference/cold-start.md) implements that
bounded loading path with four 16 MiB buffers and per-chunk integrity checks.
It does not reproduce ServerlessLLM's distributed scheduler or live migration.

[fastsafetensors (IEEE CLOUD 2025)](https://arxiv.org/abs/2505.23072)
copies groups of parameters to device memory before creating tensor views.
This avoids constructing and copying each host tensor separately. Its
[implementation](https://github.com/foundation-model-stack/fastsafetensors/blob/main/docs/overview.md)
supports GPUDirect Storage and an explicit host-buffer path when that facility
is unavailable; either path needs its actual execution mode recorded.

[Dynamo's fast-loading guide](https://docs.nvidia.com/dynamo/dev/kubernetes/model-deployment/model-loading/fast-model-loading)
uses `runai_streamer` for concurrent large reads in vLLM. It also recommends
spreading files over multiple storage targets on Lustre, a parallel filesystem;
that layout advice does not apply to our single local ext4 volume.

These approaches address storage access and data transfer. Enabling Dynamo
request routing or separating prompt processing from token generation does not,
by itself, implement a faster disk loader.

## Process restoration is a different intervention

[GCR (FAST 2026)](https://www.usenix.org/conference/fast26/presentation/zeng)
and [PhoenixOS (SOSP 2025)](https://github.com/SJTU-IPADS/PhoenixOS)
study GPU process checkpoint and restore. They are relevant when restoring GPU
execution state dominates, but adopting their runtimes is a separate compatibility
experiment, not a loader flag for our existing CRIU pipeline.

For this repository, first locate validation, CRIU, and post-restore time with
the [host-clock profiler](../experiments/inference/disk_profile.py). Then compare
the default validation order with dependency-first validation on the same storage;
all hashes remain checked, and the snapshot is the last large file read before
CRIU consumes it.

The papers' reported speedups describe their workloads and hardware. Our evidence
must include the cold-cache boundary, model precision, actual storage, and complete
first-token timing before assigning any speedup to these techniques.
