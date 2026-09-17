# Three-hour inference POC: latency versus retained resources

**Scope review, not new benchmark evidence.** Use the current A10G and downloaded
Qwen3-8B BF16. Keep [validated small-model recovery](../experiments/results.md#independent-lifecycle-validation--2026-09-17)
as fallback. No larger GPU or new training matrix in this time box.

## One experiment, four starting states

| Starting state | What remains while idle | What the next request measures |
| --- | --- | --- |
| Fresh process | Model files | Launch through first generated token. |
| Resident worker | CPU process and GPU allocations | Request through first token; GPU capacity remains occupied. |
| GPU state parked in RAM | Live CPU process and staged GPU bytes | GPU restore/unlock through first token; process/host loss loses parked state. |
| Published process image | Disk image and required dependencies | Fresh restore worker, verification, and first token after original exit. |

```text
GPU-resident model --NVIDIA checkpoint--> live process with staged RAM
                  --CRIU + persistence--> disk image; original process exits
RAM/disk state ----restore + request----> first token; compare output
```

Arrows show state transitions; the RAM and disk routes are separate experiments.
Call RAM activation a wake-up, not durable cold-start recovery. The disk route
establishes same-host process recovery, not replacement-host or volume-loss recovery.

Use the existing NVIDIA helper for RAM parking and the independent CRIU workers
for disk capture. At an idle boundary, discard request-specific attention caches.
The manual helper owns RAM transitions; CRIU's plugin owns disk-route transitions.
Never run both owners' restore/unlock operations on the same transition.

Show measured first-token latency, GPU memory released, peak/retained host RAM,
snapshot bytes, and park/capture preparation time. During parking, run a short
independent GPU job and then verify resumed output. Distinguish reused capacity
from cloud billing: releasing VRAM does not stop the VM's bill.

Use identical model/prompt/dtype and explicitly recorded storage-cache conditions.
Keep validation in recovery timing; use one observer clock, externally visible
first token, and three paired repetitions where time permits. Report median and
range; never claim p95 from three runs or promise a positive snapshot speedup.

## Production lesson and supporting evidence

Illustrate a small routing policy: reuse a resident worker; wake a parked worker
if host RAM and latency budget permit; otherwise restore an admitted image or
load normally. Present this as a proposed control policy, not a deployed scheduler.
Version images by model/runtime/driver compatibility and retain a reload fallback.

Reuse existing LoRA evidence: about 6.7 MB application state versus 3.7 GB process
images on the validated 0.5B workload. Frozen weights explain why a rank-2/rank-4
sweep adds less insight than the resource/latency comparison. Treat 70B as an
explicit estimate; an eviction-window calculation is supplementary, not a new demo.

## Reuse research, avoid a new systems integration

- [NVIDIA](https://github.com/NVIDIA/cuda-checkpoint): installed GPU-to-RAM staging mechanism; CPU process remains alive.
- [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/): production-facing analogue; level 1 offloads weights and discards attention cache. Do not install a new serving stack merely for this comparison.
- [Modal snapshots](https://modal.com/blog/truly-serverless-gpus): initialized-worker snapshots as a serving pattern; vendor performance is not our result.
- [GCR](https://www.usenix.org/system/files/fast26-zeng.pdf): separate control/data costs and avoid copying unchanged memory; use its ideas to explain bottlenecks, not claim we implemented it.
- [PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) and [CRIU compression](https://radostin.io/files/stoyanov-euromlsys-2026.pdf): defer integrations. Smaller images alone do not establish lower interruption latency.

## Hard time boxes

0–20 min: verify fresh/resident 8B inference and RAM headroom. If blocked, use the
validated small model for this comparison and label its size plainly.
20–60 min: RAM parking/wake-up and functional GPU reuse.
60–110 min: disk capture/restore; stop new debugging at this boundary.
110–150 min: repeat working routes, compare outputs, and make two evidence plots.
150–180 min: rehearse the five-minute narrative and record a backup.

The presentation contains one live state transition, a latency/resource table,
and a production-policy diagram. No new rank sweep, quantization stack, 70B run,
compression implementation, distributed restore, or full dashboard.
