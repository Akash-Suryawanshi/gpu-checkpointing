# Every experiment, in one page

One line each: what it asked, what it answered. [Results](results.md) holds the
evidence and the qualifications; the runbooks hold the commands.

```text
tooling          training              inference activation        serving engine
CRIU? DMTCP?  -> LoRA capture   ->  4 routes, 8B  ->  cold cache,  ->  vLLM cold
cuda-checkpoint  and restore        fresh/resident     storage,         vs snapshot
                                    RAM/disk          loaders
2026-09-12/14    09-14/17           09-17            09-18            09-18
```

## Can a GPU process be checkpointed at all?

| Asked | Answered | Evidence |
| --- | --- | --- |
| Does packaged CRIU work in the container? | No: it fails creating a network namespace before any capture. | [CRIU capture](results.md#criu-process-capture--2026-09-14) |
| Does NVIDIA's `cuda-checkpoint` alone suffice? | It suspends and resumes GPU state, but writes no process image. | [GPU-only suspension](results.md#nvidia-gpu-only-suspension--2026-09-14) |
| Is DMTCP an alternative? | Yes for native CUDA; it reacquires the GPU after writing a checkpoint. | [DMTCP](results.md#dmtcp-native-cuda-investigation--2026-09-14) |
| Does CRIU's CUDA plugin work on the EC2 A10G? | Yes: CPU, GPU-tensor and LoRA processes all restored. This became the route. | [EC2 validation](results.md#ec2-criu-validation--2026-09-14) |

## Can LoRA training survive a snapshot?

| Asked | Answered | Evidence |
| --- | --- | --- |
| Does a captured trainer resume identically? | Yes, across four updates and both dropout settings. | [Acceptance](results.md#four-update-lora-acceptance-and-timing--2026-09-14) |
| Can capture own the trainer independently of the script that started it? | Yes: 30 CPU checks and all 15 GPU cases passed. | [Independent lifecycle](results.md#independent-lifecycle-validation--2026-09-17) |

## Is restoring a loaded model faster than loading it? (Qwen3-8B)

| Asked | Answered | Evidence |
| --- | --- | --- |
| How do fresh, resident, RAM-parked and disk-restored activation compare? | All four work; the disk route is the slow one. | [Four routes](results.md#inference-activation-and-gpu-reuse--2026-09-17) |
| With the file cache genuinely cold? | Both routes become storage-bound; EBS caps at 125 MiB/s. | [Cold cache](results.md#cold-cache-activation-on-ebs--2026-09-18) |
| Does faster storage rescue it? | No. Snapshot ÷ fresh is 3.03–3.10 on EBS and 2.85–3.03 on instance NVMe. | [Storage arm](results.md#storage-arm-the-same-comparison-on-instance-store-nvme--2026-09-18) |
| Is our software the bottleneck? | No: the raw devices cap at 0.13 and 0.32 GB/s whatever the concurrency. | [Bandwidth](results.md#both-volumes-are-bandwidth-limited-and-our-loaders-already-saturate-them--2026-09-18) |
| Does validation order matter? | Yes: hashing the payload last leaves it in cache, 164.10 s → 119.42 s. | [Order](results.md#validation-order-decides-whether-criu-reads-the-image-from-cache--2026-09-18) |
| At the snapshot's absolute floor? | 69.84 s against a 57.24 s fresh start: still 1.21–1.29×. | [Floor](results.md#the-snapshot-route-at-its-floor-still-trails-a-fresh-start--2026-09-18) |
| Does the restored process reopen the weights? | No: 92 files reopened, none of them a model file. | [File audit](results.md#what-a-restored-image-actually-reopens--2026-09-18) |
| Can a faster loader beat the default? | The packed data path wins, then loses it building the model serially. | [Loaders](results.md#packed-loaders-win-the-data-path-and-lose-it-again-to-serialization--2026-09-18) |
| Does an HTTP endpoint change the answer? | No, the request path adds nothing measurable. | [Endpoint](results.md#the-endpoint-boundary-costs-nothing-measurable--2026-09-18) |
| Does a smaller or compiled model help? | No. 0.5B is worse (17.95 s against 11.12 s); a compiled model cannot be captured. | [Probes](results.md#a-smaller-model-and-a-compiled-model-both-fail-to-help--2026-09-18) |
| Does it all reproduce in a container? | Yes, every host result. | [Container](results.md#the-container-reproduces-every-host-result--2026-09-18) |

## Does a serving engine change the answer? (vLLM, Qwen2.5-0.5B)

| Asked | Answered | Evidence |
| --- | --- | --- |
| Does vLLM have startup work a file cannot supply? | Yes: weights load in 0.17 s, `LLM()` takes 22.3 s. | [vLLM](results.md#vllm-a-real-compile-cost-and-a-snapshot-that-still-loses--2026-09-18) |
| Can a warmed vLLM worker be captured? | Yes. The blockers were PyTorch's io_uring ring and its self-connected socket, not CUDA. | Same |
| Is the snapshot faster? | No: 30.31 s against 27.65 s, reading 6.42 GiB against 0.92 GiB. | Same |

## What it adds up to

Restoring a process reliably reproduces GPU state. It is not faster here, because
every route we measured is limited by how fast this host reads bytes, and an image
is always larger than the weights. The one case with real startup computation to
save, vLLM, still lost on a 0.21 GiB/s volume — and NVIDIA's own product restores a
same-sized image in 2.4 s on striped NVMe with patched CRIU.

Open: repeat the vLLM comparison on hardware with faster storage. The
[plan](../docs/vllm-snapshot-plan.md) states the conditions, and the
[runbook](vllm/README.md#running-on-other-hardware) states what to re-derive.
