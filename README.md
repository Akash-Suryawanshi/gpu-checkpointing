# Learning CPU and GPU checkpointing

This experiment captures a running LoRA trainer as files, ends the original process,
and restores it to continue training. The scope is one Linux host, one process,
and one NVIDIA GPU. The docs assume basic programming; OS concepts are introduced
where needed.

## Start with the machine

The **CPU** runs Python and submits numerical work to the **GPU**. CPU-side objects
live in **RAM**; GPU tensors live in **VRAM**. **Storage** holds checkpoint files.
A **process** is a running program, including its memory and execution position.
Linux's **kernel** manages these resources; NVIDIA's **driver** manages the GPU.

![GPU data is staged in host RAM, written through the Linux page cache, and synchronized to storage.](docs/assets/snapshot-memory-storage.svg)

*Arrows show the data path. RAM and VRAM are live memory; file visibility alone
does not prove that writes reached storage.*

## Why model weights are only part of the state

Training also depends on optimizer history, random-generator state, input position,
and learning-rate schedule. An **application checkpoint** saves selected values
and loads them into a fresh trainer. A **process snapshot** reconstructs the
running process and supported resources. Neither automatically restores external
files or services.

## Reading path

| Read | Purpose |
| --- | --- |
| [CPU checkpointing](docs/01-cpu-checkpointing.md) | Memory, execution state, Linux resources, and CRIU restoration. |
| [GPU checkpointing](docs/02-gpu-checkpointing.md) | CUDA staging, CPU/GPU coordination, limitations, and cost. |
| [Run the experiment](experiments/finetuning/README.md) | Setup, commands, code walkthrough, and reporting. |
| [Results](experiments/results.md) | Dated evidence, measurements, and qualifications. |
| [Workload contract](docs/implementation-plan.md) | Implemented configuration and acceptance requirements. |
| [Next lifecycle plan](docs/independent-lifecycle-plan.md) | Proposed independent trainer, capture, and restore commands. |
| [Inference snapshot plan](docs/inference-snapshot-plan.md) | Planned inference comparisons, implementation gates, and scaling. |

The [research note](research/single-gpu-finetuning.md) compares approaches; the
[teaching plan](docs/readability-plan.md) tracks remaining explanatory work.

## What has been observed

On EC2 A10G, application restart and CRIU restore matched uninterrupted LoRA
training, including active dropout and repeated capture. The September 17
[validation](experiments/results.md#readability-refactor-validation--2026-09-17)
passed after the readability refactor. CRIU resumed faster but saved more slowly
and produced larger snapshots in the recorded
[comparison](experiments/results.md#four-update-lora-acceptance-and-timing--2026-09-14).

These are qualified same-host results: shared-resource warnings remain unresolved.
Spot recovery, replacement-host restoration, and inference cold starts are untested.
Historical container results are recorded separately.
