# Single-GPU fine-tuning snapshots

**Single-GPU fine-tuning is a sensible next experiment. The checkpoint mechanism stays broadly the same; the training state, compatibility checks, and economic comparison become more demanding.** Start with a small pretrained model, ordinary LoRA, and a non-paged optimizer. Compare restoring the whole process with restarting from a complete training checkpoint.

Companies already implement GPU snapshots. NVIDIA supplies the driver mechanism; MemVerge and Cedana document workload checkpoint/restore; Modal, Beam, Google Cloud, and NVIDIA Dynamo expose related startup features. Their documented scope varies, especially between resuming training and starting an initialized inference worker. The company comparison below makes that distinction explicit.

This report covers language-model supervised fine-tuning: learning from examples with desired outputs, on one NVIDIA GPU. It compares full fine-tuning, LoRA, and quantized LoRA. Recommendations are proposed experiments, not measurements from the Jarvis instance. Product documentation was checked on September 13, 2026; individual papers and version constraints are identified below.

## 1. What changes from the tiny training POC

The current design has one Python process, a small neural network, deterministic inputs, and a checkpoint after a completed update. Fine-tuning replaces the little network with a pretrained language model and replaces synthetic numbers with tokenized text. CUDA still sees allocations, kernels, and streams; it does not need a special “fine-tuning snapshot” operation. The driver prepares supported GPU state for capture; a CPU checkpoint system preserves the process around it. [NVIDIA cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint), [CRIU GPU integration](https://www.criu.org/GPU_Checkpointing).

```mermaid
flowchart TB
    A["Pretrained model + training examples"] --> B["Python training loop"]
    B --> C["GPU: weights, gradients, optimizer tensors"]
    B --> D["CPU: next example, counters, random state"]
    C --> E["NVIDIA prepares GPU state in host memory"]
    D --> F["CRIU saves the Linux process"]
    E --> F
    F --> G["Restore the process and GPU state"]
    G --> H["Perform the next training update"]
```

The important change is the amount and meaning of the state. A model can produce sensible text after an incomplete restore while its training trajectory has already changed.

| State to preserve | Why it matters after resume |
| --- | --- |
| Trainable weights | They contain the learning completed so far. |
| Optimizer state | Adam keeps moving averages of gradients and squared gradients. Losing them changes subsequent updates. |
| Learning-rate schedule and update count | They determine the size of the next update. |
| Random-generator states | They determine dropout and other random choices. Preserve each generator actually used. |
| Dataset order and next example | Restarting at the wrong position repeats or skips training examples. |
| Partially accumulated gradients | Needed if the snapshot occurs before a complete optimizer update. Avoid this boundary initially. |
| Precision-related state | FP16 training may use a gradient scaler; quantized weights have representation metadata. Include these when present. |
| Python objects and CUDA resources | A process snapshot preserves execution machinery that an application restart normally rebuilds. |

Framework support already exists for much of the explicit training state. Accelerate saves models, optimizers, random generators, and gradient scalers, and supports registering additional stateful objects. Dataset position still needs deliberate handling. [Accelerate checkpointing](https://huggingface.co/docs/accelerate/usage_guides/checkpoint).

**One GPU does not automatically mean one process.** Data loading, experiment tracking, and compilation can introduce workers or external services. Our proposed baseline uses already-tokenized local examples, no data-loader worker processes, and local logs. PyTorch documents the distinction between loading data in the main process and using multiple workers. [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html).

### Choose a complete update as the pause point

A training update is: compute predictions, compute the loss, run the backward pass, update weights, update the learning-rate schedule if applicable, and clear gradients. With gradient accumulation, several small batches contribute to one update. The checkpoint boundary should come after that whole update.

```mermaid
flowchart LR
    A["Forward + backward"] --> B["Optimizer update"]
    B --> C["Schedule + clear gradients"]
    C --> D["Finish GPU work"]
    D --> E["Record state and wait"]
```

At the boundary, use an explicit readiness/release handshake, rather than a short sleep. The shell must see readiness; the training process must wait until restore is finished before proceeding. Drop references to temporary outputs where appropriate. A completed backward pass does not mean every allocated byte has been returned to the driver: PyTorch retains reusable memory in its caching allocator. Measure both allocated and reserved memory. [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management).

This is **a cooperative pause point with transparent state capture**. It deliberately makes correctness easy to inspect. It does not establish that every arbitrary interruption point works.

## 2. Full fine-tuning, LoRA, and QLoRA

**Full fine-tuning** changes the base model's weights. **LoRA** freezes those weights and learns small additional matrices, called adapters. The original model still participates in the computation and still occupies memory. [Hu et al., LoRA, 2021](https://arxiv.org/abs/2106.09685).

**QLoRA** combines a low-bit representation of the frozen base model with trainable adapters. Its original design also introduced paged optimizers to handle memory pressure. Quantization and optimizer paging are distinct choices in an implementation. [Dettmers et al., QLoRA, 2023](https://arxiv.org/abs/2305.14314).

| Method | What an application checkpoint can save | What a basic process snapshot must account for |
| --- | --- | --- |
| Full fine-tuning | Updated model, optimizer, and other training state | The same learning state, plus the process and CUDA environment |
| LoRA | Adapters and their optimizer state, other training state, and a reference to the fixed base model | The resident base model as well as adapters and process state |
| QLoRA | Adapters, optimizer and other training state; exact base/quantization configuration | Quantized resident base, its metadata, adapters, runtime state, and any supported allocations |

PEFT's adapter export intentionally omits the base model. That is useful for a compact artifact, but an adapter export alone is not a full training-resume checkpoint: optimizer, schedule, data position, and random state need their own handling. [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).

### The central tradeoff for LoRA

Imagine an unchanged 7-billion-parameter base represented with two bytes per parameter. Its raw weights alone are approximately **14 GB**, before temporary memory or runtime overhead. Four bits per parameter would be approximately **3.5 GB**, before quantization metadata and higher-precision components. These are arithmetic illustrations, not observed memory use or model fit guarantees.

An application knows that the base can be loaded again from a fixed model revision. A general snapshot mechanism sees live allocations; it cannot infer the same saving from `requires_grad=False` without additional support. Consequently, the portion we train may be small while the portion we snapshot remains large. **LoRA can make application checkpointing more attractive even while making the training job itself cheaper.** This is an analytical consequence of the different saved state, not a claim that snapshots always lose.

```mermaid
flowchart TB
    A["LoRA process in memory"] --> B["Large frozen base"]
    A --> C["Small changing adapters + training state"]
    B --> D["Full snapshot: preserve resident state"]
    C --> D
    C --> E["Application save: record changes + progress"]
    F["Fixed base model on storage"] --> G["Application restart reloads base"]
    E --> G
```

For full fine-tuning, application checkpoints also become large because all model weights change and optimizer state grows. That narrows this particular size advantage. Actual pause and resume times still need measurement.

## 3. Compatibility issues that matter first

### Paged optimizers are the most concrete QLoRA trap

bitsandbytes documents paged optimizers as using CUDA unified memory, which lets memory be managed across the CPU and GPU. Its implementation calls `cudaMallocManaged` for paged buffers. NVIDIA's current checkpoint utility lists UVM as unsupported. This creates a specific incompatibility to avoid in our first experiment. [bitsandbytes optimizer explanation](https://huggingface.co/docs/bitsandbytes/explanations/optimizers), [NVIDIA limitations](https://github.com/NVIDIA/cuda-checkpoint#functionality).

The code path is inspectable: `get_state_buffer()` chooses paged buffers when paging is enabled and a tensor meets its size threshold; the native allocator uses `cudaMallocManaged`. Smaller tensors can take the ordinary allocation path. Probe after optimizer state is initialized, not merely after loading the model. Managed allocation can exist before any visible paging under memory pressure. [optimizer.py, get_state_buffer](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py#L339-L355), [pythonInterface.cpp, cget_managed_ptr](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/pythonInterface.cpp#L501-L504).

**Recommendation:** start with plain `torch.optim.AdamW` over the trainable adapter parameters. Later, add four-bit base weights while retaining a non-paged optimizer. A successful ordinary LoRA run does not prove bitsandbytes compatibility; a paged-optimizer failure does not prove all QLoRA is impossible.

### Other boundaries

| Concern | First-experiment choice and reason |
| --- | --- |
| CUDA work still running | Wait at the completed update boundary. Driver checkpointing drains submitted work; it does not save a kernel halfway through execution. |
| CPU keeps advancing | Hold an explicit release gate until the restored CUDA state is ready. CUDA suspension alone is not CPU process freezing. |
| Host RAM | Leave room for the active CPU process plus staged GPU memory and checkpoint overhead; GPU fit alone is insufficient. |
| Software and hardware | Restore on the same host and GPU with the same environment. Cross-host portability is a separate experiment. |
| Compilation and optimized kernels | Start with eager execution; introduce compilation or additional kernel libraries one at a time after the baseline. |
| External state | Keep data and logs local and controlled. Restoring memory cannot roll back an external upload or reset another service's clock. |

The driver behavior above is described by NVIDIA; the experiment restrictions are our scope choices, not claims that all excluded features are fundamentally unsupported. [NVIDIA execution model](https://github.com/NVIDIA/cuda-checkpoint#the-utility).

**“Gradient checkpointing” means something else.** It saves training memory by discarding selected intermediate results and recomputing them during the backward pass. It does not create a restartable process snapshot. We can add it later if memory pressure requires it. [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html).

## 4. Papers most relevant to this project

These papers answer different questions. Training experiments establish more than a model-loading demo, but neither automatically establishes our precise LoRA or QLoRA configuration.

### CRIUgpu — closest to our implementation path

Stoyanov et al., February 2025, describe integrating vendor GPU capture with CRIU, including language-model training. Their single-H100 Llama 3.1 8B experiment reports **77.40 seconds to checkpoint, 38.83 seconds to restore, and a 55.89 GB unified snapshot**. The setup used an older driver and HDD storage; these figures are not predictions for our L4. The paper does not establish a tested LoRA/QLoRA recipe.

Read its workflow, Table 2, and Table 4. Its practical lesson is that saving GPU state to host memory and writing a complete process snapshot are different costs. The implementation path leads to upstream CRIU's CUDA plugin, rather than a new fine-tuning framework. [CRIUgpu paper](https://arxiv.org/html/2502.16631v1), [upstream CUDA plugin](https://github.com/checkpoint-restore/criu/tree/criu-dev/plugins/cuda).

### CRIU-LZ4 — direct fine-tuning evidence, with a useful negative result

Stoyanov et al., EuroMLSys 2026, explicitly evaluate supervised fine-tuning on a single B200. Section 5.3 reports training snapshots shrinking by roughly **1.1–1.6×**, while checkpoint freeze time grows **2–3×** and restore time grows **1.5–1.7×**. Training state compresses less effectively than empty inference caches. The exact adapter/quantization recipe is not specified sufficiently to reproduce our proposed workload.

This is a strong reason to measure compression rather than assume it helps. Current upstream `criu-dev` documentation includes LZ4 compression options; availability must be checked in the actual installed build. [CRIU-LZ4 paper, §5.3 and Figure 9](https://radostin.io/files/stoyanov-euromlsys-2026.pdf), [CRIU compression options](https://github.com/checkpoint-restore/criu/blob/criu-dev/Documentation/criu.txt).

### GCR — the most useful performance paper to read next

Zeng et al., FAST 2026, combine driver-managed control state with separately managed data copying. Their incremental design tracks modified memory, allowing unchanged buffers to be skipped. They report **72.1% lower checkpoint latency** against their NVIDIA baseline; the often-cited **86.6% size reduction concerns inference experiments**. Their implementation stores checkpoints in CPU memory, so these results are not durable disk-checkpoint timings.

Read §§5, 6.1.3, and 6.2 together. The public artifact uses a tested two-A100 environment and specific framework versions. It is a research implementation, not evidence that our modern single-GPU LoRA stack works unchanged. [GCR paper](https://www.usenix.org/system/files/fast26-zeng.pdf), [FAST paper and artifact recognition](https://www.usenix.org/conference/fast26/presentation/zeng), [GCR source and evaluation](https://github.com/thustorage/GCR).

### PhoenixOS — copying while execution continues

Wei et al., SOSP 2025, study overlapping checkpoint/restore with application execution. Their system infers which memory operations matter and validates those inferences. The public Apache-2.0 repository explicitly supports single-GPU checkpoint/restore and describes substantial build machinery and ongoing development.

This is relevant once stopping to copy memory becomes the measured problem. It would add considerable systems work to our first demo. [PhoenixOS paper, September 2025 revision](https://arxiv.org/html/2405.12079v4), [PhoenixOS source and support notes](https://github.com/SJTU-IPADS/PhoenixOS).

### Singularity — why a cloud provider builds this

Shukla et al., Microsoft, February 2022, present a scheduling service that makes deep-learning jobs preemptible, migratable, and resizable through a device-proxy architecture. It connects checkpoint mechanisms to better fleet utilization and reliability.

It is valuable background for the Jarvis interview, but the publication is not a downloadable, supported single-GPU fine-tuning package. Its 2022 description also does not establish the exact current scope of an Azure product. [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/singularity-planet-scale-preemptive-and-elastic-scheduling-of-ai-workloads/), [Singularity paper](https://arxiv.org/abs/2202.07848).

### CRAC — evidence that driver limitations are not universal laws

Jain and Cooperman, SC 2020, describe a DMTCP-based design supporting CUDA streams and unified memory. It uses a different architecture from NVIDIA's driver-native utility. The public early-development repository warns of incomplete portability work. It is historical context, not a low-effort fix for today's QLoRA UVM restriction. [CRAC paper](https://arxiv.org/abs/2008.10596), [CRAC source and caveats](https://github.com/DMTCP-CRAC/CRAC-early-development).

The original **LoRA** and **QLoRA** papers explain how training state becomes smaller; they do not claim transparent CPU/GPU process restoration. Keep those research questions separate. [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314).

## 5. Open-source tools and frameworks

There are two layers to choose: the code that trains the model, and the code that captures the running process. A training framework with a `resume` option does not thereby preserve a CUDA process.

| Tool | What it provides | Fit for this POC |
| --- | --- | --- |
| [cuda-checkpoint + CRIU CUDA plugin](https://www.criu.org/GPU_Checkpointing) | Supported GPU state capture combined with Linux process capture. The CLI/plugin are public; GPU internals remain in NVIDIA's driver. | **First choice for transparent capture.** |
| [Transformers + PEFT](https://huggingface.co/docs/peft/developer_guides/checkpoint) | Pretrained models, adapters, and explicit model/adaptor saving. | **First choice for a readable training loop.** |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) | A supervised fine-tuning trainer built on Transformers Trainer, including training resume. | Add after the explicit loop, to check a common framework path. |
| [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/checkpoint) | Application-state save/load and registration of custom state. | Useful reference or implementation for the application baseline. |
| [torchtune](https://meta-pytorch.org/torchtune/stable/deep_dives/checkpointer.html) | Full and LoRA fine-tuning recipes with model and recipe-state checkpoints. | Another application-level comparator; the cited 0.6 guide describes epoch-end intermediate saves. |
| [GCR](https://github.com/thustorage/GCR) / [PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) | Public research implementations of more advanced GPU capture. | Read and reproduce later if measured pause costs justify the effort. |
| [Cedana daemon](https://github.com/cedana/cedana) | AGPL-3.0 orchestration around checkpoint/restore. | Its GPU options include the community CUDA plugin and a separately proprietary Cedana plugin; do not treat the whole stack as open source. |

TRL's resume interface restores model, optimizer, and scheduler state from a Trainer checkpoint. In Transformers, `save_only_model=True` omits state needed for full training resume. Compare against a proper training checkpoint, not merely `save_pretrained()` or a final adapter export. [TRL resume API](https://huggingface.co/docs/trl/sft_trainer#trl.SFTTrainer.train), [Transformers training arguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer).

Code worth opening while implementing: [PEFT adapter serialization](https://github.com/huggingface/peft/blob/main/src/peft/utils/save_and_load.py), [TRL trainer](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py), and the [CRIU CUDA plugin](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/cuda/cuda_plugin.c). Check out fixed revisions before reproducing an experiment; these links follow moving development branches.

## 6. Companies already doing related work

“No company implements this” is incorrect. The narrower question is which product documents the ability to resume our kind of training, under which constraints. Documentation is evidence of an offered capability; it is not independent validation of every supported workload or a count of production customers.

| Company / offering | Public evidence | Relevance to single-GPU fine-tuning |
| --- | --- | --- |
| **MemVerge** | GPU Cluster Manager documents pausing/resuming training workspaces. Memory Machine Batch documents single-GPU restore on the same GPU model and full GPU backups. [Workspace guide](https://docs.memverge.com/AI/GPU_Cluster_Manager/0.4.0/quickstart_guide/checkpoint-restore/), [Batch requirements](https://docs.memverge.com/MMBatch/latest/User%20Guide/Getting%20Started/prerequisites/) | One of the closest commercial use cases. The workspace quickstart contains a placeholder for training code, so it is not a reproducible LoRA correctness benchmark. |
| **Cedana** | General GPU save/migrate/resume, with public usage documentation. A vendor benchmark includes training on an L4. [GPU guide](https://docs.cedana.ai/daemon/checkpoint-restore/cr-1), [vendor performance article](https://docs.cedana.ai/articles/performance-of-cedanas-gpu-interception) | Relevant commercial implementation. Proprietary GPU plugin requires access. The cited compatibility table stops at driver 570, so confirm newer-driver support before considering it for our recorded 580 host. |
| **Modal** | GPU Memory Snapshots are documented as **alpha**, capturing initialized state before function execution. [Memory Snapshots](https://modal.com/docs/guide/memory-snapshots) | Strong evidence for GPU startup snapshots; the documented lifecycle is not an arbitrary mid-training save API. Modal explicitly warns that weight-loading-bound startup may not improve. |
| **Google Cloud GKE** | Pod snapshots capture GPU state using `cuda-checkpoint`, with gVisor for the sandbox. Docs updated September 9, 2026 list single-GPU support and specific L4/A100/H100 machine types. [GKE Pod snapshots](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/pod-snapshots) | Real managed GPU snapshot support. The documented motivation is fast initialization. Restore needs matching hardware/software and changes external network connections; exact training continuation still requires workload validation. |
| **NVIDIA Dynamo** | Snapshot infrastructure combines CRIU and `cuda-checkpoint` to start initialized inference workers; snapshot-agent remains marked **preview**. [Snapshot guide](https://docs.nvidia.com/dynamo/v1.0.2/kubernetes-deployment/deployment-guide/snapshot), [current artifact status](https://docs.dynamo.nvidia.com/dynamo/dev/reference/release-artifacts) | Useful implementation reference for a future platform. The documented supported workers are inference workers, not a fine-tuning trainer. |
| **Beam** | `checkpoint_enabled=True` snapshots a container after `on_start`, including GPU initialization. [Cold-start documentation](https://docs.beam.cloud/v2/topics/cold-start) | A shipped startup feature with a defined lifecycle. It does not establish saving arbitrary training progress. |
| **Cerebrium** | July 1, 2026 engineering article describes CPU/GPU snapshots of warmed containers. [Engineering article](https://cerebrium.ai/blog/reducing-gpu-cold-starts-with-memory-snapshots-restoring-cuda-workloads-in-second) | Another commercial cold-start implementation. Its published performance evidence concerns serving workloads. |
| **Microsoft Singularity** | Microsoft describes transparent training preemption and migration in its 2022 research publication. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/singularity-planet-scale-preemptive-and-elastic-scheduling-of-ai-workloads/) | Strong systems precedent; public evidence here is a research/service description, not a currently purchasable standalone snapshot library. |

NVIDIA Run:ai also documents preemptible training using application checkpoints saved periodically or around interruption. That shows the alternative a transparent snapshot competes with; it does not establish the absence of other Run:ai mechanisms. [Run:ai training checkpoint guidance](https://run-ai-docs.nvidia.com/self-hosted/workloads-in-nvidia-run-ai/using-training/checkpointing-preemptible-workloads).

### Why startup snapshots can be an easier product

An initialized inference image can serve as a template for many future workers. The expensive snapshot creation is paid once and reused. Each useful training checkpoint represents new learning; repeated saves contain new state, and many will never need restoration. This is an economic difference, even when both features share underlying technology.

Startup also gives a controlled capture point. Arbitrary training suspension has to handle the current data position, pending updates, attached services, and library behavior. Inference engines can sometimes discard empty caches before saving; optimizer history cannot simply be discarded while claiming exact training continuation. These are reasons to offer constrained products and explicit support matrices, rather than a universal promise.

## 7. The bounded experiment to do next

**Recommendation: ordinary LoRA on Qwen2.5-0.5B, one GPU, one training process, one save/restore.** Its published base model has approximately 0.49 billion parameters. Its small size keeps the question about restoration rather than GPU capacity. This is a mechanism demonstration, not an attempt to improve model quality in 25 steps. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).

### Proposed configuration

| Choice | Proposed baseline |
| --- | --- |
| Training code | Small explicit PyTorch loop using Transformers and PEFT |
| Model | Fixed revision of `Qwen/Qwen2.5-0.5B`; downloaded before the experiment |
| Method | LoRA rank 8, explicit query/value projection targets, no additional trainable base modules |
| Precision | FP32 for this small correctness baseline; BF16 is a later controlled change |
| Optimizer | Non-paged `torch.optim.AdamW`, trainable parameters only |
| Data | Small fixed local set of prompt/answer examples, tokenized before launch; record its hash and batch order |
| Work per update | Batch size 1, short fixed sequence length, initially no gradient accumulation |
| Execution | Eager attention, no `torch.compile`, no extra kernel packages, no remote tracking |
| Boundary | After update 20: clear gradients, finish GPU work, record state, publish readiness, wait for release |
| Comparison | Finish at update 25; same host, GPU, model revision, data and package versions |

LoRA target selection is explicit to make the trainable state easy to inspect, not a recommendation for best model quality. PEFT documents the relevant configuration and parameter counting. [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora).

### Three runs answer three different questions

1. **Uninterrupted reference:** train from the chosen initial state through update 25. Record state at updates 20 and 25.
2. **Application restart:** train to update 20, save adapters plus complete training state, terminate the process, rebuild from the fixed base and saved state, then perform five more updates.
3. **Whole-process restore:** train to update 20, capture CPU and GPU state, verify the original process has exited, restore it, then perform five more updates. Application checkpoint files must not be its restore input.

Use exactly the same training loop for all three runs. Keep the base model cached on disk for both restore paths so a network download does not manufacture an apparent snapshot advantage. Any diagnostic tensor copies used for validation belong outside the measured checkpoint interval and should not inflate the captured process accidentally.

The first correctness baseline may disable dropout. A small follow-up should enable it and verify random-state restoration; otherwise matching random-state hashes alone has not exercised stochastic continuation. Before interpreting final-step differences, establish that two uninterrupted runs match in this fixed environment. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

### What counts as success

Before the next update after restore, require exact equality of adapter tensors, optimizer tensors, update count, next batch position, and used random-generator states with the corresponding pre-save values. Verify the fixed base and configuration too. Then compare updates 21–25 with the uninterrupted reference, including losses and trainable state. A decreasing loss alone is insufficient.

For deterministic execution in the same environment, target exact continuation. If a numerical tolerance is necessary, retain values, choose the tolerance in advance, and explain the source of variation. Hashes prove equality, but cannot measure how close unequal tensors are.

Also establish that the GPU was released during suspension and could be used by a separate tiny disposable workload. GPU-only suspend/resume is a useful intermediate check, but the final claim requires the original CPU process to exit and the saved process image to be restored.

### Where this fits in the repository

The existing [implementation plan](implementation-plan.md) names `train.py` and `checkpoint.sh`; those files are not present in this checkout yet. Its [review](implementation-plan-review.md) already identifies the explicit release gate, deterministic reference, detached restore, cleanup, and the need to choose one owner for CUDA restore. Retain that small division of responsibilities when extending to fine-tuning.

The last [host gate record](gate-results.md) reports driver **580.126.20** and both tools missing from PATH. That is a setup gap, not proof of incompatibility. The next execution session must establish permissions and CPU restore first, then tiny CUDA restore, before spending time on fine-tuning. Later driver features must not be assumed on this recorded host.

If CRIU's CUDA plugin restores and unlocks CUDA, the shell must not unconditionally repeat those transitions. This is already recorded in the project guidance. [Upstream restore hook](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/cuda/cuda_plugin.c#L493-L505).

### Add one source of complexity at a time

| Stage | What it teaches | Effort judgment |
| --- | --- | --- |
| Tiny existing model | Whether this host can restore any training process | Necessary first gate |
| Small ordinary LoRA | Whether real pretrained-model training continues correctly | Highest-value next extension |
| BF16, then four-bit base with non-paged optimizer | Precision and quantization compatibility, separately | Bounded follow-ups |
| TRL SFTTrainer | Behavior in a common training framework | Useful once the explicit loop works |
| Larger model or full fine-tuning | How state size changes the comparison | Only after correctness and headroom are established |
| Compression or incremental capture | Whether a measured bottleneck can be reduced | Separate optimization work; incremental capture is substantially more involved |

These are relative effort judgments, not delivery estimates. Kernel permissions and driver/library interactions can dominate setup time. A reproducible blocker is a valid feasibility finding, but it is not a successful restore demonstration.

## 8. Measure the return, not just the snapshot

Record separate timestamps for **request to GPU release**, **request to completed image**, and **restore request to the next completed optimizer update**. If the image has not been flushed or copied to storage that survives the relevant failure, do not call it durable. A fast in-memory snapshot solves a different failure case from a checkpoint on independent storage.

Also record image size, host-memory peak, GPU allocated/reserved memory at the boundary, uninterrupted step time, and any failures. For a later timing comparison, repeat a few times under the same cache and storage conditions and report the spread. Do not infer an L4 result from an H100 or B200 paper.

For a planned pause, both methods can stop after the same update, so neither needs to redo useful work. Compare the combined save and restart costs, plus the engineering cost of preserving the whole process. For an unexpected failure, only an already-saved checkpoint helps; one cannot capture a dead GPU after the event.

An illustrative event-cost comparison is:

```text
Process snapshot cost = save time + restore-to-next-update time
Application restart cost = save time + rebuild-to-next-update time + work repeated
```

The terms must use the same completion and storage criteria. Periodic save overhead also matters even when no failure happens. If failures are equally likely throughout a checkpoint interval, average lost work is roughly half that interval; this is a simplifying assumption, not a guarantee.

**Choose an application checkpoint when** training state is explicit, adapter saves are small, startup is tolerable, and portability matters. **Choose a process snapshot when** preserving a complex live environment or avoiding expensive reconstruction is worth its state-transfer cost and tighter compatibility requirements. Often the practical design uses process snapshots for planned pause/resume and keeps application checkpoints as a separate recovery/export path.

The interview claim to aim for is precise: **“We restored one supported single-GPU LoRA process and verified its next updates. We measured its cost against a full training-state checkpoint and identified where the approach stops being attractive.”** That demonstrates more judgment than claiming universal or instantaneous GPU migration.

## 9. Remaining uncertainties

Public evidence supports the feasibility of GPU-aware training checkpoint/restore and several commercial uses. It does not settle compatibility or cost for our exact Jarvis container, package versions, model, and allocator. The B200 fine-tuning paper does not provide a sufficiently explicit LoRA/QLoRA configuration; Cedana's cited driver table needs current confirmation; hosted startup features do not establish an arbitrary mid-training API.

The recommended decision is therefore to measure a narrow supported case first. The most promising later optimization for LoRA is avoiding repeated capture of an unchanged base model. Proving that safely requires more than ignoring tensors marked frozen, because the snapshot layer must still restore addresses and resource relationships correctly.

## Sources and reading order

All web documentation and repositories below were consulted on September 13, 2026. “Living documentation” means no stable publication date was established; pin a release or commit before reproducing an implementation. Performance numbers above are author-reported and tied to the stated experiment.

### Start with these papers

1. Stoyanov et al. **CRIUgpu: Transparent Checkpointing of GPU-Accelerated Workloads.** arXiv v1, February 23, 2025. [Paper](https://arxiv.org/html/2502.16631v1). Closest architecture and training measurements.
2. Zeng et al. **GPU Checkpoint/Restore Made Fast and Lightweight.** FAST, February 24–26, 2026. [Paper](https://www.usenix.org/system/files/fast26-zeng.pdf), [artifact](https://github.com/thustorage/GCR). Separate CPU-memory results and inference increments from durable training saves.
3. Stoyanov et al. **Towards On-the-Fly Snapshot Memory Compression for Low-Latency Elastic Inference Serving Systems.** EuroMLSys, April 27–30, 2026. [Paper](https://radostin.io/files/stoyanov-euromlsys-2026.pdf). Section 5.3 supplies the training counterexample to compression optimism.
4. Wei et al. **PhoenixOS: Concurrent OS-level GPU Checkpoint and Restore with Validated Speculation.** SOSP 2025; cited arXiv revision September 1, 2025. [Paper](https://arxiv.org/html/2405.12079v4), [code](https://github.com/SJTU-IPADS/PhoenixOS).
5. Shukla et al. **Singularity: Planet-Scale, Preemptive and Elastic Scheduling of AI Workloads.** Microsoft, February 2022. [Paper](https://arxiv.org/abs/2202.07848), [publisher page](https://www.microsoft.com/en-us/research/publication/singularity-planet-scale-preemptive-and-elastic-scheduling-of-ai-workloads/).
6. Jain and Cooperman. **CRAC: Checkpoint-Restart Architecture for CUDA with Streams and UVM.** SC 2020. [Paper](https://arxiv.org/abs/2008.10596), [early code](https://github.com/DMTCP-CRAC/CRAC-early-development).
7. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv, June 2021. [Paper](https://arxiv.org/abs/2106.09685).
8. Dettmers et al. **QLoRA: Efficient Finetuning of Quantized LLMs.** arXiv, May 2023. [Paper](https://arxiv.org/abs/2305.14314).

### Implementation references

- NVIDIA. **cuda-checkpoint**, living README and source. [Repository](https://github.com/NVIDIA/cuda-checkpoint).
- CRIU maintainers. **GPU Checkpointing**, CUDA plugin, and current command documentation. [Guide](https://www.criu.org/GPU_Checkpointing), [plugin](https://github.com/checkpoint-restore/criu/tree/criu-dev/plugins/cuda), [CLI and compression options](https://github.com/checkpoint-restore/criu/blob/criu-dev/Documentation/criu.txt).
- Hugging Face / bitsandbytes maintainers. **Paged optimizers** and allocation implementation, living documentation/source. [Explanation](https://huggingface.co/docs/bitsandbytes/explanations/optimizers), [optimizer.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py), [native allocation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/pythonInterface.cpp).
- Hugging Face. **PEFT checkpoint format and LoRA reference**, living documentation. [Format](https://huggingface.co/docs/peft/developer_guides/checkpoint), [LoRA](https://huggingface.co/docs/peft/package_reference/lora).
- Hugging Face. **Transformers Trainer, TRL SFTTrainer, and Accelerate checkpointing**, living documentation. [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer), [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/checkpoint).
- PyTorch / Meta. **Checkpointing in torchtune**, cited stable documentation version 0.6. [Guide](https://meta-pytorch.org/torchtune/stable/deep_dives/checkpointer.html).
- PyTorch contributors. **CUDA memory management, data loading, reproducibility, and activation checkpointing**, living documentation. [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management), [data](https://docs.pytorch.org/docs/stable/data.html), [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html).
- Qwen Team. **Qwen2.5-0.5B model card**, model family released September 2024. [Model](https://huggingface.co/Qwen/Qwen2.5-0.5B).

### Product evidence

- MemVerge. **Working with Snapshots**, GPU Cluster Manager 0.4.0, and **Memory Machine Batch prerequisites**, living documentation. [Training workspace](https://docs.memverge.com/AI/GPU_Cluster_Manager/0.4.0/quickstart_guide/checkpoint-restore/), [Batch limits](https://docs.memverge.com/MMBatch/latest/User%20Guide/Getting%20Started/prerequisites/).
- Cedana. **Checkpoint/restore with GPUs**, **Performance of Cedana's GPU Interception**, and daemon source, living documentation. [GPU support](https://docs.cedana.ai/daemon/checkpoint-restore/cr-1), [performance](https://docs.cedana.ai/articles/performance-of-cedanas-gpu-interception), [source](https://github.com/cedana/cedana).
- Modal. **Memory Snapshots**, living documentation; GPU feature marked alpha at access. [Guide](https://modal.com/docs/guide/memory-snapshots).
- Google Cloud. **About GKE Pod snapshots**, updated September 9, 2026. [Guide](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/pod-snapshots).
- NVIDIA Dynamo. **Snapshot**, cited version 1.0.2 guide, and current **Release Artifacts**. [Guide](https://docs.nvidia.com/dynamo/v1.0.2/kubernetes-deployment/deployment-guide/snapshot), [status](https://docs.dynamo.nvidia.com/dynamo/dev/reference/release-artifacts).
- Beam. **Cold Start Performance**, living documentation. [Guide](https://docs.beam.cloud/v2/topics/cold-start).
- Hamdulay, Yaseen / Cerebrium. **Reducing GPU Cold Starts with Memory Snapshots: Restoring CUDA Workloads in Seconds.** July 1, 2026. [Article](https://cerebrium.ai/blog/reducing-gpu-cold-starts-with-memory-snapshots-restoring-cuda-workloads-in-second).
- NVIDIA Run:ai. **Checkpointing Preemptible Training Workloads**, living documentation. [Guide](https://run-ai-docs.nvidia.com/self-hosted/workloads-in-nvidia-run-ai/using-training/checkpointing-preemptible-workloads).
