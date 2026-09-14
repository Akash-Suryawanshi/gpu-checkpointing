# CPU/GPU snapshotting: learning scope

Agreed direction from the September 14 discussion: understand the mechanism first, then finalize the implementation plan. This document defines learning outcomes and candidate observations; it is not a finalized implementation plan or evidence of a successful restore.

The [study guide](r-2026-09-14T10-56-45.html) contains the explanation, source links, economic corrections and worked transfer example.

## Questions to answer before finalizing implementation

1. What process memory, registers, thread state and Linux resources does CRIU capture? What remains outside the snapshot?
2. How do `/proc`, `ptrace`, CRIU's injected helper and its restorer cooperate? Which image records can we inspect?
3. What belongs to CUDA Runtime/Driver APIs, the NVIDIA driver, cuda-checkpoint and the CRIU CUDA plugin?
4. Why must GPU work drain? How do GPU bytes reach host RAM, and how are virtual addresses and CUDA objects restored?
5. Why does the plugin temporarily run NVIDIA's helper while coordinating frozen application threads?
6. When is the GPU free for another job, and when is the snapshot safe against host loss? What are the distinct RAM, storage and time costs?
7. Which compatibility and permission constraints apply to the actual installed versions?
8. When does a process snapshot beat a full application checkpoint triggered at the same pause boundary?

## Proposed minimal observations

- One Linux host, one GPU; first restore a CPU-only counter with an open-file position.
- Then a tiny deterministic PyTorch training process with real optimizer state and an explicit completed-update waiting point.
- Observe NVIDIA checkpoint state and RAM/VRAM before capture, after GPU release, and after restore. Let a separate tiny job B use the released GPU and exit before restoring A.
- Require A's original process to exit after the full dump; restore from process images without application checkpoint files as restore input.
- Compare the restored update-20 state before any update, then updates 21–25 against the uninterrupted reference.
- Compare full application save/restart against process capture/restore using the same training loop, boundary, cached assets and storage-completion criterion.
- Record request-to-GPU-release, request-to-completed-image, restore-to-next-completed-update, image size and peak host RAM. Establish durability separately if the storage is local only.
- Inspect CRIU images and CUDA-plugin logs to connect each visible effect to its mechanism.

The same-host handoff to B illustrates GPU reuse, not a production spot scheduler or cross-host recovery guarantee. A simulated warning establishes only behavior under that chosen deadline.

## Economic assumptions

- T is the interval of useful training between saves; C is save duration; W is warning lead time. They are not interchangeable.
- T/2 is an approximate expected loss under uniform interruption timing and short saves, not half the save duration.
- A warning-triggered full application checkpoint can avoid the same rollback as a process snapshot if it completes in time.
- Billing during capture, warning reliability, interruption frequency and delayed service to B remain unknown provider-policy inputs. A 20% share of measured savings per interruption is a pricing hypothesis, not proof of a viable hourly premium.
- Retaining the last completed image protects recovery; it does not reduce bytes written per save.

## Later plan

Retain the [minimal plan](implementation-plan.md) and its [review](implementation-plan-review.md) as drafts to reconcile after the walkthrough. The [September 13 fine-tuning report](r-2026-09-13T07-42-27.html) describes a later small LoRA extension. Initial work excludes multi-GPU coordination, cross-host migration, compression/incremental GPU capture, billing integration and a production scheduler.
