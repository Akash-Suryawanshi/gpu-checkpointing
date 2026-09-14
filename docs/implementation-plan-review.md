# Implementation plan review

**Verdict: revise before implementation.** The one-process, one-GPU scope is appropriate. Six gaps could cause a hung run or a misleading success claim; fixing them does not require a larger project.

Reviewed `docs/implementation-plan.md` on 2026-09-12. References below use its current line numbers. This review does not change the plan or run anything on the Jarvis instance.

## 1. P1 — Define what the restored run must match

**Location:** `docs/implementation-plan.md:88` and `:94`.

The plan collects losses and hashes, but never defines a comparison with an uninterrupted run. Step 25 should normally differ from step 20, and a decreasing loss alone would not catch lost optimizer state. A hash mismatch also cannot tell us a numerical tolerance.

**Minimal fix:** use two runs of the same script: an uninterrupted 25-step baseline, and a run checkpointed after step 20 that then completes five steps. Recompute the model and optimizer digests immediately after restore, before another update, and require exact equality with the pre-dump state. Compare step 25 with baseline step 25. Include the step counter, batch position, and any random-generator state the loop uses. Use an optimizer with real accumulated state, such as SGD with momentum.

For the final-step comparison, set deterministic execution options in the same environment; a fixed seed alone is insufficient. If allowing numerical differences, retain tensor values for comparison and choose the tolerance before the run. Such output is verification evidence, not a restore mechanism. [PyTorch reproducibility guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).

## 2. P1 — Replace the brief wait with an explicit release

**Location:** `docs/implementation-plan.md:78`.

The shell can miss the brief wait and checkpoint a later step. CUDA suspension does not freeze CPU threads, so the marker alone cannot keep the Python counter and recorded state fixed. [NVIDIA's execution model](https://github.com/NVIDIA/cuda-checkpoint#the-utility).

**Minimal fix:** at step 20, finish GPU work, record state, publish readiness, and wait for an explicit release from the shell. A fresh directory containing readiness/release files is enough; the shell times out and reports failure if readiness never arrives. Release only after restoration. `torch.cuda.synchronize()` makes GPU completion explicit for this chosen training boundary. [PyTorch API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html).

Describe this honestly as a demo using a cooperative pause point with transparent state capture. It does not establish checkpointing at arbitrary moments.

## 3. P1 — Choose who restores CUDA: the script or the plugin

**Location:** `docs/implementation-plan.md:82`; related context: `README.md:24`.

The plan puts manual CUDA restore/unlock after CRIU restore while the README refers to CRIU integration. With the upstream CUDA plugin enabled, CRIU's restore hook already restores and unlocks CUDA, including a process that was manually checkpointed before the dump. The extra commands can then fail because CUDA is already running. [CRIU plugin restore hook](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/cuda/cuda_plugin.c#L493-L505), [CUDA state requirements](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html).

**Minimal fix:** select one approach and record the installed versions. For the proposed explicit-command teaching demo, use a verified manual path with automatic CUDA plugin handling disabled. Alternatively let the plugin own all CUDA transitions and remove the redundant commands. Do not mix the two implicitly.

## 4. P2 — Test execution capability, not just command presence

**Location:** `docs/implementation-plan.md:64` and `:70`.

`command -v` only checks whether a command is on PATH. Installing both executables would not establish the container permissions or kernel support needed to save and rebuild a process. Conversely, their absence is a setup task, not proof the host cannot support checkpointing.

**Minimal fix:** record tool versions/options, run `criu check`, and attempt a tiny disposable CPU process dump/restore under the same user and container. Check actual PyTorch CUDA execution separately. Inspect container permissions when a probe fails; do not assume root inside a container has all host privileges. [CRIU kernel checks](https://criu.org/Check_the_kernel), [CRIU privilege requirements](https://github.com/checkpoint-restore/criu/blob/criu-dev/Documentation/criu.txt).

The existing `docs/gate-results.md:29` records driver 580.126.20 and both tools missing from PATH. It does not establish that installing them will succeed or that additional privileges are unavailable. Keep setup bounded: an installable tool gap can be addressed; a host-level restriction should become a concrete feasibility finding.

## 5. P2 — Specify process exit, detached restore, and failure cleanup

**Location:** `docs/implementation-plan.md:45` through `:52`.

`set -euo pipefail` exits on errors but does not undo a CUDA lock or suspended process. Also, a foreground CRIU restore can keep the shell waiting before it reaches the manual CUDA restore/release step.

**Minimal fix:** define a detached launch with stdin disconnected and output in a regular file; use the matching CRIU options and `--restore-detached`. Require the original process to exit after a successful dump and record the restored PID. Add bounded waits and a small failure handler that saves logs and the last known CUDA state. Recover only from a verified state; otherwise terminate only this disposable demo process. [CRIU lifecycle options](https://github.com/checkpoint-restore/criu/blob/criu-dev/Documentation/criu.txt).

Do not promise recovery after every driver error: NVIDIA explicitly leaves some failure states undefined for application health. [NVIDIA limitations](https://github.com/NVIDIA/cuda-checkpoint#functionality).

## 6. P2 — Separate a blocked experiment from a successful POC

**Location:** `docs/implementation-plan.md:102` through `:105`.

The current criteria call the POC complete even if nothing can be restored. That would let the existing missing-tool report satisfy a goal stated as proving GPU training continuation.

**Minimal fix:** report distinct outcomes: **full restore demonstrated**, **GPU pause/resume only**, or **environment blocked**. The last two are useful findings about the mechanism and its limits, but only the first meets the full-process POC goal. Keep any `torch.save` comparison explicitly separate.

## Recommended order after these corrections

1. Resolve installable prerequisites on the existing Jarvis instance and verify CPU process restore.
2. Run the tiny CUDA training baseline and record its step-20 and step-25 state.
3. Verify GPU-only pause/resume using the same training script to isolate driver issues.
4. Add the full CPU/GPU dump/restore cycle and compare the corresponding states.
5. Report timings, image size, outcome, and the exact limits observed.

Keep the implementation in `train.py` and `checkpoint.sh`, with explanation and results in the README. The two execution paths provide the meaningful acceptance check; no general testing framework is needed for this first demo.
