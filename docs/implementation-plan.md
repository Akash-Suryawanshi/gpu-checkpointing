# Implementation plan: observable CPU/GPU snapshots

Implementation started September 14, 2026, following the user's instruction to move from theory to implementation. This replaces the earlier draft's brief wait and mixed manual/plugin restore sequence. The [learning scope](snapshot-learning-scope.md) and [review](implementation-plan-review.md) explain the requirements.

## Goal and boundary

Demonstrate that a saved process continues correctly after the original ends. One Linux host, first one CPU process, then one tiny single-GPU training process. No production scheduler, multi-GPU support, cross-host migration or fine-tuning framework.

The Mac is the authoring machine. Linux CPU restore is a prerequisite for GPU integration; missing tools are setup work, while denied host capabilities require an environment change. A blocked environment is an outcome to report, not a successful restore.

## 1. CPU process round trip

- [x] Implement `cpu_counter.py`: read records 1–25; hold a random token in memory; keep the input file open without buffering; wait after record 20 for an explicit release file.
- [x] Implement `scripts/check-host.sh cpu`: record OS, user, CRIU version and actual `criu check` result.
- [x] Implement `checkpoint.sh cpu RUN_DIR`: use a fresh run directory, run a reference, dump the waiting process, require its exit, restore detached, release, verify, and retain logs/timestamps.
- [x] Implement `verify_cpu.py`: reject repeated/missing records, changed token, wrong file position or wrong event order.
- [x] Verify the workload and checker locally, with tests explicitly distinguished from CRIU validation.
- [ ] Run the actual Linux round trip and inspect process-tree, core and memory images.

Acceptance: before/after state is step 20, offset 100, identical token; next record is 21; final step is 25, offset 125. Both CRIU commands must succeed and the original must be gone before restore. Numeric PID reuse during restore is expected.

## 2. Tiny CUDA training and application comparison

After CPU restore succeeds, check the GPU driver, installed CUDA checkpoint tool, CRIU CUDA plugin and actual PyTorch GPU execution. Record versions; do not infer compatibility from tool presence.

Implement `train.py` with a two-layer model, synthetic input and SGD with momentum. Run 25 updates. Set deterministic execution and disable reduced-precision shortcuts for the comparison; keep execution on the same GPU/software environment. Exercise random-number state by generating training batches during the loop.

At completed update 20, synchronize CUDA, record state and wait for explicit release. Record model, optimizer, step, next-batch position and every random generator actually used. Recompute evidence immediately after release, before update 21. Evidence files must never become restore inputs for the process-snapshot path.

Run an uninterrupted reference and a full application save/restart using the same loop. The application checkpoint includes model, optimizer, step/data position and RNG state. Require identical update-20 state and identical updates 21–25 in this controlled deterministic experiment. Any mismatch fails; investigate before introducing tolerances.

## 3. GPU release and full CPU/GPU restoration

Choose **plugin-managed CUDA transitions** for full process snapshots. The CRIU CUDA plugin owns lock, checkpoint, restore and unlock. No unconditional manual restore/unlock follows `criu restore`.

First isolate driver behavior with a separate GPU-only pause/resume experiment. Record CUDA state and host RAM/VRAM while the application waits. Run a small job B after the driver reports A's GPU state checkpointed, let B exit, then restore A. This experiment does not prove process reconstruction.

Then add full plugin-managed CRIU dump/restore. End A after the dump; preserve image files and external assets; run B before restoring A. Verify CUDA plugin logs, restored state at 20, and updates 21–25 against the reference. Retain independent outcomes: GPU pause/resume only, full restore verified, or environment blocked.

Use bounded waits and fresh directories. On failure retain logs and terminate only the disposable experiment process when its identity can be verified. Do not attempt speculative CUDA state transitions after an unknown driver failure.

## 4. Measurements and report

Measure request-to-GPU-release, request-to-completed-image, restore-to-next-completed-update, image bytes and peak host RAM. A polling observation bounds the event time; label its resolution. Record the storage completion criterion for both application and process checkpoints.

Compare at the same completed update using the same storage. Local filesystem sync alone does not prove survival after host removal. Report actual GPU/driver/tools, correctness result, timings and limits; do not label local tests or a warning simulation as real spot recovery.

Current evidence: local and remote workload/checker tests pass. On the resumed Linux host, CRIU 3.16.1 fails during its startup network feature probe before capturing the CPU process. A separate GPU-only experiment passed: a 64 MiB tensor survived CUDA suspension and restoration, another job used the GPU in between, and computation resumed correctly. The original CPU process remained alive; full process restore and instance-loss recovery remain unverified. See `gate-results.md` for measurements and environment restrictions.
