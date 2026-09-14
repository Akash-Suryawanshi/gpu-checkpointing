# Four-update LoRA snapshot experiment

Use the EC2 host execution context with GPU access and working sudo. The restricted
tool sandbox does not expose the GPU. Keep all assets, environments, and run output
under ignored `runs/`.

Create an isolated environment, then prepare the pinned local model and tokens:

```bash
python3 -m venv runs/finetuning/venv
runs/finetuning/venv/bin/pip install -r experiments/finetuning/requirements.lock
runs/finetuning/venv/bin/python experiments/finetuning/prepare.py runs/finetuning/assets
```

To reuse the model from the preliminary probe, add
`--model-dir runs/tools/qwen2.5-0.5b`. Preparation refuses to overwrite existing
output. It records the model revision, asset hashes, tokens, and environment.

Create two matching references, then run one process-image trial:

```bash
runs/finetuning/venv/bin/python experiments/finetuning/run.py reference runs/finetuning/reference-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py criu runs/finetuning/criu-zero \
  --reference runs/finetuning/reference-zero
runs/finetuning/venv/bin/python experiments/finetuning/run.py application runs/finetuning/application-zero \
  --reference runs/finetuning/reference-zero
```

Each run directory must be new. Add `--dropout 0.1` to both commands to exercise
CUDA randomness. The early gate uses `--until 2` for both commands and `--capture 1`
for the CRIU trial. The default is four updates, with capture after two.

The application route saves adapters, buffers, Adam, schedule, data position, RNG,
and training behavior. It flushes and atomically publishes the save, verifies the
original process has exited, runs job B, then builds a fresh trainer from the pinned
base model and the save. It restores RNG last and compares state before update 3.
The process-image route restores solely from CRIU images.

`run.py` uses the isolated interpreter that launched it. `session.py` invokes CRIU's
CLI and handles waiting, exit verification, and cleanup; it does not implement
checkpointing. CRIU also offers `pycriu` RPC bindings, but the CLI keeps the single
privileged call separate from the trainer without introducing a worker/service
interface. NVIDIA transitions belong entirely to CRIU's CUDA plugin.

The default `--tools runs/tools` expects `criu-4.2.1/criu/criu`, its
`plugins/cuda/cuda_plugin.so`, libraries under `criu-deps/usr/lib/x86_64-linux-gnu`,
and `cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint`. This matches the isolated
EC2 tool build recorded in the project results. No host Python packages are used.
To reproduce the tool build on Ubuntu 22.04 x86_64 with Git, GCC, Make, and
pkg-config available, run `bash experiments/criu/build.sh runs/new-tools`, then
pass `--tools runs/new-tools`. The script extracts downloaded development packages
locally and installs no system packages.

Results contain separate lifecycle/numerical verdicts and compatibility warnings.
References are reused only when their environment, assets, settings, and code
fingerprints match. Full state evidence is read only by the verifier. The restored
trainer waits until the controller compares it before allowing another update.

Run the small CPU-side contract tests with:

```bash
PYTHONPATH=experiments/cpu:experiments/finetuning:experiments/criu \
  runs/finetuning/venv/bin/python -m unittest discover -s tests -v
```

This is a same-host correctness experiment; model quality and spot recovery are
outside its acceptance claim. Shared-memory ownership and CRIU warnings remain
explicit qualifications even when numerical continuation passes.
