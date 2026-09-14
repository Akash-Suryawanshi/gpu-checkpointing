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

Training and controller commands will be added at their tested milestones. This
is a same-host correctness experiment; model quality and spot recovery are outside
its acceptance claim.
