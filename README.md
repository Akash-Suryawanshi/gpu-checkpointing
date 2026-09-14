# CPU/GPU checkpointing POC

A small, observable experiment to learn what GPU snapshotting is, how it works, and where it can help. One Linux host, one process, then one NVIDIA GPU. Start by proving CPU process restoration before adding GPU training.

## Current status

**GPU-only checkpoint/restore passed on the L4.** A 64 MiB tensor matched byte-for-byte after NVIDIA released and restored its GPU state; a second GPU job ran while it was suspended, and the original process completed its next GPU operation correctly.

**DMTCP restored a complete PyTorch GPU process, with a compatibility qualification.** The original process exited; a disk image restored into a new Linux process; the full tensor hash and next GPU operation checks passed. Four anonymous shared-memory warnings remain unresolved, so this is not yet an unqualified fine-tuning acceptance result. DMTCP's CPU counter restore passed. See the [fine-tuning feasibility audit](docs/r-2026-09-14-finetuning-feasibility.html) and [measured results](docs/gate-results.md).

**The CRIU route remains blocked on this container.** Both its 3.16.1 capability check and direct CPU dump fail during startup feature detection. The CRIU scripts below remain useful on a compatible execution host.

The [study guide](docs/r-2026-09-14T10-56-45.html) is unchanged. The [implementation plan](docs/implementation-plan.md) now follows the agreed CPU-first scope.

## First experiment: CPU state

Use a Linux environment where you may checkpoint your own disposable process. Run the gate and the experiment under the same user, in the same container or host. Root inside a restricted container may still lack required permissions.

```bash
bash scripts/check-host.sh cpu
bash checkpoint.sh cpu "$PWD/runs/cpu-01"
```

The second command requires a **new** run directory and refuses to overwrite an existing one. It creates all input files. Python needs only the standard library. Linux prerequisites are CRIU and the usual GNU/core Linux tools (`timeout`, `sync`, `setsid`).

Do not change host security settings to force a failed probe through. Read the retained error first; a missing executable and a denied kernel operation are different setup problems. See [previous host observations](docs/gate-results.md).

The experiment performs:

1. An uninterrupted reference run from 1 through 25.
2. A second process that reads 20 fixed-size records, then waits with its counter, random token and open-file position still in memory.
3. A CRIU dump that ends that process; the shell reaps it and checks its PID has disappeared.
4. A detached CRIU restore, followed by explicit release of the waiting process.
5. Comparison of state before/after the pause and all 25 records against the reference.

`cpu_counter.py` never reads an application checkpoint or the evidence logs. CRIU is the only restore mechanism in this experiment. The input file must remain present and unchanged. CRIU commonly restores the original numeric PID; disappearance before restore, not a different PID afterward, demonstrates that the original ended.

## Read the result

```bash
cat runs/cpu-01/comparison.json
cat runs/cpu-01/lifecycle.txt
cat runs/cpu-01/process.jsonl
```

Expected state: step 20, byte offset 100 and the same random token before/after restoration; next record 21; final step 25 at offset 125.

A matching JSON log **alone does not prove CRIU ran**. The command script also checks original-process exit and successful dump/restore commands. Retain `images/dump.log`, `images/restore.log`, `original.pid`, `restored.pid` and `lifecycle.txt` together with the state comparison.

`host.log` records the environment and CRIU gate. `timing-ns.txt` records monotonic timestamps; subtract pairs and divide by 1e9 for seconds. `image-directory-bytes.txt` records the image directory's apparent size, including its logs. Polling and timestamp-command overhead are included; this is an initial observation, not a precise benchmark.

`sync -f` completes a filesystem sync on this Linux host. That does not establish survival if the provider removes the host's local storage. This first experiment is same-host restoration.

If CRIT is installed:

```bash
crit decode -i runs/cpu-01/images/pstree.img --pretty
```

Inspect the image directory for the matching `core-*.img`, `mm-*.img`, `pagemap-*.img` and raw `pages-*.img`. File names depend on CRIU version; raw page files are not structured CRIT records.

## Files to read, in order

| File | Purpose |
| --- | --- |
| `cpu_counter.py` | The running program and its explicit wait at 20 |
| `checkpoint.sh` | Visible CRIU dump/restore commands and lifecycle evidence |
| `verify_cpu.py` | Check continuity, counter, file position and token |
| `scripts/check-host.sh` | Verify the actual execution environment |

Local checks:

```bash
python3 -m unittest discover -s tests -v
bash -n checkpoint.sh scripts/check-host.sh
```

## GPU-only probe

This separate experiment uses manual NVIDIA transitions and keeps the original CPU process alive. It does not use CRIU, create a durable process image, or prove training restart after process/host loss.

```bash
bash scripts/probe-gpu.sh /home/gpu-checkpointing-tools/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint "$PWD/runs/gpu-only-02"
```

Use a fresh run directory. `gpu_memory_probe.py` creates and hashes a 64 MiB tensor, waits for external release, then verifies its contents and another GPU operation. The shell records CUDA state, process GPU memory, host RSS and transition times; runs job B while A is checkpointed; and restores/unlocks A before release.

The tested NVIDIA utility is version 595.58.03, source commit `00d5cce84c628088d6caa203fc4af40c1538b6f7`. Obtain it from [NVIDIA's official repository](https://github.com/NVIDIA/cuda-checkpoint). These utility files are in a separate tools directory on the instance.

## Next: full process restore and training

The current-host route is DMTCP's native CUDA plugin at the pinned maintenance revision recorded in the audit. Its basic CPU/GPU process restoration has been observed; next investigate the shared-memory warnings and implement the small deterministic training comparison. The complete LoRA workload has not yet been tested.

CRIU remains an alternative on an environment with the necessary permissions. That route needs CRIU 4.0 or newer with its CUDA plugin; the Ubuntu 3.16.1 package was used only for the initial CPU feasibility check. For either integration, let its plugin own CUDA restoration and do not append unconditional manual restore/unlock commands. Verify the installed plugin and tool versions before adding orchestration.

Generated images contain process memory. They are kept under ignored `runs/` with restrictive permissions. Failed runs preserve evidence and attempt to terminate only this experiment's process.

Sources: [CRIU example](https://criu.org/Simple_loop), [CRIU command options](https://github.com/checkpoint-restore/criu/blob/criu-dev/Documentation/criu.txt), [CUDA plugin](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/cuda/cuda_plugin.c).
