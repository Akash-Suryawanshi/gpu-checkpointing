# Experiment results

This page records what was actually observed. The current verdict is deliberately split across three mechanisms:

- **CRIU process capture is blocked in this container.** CRIU 3.16.1 fails during startup feature detection before it captures the CPU counter.
- **NVIDIA GPU-only checkpoint/restore passed.** A 64 MiB tensor survived and another job used the GPU while it was suspended, but the original CPU process stayed alive and no durable process image was created.
- **DMTCP restored a complete PyTorch GPU process after the original exited.** Lifecycle, tensor, and subsequent GPU-compute checks passed. Four anonymous shared-memory warnings leave sharing semantics unresolved, so strict acceptance failed and fine-tuning compatibility remains unproven.

No experiment changed container security settings or the NVIDIA driver. Curated DMTCP measurements are also available as [JSON](evidence/2026-09-14/dmtcp-summary.json). Raw process images and logs remain under ignored `runs/`; coordinator logs can contain inherited environment values and must not be published wholesale.

## Run the maintained experiments

Use a Linux NVIDIA host for GPU work. Run the host check and experiment under the same user and in the same container or host. Every launcher requires a new run directory and refuses to overwrite an existing one.

```bash
# CPU process capture through CRIU
bash experiments/check-host.sh cpu
bash experiments/cpu/checkpoint.sh cpu "$PWD/runs/cpu-01"

# NVIDIA-only GPU suspension; the CPU process remains alive
bash experiments/gpu/probe-gpu.sh \
  /home/gpu-checkpointing-tools/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint \
  "$PWD/runs/gpu-only-02"

# Local CPU evidence checks
PYTHONPATH=experiments/cpu python3 -m unittest discover -s tests -v
```

The CPU experiment needs Python's standard library, CRIU, and the usual GNU/Linux tools (`timeout`, `sync`, and `setsid`). The NVIDIA probe needs PyTorch with CUDA and a compatible `cuda-checkpoint` executable. DMTCP source is retained as `experiments/dmtcp/probe-dmtcp.py`; its CLI is documented by `--help` because tool and run locations are explicit inputs.

```bash
mkdir -p runs
python3 experiments/dmtcp/probe-dmtcp.py cpu \
  "$PWD/runs/dmtcp-cpu-03" \
  --dmtcp-root /home/gpu-checkpointing-tools/dmtcp-4.2.0
python3 experiments/dmtcp/probe-dmtcp.py gpu \
  "$PWD/runs/dmtcp-gpu-02" \
  --dmtcp-root /home/gpu-checkpointing-tools/dmtcp-4.2.0
```

The DMTCP parent `runs/` directory must already exist, and each run directory must be fresh. The controller uses a restrictive file-creation mask. Its GPU mode rejects unsupported-resource warnings even when lifecycle and numerical checks pass; no new GPU run was performed as part of the repository cleanup.

Generated images contain process memory. Keep `runs/` private and on storage appropriate to the failure being tested. `sync -f` flushes to the current filesystem; it does not prove that local storage survives instance deletion.

## Read the CPU evidence

The CPU experiment first runs an uninterrupted reference from records 1 through 25. A second process reads through record 20, then waits while retaining its random token and open-file position in memory. After process restoration and release, it must continue at record 21 and finish at record 25. The expected saved boundary is step 20 at byte offset 100; the final position is step 25 at byte offset 125.

`experiments/cpu/cpu_counter.py` never loads an application checkpoint or reads the evidence logs. The input file must remain present and unchanged because its restored open descriptor still refers to that file. The launcher verifies that the original process disappeared before restore. CRIU can restore the same numeric PID, so a different post-restore PID is not required proof.

Read `comparison.json`, `lifecycle.txt`, and `process.jsonl` together. Retain the dump and restore logs, `original.pid`, and `restored.pid` with them; matching application JSON by itself does not prove that CRIU performed the lifecycle. If CRIT is installed, inspect the process tree with:

```bash
crit decode -i runs/cpu-01/images/pstree.img --pretty
```

The image directory can also contain `core-*.img`, `mm-*.img`, `pagemap-*.img`, and raw `pages-*.img`; raw page files are not structured CRIT records. `timing-ns.txt` contains monotonic nanosecond timestamps, so subtract a start value from its matching end value and divide by 1e9 for seconds. `image-directory-bytes.txt` is the apparent size of the whole image directory, including logs. Polling, command launch, and timestamp overhead are included, so these values are initial observations rather than precise benchmarks.

## DMTCP native CUDA investigation — 2026-09-14

DMTCP v4.2.0 introduced a native NVIDIA CUDA plugin. The experiments used maintenance commit `b175bb5ccadd2f02d11cf052f586d2d9ac62ad53` from September 7, 2026, without local source patches and with `--disable-prgname-prefix`. This route launches the workload through DMTCP from the beginning; it is not an attach-to-any-process interface. The CUDA driver still stages GPU state, while DMTCP captures and rebuilds the CPU process. [DMTCP v4.2.0 release](https://github.com/dmtcp/dmtcp/releases/tag/v4.2.0), [pinned CUDA plugin](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/plugin/cuda/cuda-ckpt.cpp).

### CPU restore

Run `dmtcp-cpu-02` passed. The original real PID 11118 exited and real PID 11128 restored from a 38,118,736-byte image. Token, file position, and steps 1–25 matched. Image finalization was observed after 0.510 seconds, before the separate filesystem sync.

The controller's completion condition follows the pinned DMTCP write path. DMTCP writes a name containing `.dmtcp.temp`, waits for the checkpoint writer, and then renames the temporary file to its final name. With `--no-gzip`, the default non-forked checkpoint mode, and a fresh run directory, appearance of the final `*.dmtcp` file is an observable image-finalization boundary for this probe. It remains distinct from the later filesystem sync and does not establish durability after host loss. [Temporary-name definition](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/processinfo.h#L114), [writer wait and final rename](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/dmtcpworker.cpp#L490-L504).

An earlier controller attempt checked for the image too soon after the blocking checkpoint command. Correcting the controller to wait for finalization resolved it; that attempt was not evidence of a DMTCP restore failure.

### GPU process restore

Run `dmtcp-gpu-01` produced a 3,535,032,587-byte image (3,535,036,416 allocated filesystem bytes, about 3.29 GiB) for a 67,108,864-byte tensor payload. The original real PID 11208 exited, job B used the GPU, and restored real PID 11262 returned the same full tensor SHA-256 and passed the next addition endpoint checks. DMTCP preserved virtual PID 40000 inside the application. Image finalization was observed after 1.313 seconds, before sync.

```text
tensor SHA-256 before and after:
bcfcc724743f7bf094ad3ecaf64d1d5fcc08e80c5801a5c00d368c99bcf8f709

memory_matches: true
next_gpu_operation_correct: true
job_B_gpu_sum: 1048576.0
```

The strict controller nevertheless rejected unqualified acceptance because four `/dev/zero (deleted)` shared-memory warnings occurred. DMTCP saves the bytes in those regions but restores them as private anonymous memory, so an original sharing or aliasing relationship can be lost. Four writable 2 MiB mappings with distinct backing identities were present; their ownership and driver-side dependencies remain unknown. A separate session-leader warning also appeared. No unsupported `anon_inode:` mapping appeared in the saved pre-checkpoint maps, but later workloads must be checked independently. [Shared-memory serialization](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/writeckpt.cpp#L479-L494), [kernel-backed mapping diagnostics](https://github.com/dmtcp/dmtcp/blob/b175bb5ccadd2f02d11cf052f586d2d9ac62ad53/src/plugin/ipc/file/fileconnlist.cpp#L451-L463).

This establishes restoration of this complete GPU process image and subsequent GPU computation. It does not establish optimizer-state restoration, stochastic continuation, repeated restore, replacement-host restore, a real spot interruption, or LoRA fine-tuning.

### Build and capacity observations

The active PyTorch runtime used CUDA 13.0, while system `nvcc` and `/usr/local/cuda` development headers were CUDA 12.6. Those headers lacked the checkpoint API declarations; CUDA 13 runtime headers alone then lacked `crt/host_defines.h`. The plugin was compiled with isolated `nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-nvcc==13.0.88`, and `nvidia-cuda-crt==13.0.88` headers and linked the installed `libcuda.so.1`. The existing Python environment was not replaced. The include paths identify the headers used more reliably than the Makefile banner, which still reported the 12.6 `nvcc` on `PATH`.

| Resource | Observation |
| --- | --- |
| GPU | NVIDIA L4, 23,034 MiB total |
| Driver | 595.58.03 |
| Container RAM limit | 133,143,986,176 bytes (124 GiB) |
| `/home` free space | 106,499,379,200 bytes at inspection (about 99.2 GiB) |
| Runtime | PyTorch 2.11.0+cu130; native allocator; no allocator override |
| Training packages | Transformers 5.7.0 installed; PEFT absent |

The cgroup RAM limit, rather than host-wide `free`, controls available staging headroom.

## CRIU process capture — 2026-09-14

On Ubuntu 22.04.5 with Linux 6.8.0-110-generic x86_64, both the packaged CRIU 3.16.1 check and a direct dump of the disposable CPU counter returned exit 1:

```text
Unable create a network namespace: Operation not permitted
kerndat_has_nftables_concat failed when initializing kerndat.
Could not initialize kernel features detection.
```

This happens before counter capture. CRIU performs the nftables check in a temporary network namespace; the counter itself does not require networking. The container's effective and bounding capability masks were `a80425fb`, without `CAP_CHECKPOINT_RESTORE`, `CAP_SYS_PTRACE`, or `CAP_SYS_ADMIN`. `Seccomp: 2` shows filtering is active but does not identify the rejected rule. A direct `unshare --user --map-root-user true` also returned `Operation not permitted`, even though visible user-namespace sysctls were enabled. No security settings were changed. [CRIU 3.16.1 feature check](https://github.com/checkpoint-restore/criu/blob/v3.16.1/criu/kerndat.c), [CRIU 4.2.1 permission documentation](https://github.com/checkpoint-restore/criu/blob/v4.2.1/Documentation/criu.txt#L840-L871).

The result is a container-permission blocker for the CRIU route, not a workload requirement and not proof that CRIU cannot work on a suitably permitted host.

## NVIDIA GPU-only suspension — 2026-09-14

The probe used NVIDIA `cuda-checkpoint` commit `00d5cce84c628088d6caa203fc4af40c1538b6f7`, whose x86_64 utility reported version 595.58.03. The CPU process remained alive throughout; no CRIU image or durable process image existed.

| Observation | Before | Suspended | Restored |
| --- | --- | --- | --- |
| CUDA state | running | checkpointed | running |
| `nvidia-smi` process | PID 1029, 252 MiB | absent | PID 1029, 252 MiB |
| Host RSS | 609,740 KiB | 816,396 KiB | 609,832 KiB |

The 64 MiB tensor hash matched before and after. Job B allocated a CUDA tensor and returned the expected sum 1,048,576 while A was suspended; A then performed its next GPU addition correctly. Both processes exited and the final GPU compute-process listing was empty.

Single-run transition observations, including launch and timestamp overhead, were 0.0283 s to lock, 0.2218 s to checkpoint, 0.2069 s to restore, and 0.0296 s to unlock. These are mechanism observations, not training benchmarks. Suspended host RSS rose by about 202 MiB, showing that driver and allocator state exceeded the visible tensor payload.

## Environment history

The initial September 12 host observation reported driver 580.126.20 with CUDA 13.0 and neither `criu` nor `cuda-checkpoint` on `PATH`. After the instance resumed on September 14, it reported driver 595.58.03. Workspace persistence therefore did not imply an unchanged driver environment. Each experiment must record its live GPU, driver, runtime, tools, permissions, and cgroup limits.

The local development machine was macOS ARM64 with no NVIDIA GPU tools. It is suitable for authoring and CPU-side checks, but NVIDIA CUDA checkpointing is a Linux feature and was validated only on the Linux L4 host.
