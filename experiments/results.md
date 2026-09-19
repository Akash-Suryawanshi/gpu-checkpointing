# Experiment results

**Question:** is restoring a checkpointed GPU process faster than starting it fresh?

**Answer:** it works, and it is slower — until both the workload and the reader are
right. Training survives capture exactly. Inference restoration lost on every
model and volume we tried, for one reason: an image is larger than the weights,
so it only wins when reading is fast. It finally won on an H100 once we stopped
reading the image single-threaded: **8.99 s against a 15.80 s cold start**.

Per-run detail, provenance and qualifications: [long-form record](results-detail.md).

| # | Experiment | Verdict |
| --- | --- | --- |
| 1 | Which checkpoint tool works on a GPU process | CRIU + its CUDA plugin; two alternatives rejected |
| 2 | Does LoRA training survive a snapshot | Yes, exactly, at 550× the saved bytes |
| 3 | Is restoring an 8B inference model faster than loading it | No, by 1.2–3.1×, on both volumes |
| 4 | Does a real serving engine change that | Not by itself; yes once verification stopped being serial |

---

## 1. Which tool can checkpoint a GPU process

Four candidates, one survivor.

| Tried | Outcome |
| --- | --- |
| Packaged CRIU 3.16.1 in a container | **Failed.** No `CAP_CHECKPOINT_RESTORE`; it died creating a network namespace before touching the workload. |
| NVIDIA `cuda-checkpoint` alone | **Partial.** Suspends and resumes GPU state in 0.22 s / 0.21 s, but writes no process image and needs the CPU process alive. |
| DMTCP 4.2 native CUDA | **Works, not accepted.** Restored a 64 MiB tensor process, but wrote 3.29 GiB of image and raised four `/dev/zero (deleted)` warnings: it copies shared mappings privately. |
| CRIU 4.2.1 + CUDA plugin, EC2 A10G | **Chosen.** CPU (0.59 s), tensor (13.82 s) and LoRA (63.69 s) processes all restored with exact state. |

Also failed on the way: development started on macOS ARM64, where none of this
runs; DMTCP needed CUDA 13.0 compiler headers while the system had 12.6.

## 2. LoRA training survives a snapshot

Qwen2.5-0.5B, LoRA on `q_proj`/`v_proj`, four updates, captured after update 1, 2
or 3, compared byte-for-byte against an uninterrupted reference.

**Found:** every case matched — weights, Adam state, schedule, cursor, and
Python/CPU/CUDA random state. 30 process-level checks and all 15 GPU cases passed.
A second job allocated the GPU between capture and restore, proving real release.

| Method | Capture → sync | Restore → next update | Saved |
| --- | ---: | ---: | ---: |
| Application checkpoint | 1.30 s | 7.95 s | 6.71 MB |
| CRIU process snapshot | 32.20 s | 4.47 s | 3.69 GB |

**The trade:** CRIU resumes 1.8× faster because nothing is rebuilt, and pays 25×
the capture time and 550× the bytes. Saving state is cheap; saving a process is not.

**What failed first:** a second capture in the same run (CRIU opens its PID file
`O_EXCL`, so filenames must be fresh); a latency calculation that came out
negative (CRIU gives the restored process a different clock namespace); cleanup
that skipped an exited child (an empty `/proc` command line splits to `[b'']`).

## 3. Restoring an 8B model is slower than loading it

Qwen3-8B, one A10G, time to first token. Four ways to reach a ready model.

| Route | Median first token |
| --- | ---: |
| resident (already loaded) | 0.076 s |
| RAM-parked, woken | 4.04 s |
| fresh load | 11.15 s |
| disk snapshot restore | 277.62 s |

With the file cache verified cold — the honest comparison — the gap held on both
volumes, because the snapshot route moves 48.49 GiB against 15.26 GiB:

| Volume | Fresh | Snapshot | Ratio |
| --- | ---: | ---: | ---: |
| EBS gp3 | 131.11 s | 401.23 s | 3.03–3.10 |
| instance NVMe | 57.12 s | 164.10 s | 2.85–3.03 |

### Everything we tried to close that gap

| Attempt | Result |
| --- | --- |
| Hash the image last, so CRIU reads it from cache | Worked: 164.10 s → 119.42 s |
| Check model identity instead of contents at activation | Worked: → 69.84 s, but still 1.21–1.29× a fresh start |
| Move to a 2.5× faster volume | Both routes scaled; the ratio did not move |
| Packed weight file, four readers | **Slower**: 63.08 s against the 54.95 s default |
| `O_DIRECT` on the packed reader | No change; the volume is the limit either way |
| A smaller model (0.5B) | **Worse** ratio: 17.95 s against 11.12 s fresh |
| A compiled model, to give the image derived state | **Capture fails outright**: `handle_device_vma plugin failed` |
| Serve it over HTTP instead of the CLI | No measurable difference, ±0.1 s |
| Repeat everything in a container | Reproduced every host number |

**Why none of it worked:** the raw devices cap at 0.13 and 0.32 GB/s regardless of
concurrency, and our loaders already ran at that speed. The restored process
reopens 92 files and not one is a model file, yet activation still hashed 15.26 GiB
of them. The image is simply bigger than the weights.

**One retracted measurement:** an early probe reported 1.95 GB/s on NVMe from a
0.27 s sample. A longer read did not reproduce it; both figures are kept.

## 4. vLLM: an engine whose startup is real computation

Everything above loads weights and little else, so a snapshot had nothing to win.
vLLM compiles kernels and captures CUDA graphs: weights load in 0.17 s while
`LLM()` takes 22.3 s with a warm compile cache and 47.3 s without.

**Getting it captured at all** cost three failures, none of them the predicted one
(`handle_device_vma` never appeared; the CUDA plugin took about three seconds):

| Failure | Cause | Fix |
| --- | --- | --- |
| `Unknown shit 600 (anon_inode:[io_uring])` | PyTorch's libuv store opens an io_uring ring at world size one | `USE_LIBUV=0` |
| `inet: Connected TCP socket` | the same store connects to itself | `--tcp-established`, with `/usr/sbin` on the privileged path |
| `Cannot re-initialize CUDA in forked subprocess` | preparation let vLLM fork a second engine core | set the engine environment inside `load()` |
| Identical weights failed their own audit | the audit carried vLLM's profiled block count, which comes from free GPU memory | keep it out of the audit |

**A10G, 3 blocks per route:** cold 27.65 s, snapshot 30.31 s — still losing, reading
6.42 GiB against 0.92 GiB.

**H100, a volume 16× faster:** the gap *widened* to 5.96 s (cold 15.96 s, snapshot
21.92 s). Two reasons: the faster GPU cut `LLM()` to 12.1 s, shrinking the prize,
and the bottleneck moved off the disk onto us — payload validation read at
0.52 GiB/s against a device doing 5.4 GB/s.

### Fixing our own reader flips the result

| Payload policy | Check | CRIU | Snapshot | Cold | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `strict-v1` (serial hash) | 14.24 s | 4.64 s | 21.92 s | 15.96 s | loses 5.96 s |
| `publication-only-v1` (no hash) | 0.00 s | 11.20 s | 14.01 s | 15.98 s | wins 1.97 s |
| `parallel-chunked-v1` (16 workers) | 1.54 s | 4.59 s | **8.99 s** | 15.80 s | **wins 6.81 s** |

**The surprise:** deleting the hash returned less than half of what it cost, because
CRIU's restore then grew from 4.64 s to 11.20 s. The hash was doing two jobs —
proving the image unchanged, and pulling 7.37 GiB into the page cache so CRIU read
from memory. Verifying *faster* keeps both; verifying *less* loses one.

Not measured: the same policy on the A10G, where the device is flat-capped and
concurrency should buy nothing. Stated there as an expectation, not a result.

## What we got wrong

| Mistake | Correction |
| --- | --- |
| Reported negative restore latency | Mixed two clock namespaces; conversion now uses logged offsets and rejects negatives |
| Claimed 1.95 GB/s device bandwidth | From a 0.27 s sample; a longer read gave 0.32 GB/s |
| Blamed the A10G loss on storage alone | True there, wrong in general: on the H100 our own serial reader was the limit |
| Compared NVIDIA's 2.4 s restore to our whole activation | Unlike quantities — theirs is a restore path and is not documented as re-hashing the payload |

## Limits

One GPU and one host per result, same boot, three trials per point — enough for
medians, not for tail latency. Single models (Qwen3-8B, Qwen2.5-0.5B), single
node. Nothing here tests recovery on a replacement host, after a driver change,
or after a spot reclaim. CRIU's interrupted-system-call and shared-mapping
warnings are retained rather than suppressed, so same-host compatibility stays
qualified. Campaigns are never pooled across hosts or settings.
