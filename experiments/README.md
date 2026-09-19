# Experiments

```text
tooling          training           inference activation     serving engine
CRIU? DMTCP?  -> LoRA capture  ->  4 routes, 8B  -> vLLM cold  -> H100, then
cuda-checkpoint  and restore       cold cache,       vs snapshot   payload policy
                                   storage, loaders
2026-09-12/14    09-14/17          09-17/18          09-18         09-18/19
```

**Findings: [results](results.md)** — what was done, what was found, what failed.
The [long-form record](results-detail.md) keeps per-run detail and provenance.

| Directory | What it runs | Commands |
| --- | --- | --- |
| `cpu/` | CRIU on a CPU-only workload | [host checks and probes](results-detail.md#run-the-maintained-experiments) |
| `gpu/`, `dmtcp/` | the two rejected tools: NVIDIA's helper alone, and DMTCP | Same |
| `criu/` | the shared CRIU session, and the isolated tool build | `bash experiments/criu/build.sh runs/tools` |
| `finetuning/` | LoRA capture, restore and continuation | [runbook](finetuning/README.md) |
| `inference/` | four activation routes, cold cache, loaders, HTTP endpoint | [runbook](inference/README.md) |
| `vllm/` | a serving engine: cold start against snapshot restore | [runbook](vllm/README.md) |

`evidence/` holds curated validation records; raw images and logs stay in the
ignored `runs/` directory, because they can contain memory and environment values.
