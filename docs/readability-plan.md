# Plan: concise explanations without assumed OS knowledge

<a id="plan-make-the-experiment-readable-without-an-os-course"></a>

Write for a CS student who knows basic Python. Define an unfamiliar concept once
where it is needed, then link back. Keep technical identifiers precise; shorten
repetition rather than removing the reason a step exists.

## What is already explained, and what is missing

The [README](../README.md) provides orientation; the [CPU](01-cpu-checkpointing.md)
and [GPU](02-gpu-checkpointing.md) chapters explain mechanisms. The
[runbook](../experiments/finetuning/README.md) owns commands and the code-reading
path; [results](../experiments/results.md) owns measured evidence. The
[workload contract](implementation-plan.md) and [lifecycle contract](independent-lifecycle-plan.md)
describe the workload and independent worker requirements.

## Teach prerequisites in this order

```text
CPU/RAM/GPU/storage → process, memory, OS resources
                   → consistent capture and restore
                   → state comparison, storage completion, timing
```

These are learning dependencies. Use the same example: finish update 2, capture,
verify original exit, restore, inspect, then permit update 3.

| Topic to strengthen | Location and code connection |
| --- | --- |
| Memory mappings and private/shared backing | CPU chapter; `/proc/PID/maps`. Explain why equal bytes do not establish preserved sharing. |
| File descriptors, pipes, terminals, sessions | CPU chapter; `session.launch()`, `--shell-job`. Distinguish open-resource state from external file contents. |
| Parent/child, zombies, reaping, PID reuse | CPU chapter; `alive()`, `reap()`, cleanup. Explain exit versus collection of exit status. |
| Signals, permissions, namespaces | CPU chapter; CRIU privileges and targeted cleanup. A namespace is an isolated resource view, not a separate kernel. |
| CUDA submission and synchronization | GPU chapter; completed-update boundary and plugin helper-thread coordination. |
| Page cache, rename, file/directory sync | GPU chapter and lifecycle contract; distinguish visibility, local persistence, and survival of instance loss. |
| Clock offsets and memory accounting | GPU chapter; `metrics.py`. Explain monotonic clocks, resident RAM, sampled peaks, and cgroup limits. |

Check additions against primary documentation and pinned source. New prose should
explain the local operation, not duplicate a full chapter.

## Keep the implementation small and traceable

Keep one readable sequence per pipeline. Extract helpers for real reuse or a
subtle contract, not one-line forwarding. Docstrings state purpose and important
preconditions/effects; grouped comments explain non-obvious ordering. Ordinary
assignments and imports need no narration.

Preserve inspection before repair, application RNG restoration last, unique
capture files, verified exit before GPU handoff, one CUDA-restore owner, and
separate diagnostic/timing paths. The runbook already maps these to functions;
do not maintain a second module inventory here.

## Diagrams and a code walkthrough

Use compact diagrams for memory sharing, process ownership, handshake order,
and timing boundaries. Label control requests separately from data movement.
Retain the [interactive CRIU walkthrough](assets/criu-walkthrough.html), its text
fallback, keyboard-accessible controls, and no autoplay. Add interaction only
when it clarifies a state transition.

## Milestones and acceptance

- **Implemented:** dedicated pipelines, inline explanations, and the runbook's
  function/marker walkthrough.
- **Validated:** September 17 CPU/GPU checks; see the
  [record](../experiments/results-detail.md#readability-refactor-validation--2026-09-17).
- **Remaining:** deeper worked examples of shared memory, ownership, storage,
  memory accounting, and clock conversion; review each against the code and
  its diagrams. An editorial shortening pass does not complete these lessons.

Accept explanations when a reader can trace update 2 through restoration to
update 3 and distinguish state equality, process exit, GPU availability, and
storage completion. Validate links, diagrams, and controls. Commit completed
milestones separately for user review.

Use the writing-tests skill before test changes. Prose edits need no tests that
repeat wording and do not refresh historical performance evidence.
