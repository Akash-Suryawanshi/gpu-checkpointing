# Independent lifecycle: implementation contract

**Proposed.** Read the [plan](independent-lifecycle-plan.md) first. Implement on a
new branch after approval and merge of the current PR.

## Files and interface

Add `checkpoint.py`, `restore.py`, and a small `control.py` under
`experiments/finetuning/`. Keep one trainer loop and existing `state.py`
comparisons. `experiments/criu/session.py` owns CRIU/process operations.
`run.py`, `criu_pipeline.py`, and `pipeline.py` become subprocess orchestration;
remove their replaced lifecycle path. Preserve reference/application comparisons.

Share substantive operations, avoid import cycles and framework abstractions,
and never initialize a model or CUDA context in a worker. Include new executable
sources in reference fingerprints. Explain non-obvious blocks concisely; link
[OS fundamentals](readability-plan.md) rather than repeating them here.

Proposed commands, using the pinned interpreter from `experiments/finetuning/`:

```bash
python train.py --assets ASSETS --run-dir JOB --external-control --until 4
python checkpoint.py --run-dir JOB --snapshot SNAPSHOT --tools TOOLS
python restore.py --snapshot SNAPSHOT --tools TOOLS
```

```text
JOB/
  job.json          Live identity and required paths
  control/          Requests, acknowledgements, operation phase
  attempts/ID/      Mutable events, logs, PID files, observations
SNAPSHOT/
  images/           Immutable CRIU payload
  before.json       Comparison evidence, never restored as model state
  manifest.json     Schema, dependencies, identities, sizes, hashes
  COMPLETE          Capture ID and manifest digest
```

Record job/capture/attempt IDs, acknowledged update, PID/start/boot identity,
UID/job path, package/source/tool fingerprints, GPU/driver, model/data identity,
and required absolute paths. Refresh registration after restore. Keep assets,
control paths, and external file contents available; CRIU does not roll back
external files. Mutable logs stay outside the hashed payload.

## Exact local storage protocol

Use a fresh private directory on validated persistent storage; reject `tmpfs`.
Record filesystem/backing volume and capacity for both old and new snapshots.
Require successful CRIU/plugin completion and finished image writers before
inventorying all required files. No page-server or lazy-page dependency is added.

Under the per-job operation lock:

1. Inventory/hash finalized image files and `before.json`.
2. Flush any remaining user-space writer buffers; `fsync` every required payload
   file, then affected image directories, deepest first.
3. Write/flush/`fsync` `manifest.json.tmp`; close and rename to `manifest.json`.
4. `fsync` the snapshot directory and parent, including newly created ancestors
   up to an already-persisted directory.
5. Write/flush/`fsync` `COMPLETE.tmp`; close, rename to `COMPLETE`, then `fsync`
   the snapshot directory again.
6. Persist the successful operation phase with file/directory synchronization;
   emit `local_snapshot_published` and return success.

Rename provides atomic visibility on the same filesystem; synchronization
provides the storage barrier. Do not replace sync with sleeps or ignore directory
sync errors. [Linux fsync](https://man7.org/linux/man-pages/man2/fsync.2.html).
The existing `sync -f` barrier remains until its replacement is validated.

Write/sync failures reject success and preserve the previous snapshot. A crash
around final publication may leave visible `COMPLETE` without acknowledged
success: retain phase evidence and quarantine/withdraw the candidate where
possible. Reject ambiguous attempts pending explicit revalidation. A failing
disk may also prevent recording the error. The original may already have exited;
failed capture cannot promise recovery of its latest work.

Restore acquires the lock and checks terminal phase, completion/manifest digest,
all payload hashes, schema, environment/assets, external paths, stale control
markers, and absence of an active job before invoking CRIU. Checksums and
same-host rereads can use page cache; neither proves power-loss durability.
Volume retention and remote publication are separate future contracts.

## Control and process ownership

Use one OS-managed exclusive lock per job, held by the worker, plus a persisted
phase record. Worker death releases the lock; it does not make a partial operation
safe to retry. Give every request and restore attempt a unique ID and fresh PID
filename. Support only capture A → restore A → capture B → restore B.

Acknowledge requests after initialized Adam and the complete update boundary:
optimizer/scheduler, cleared gradients/temporaries, counters/cursor, CUDA sync.
Record the actual update. Automate requests from progress events to avoid manual
races with the four-update workload. Wait without consuming training RNG.

The launch shell/service reaps the original trainer—collects its exit status.
An unrelated capture worker only observes exit/removal. Recheck PID identity
before CRIU and use stable process handles for signaling where supported.
The restore worker may adopt its descendant for failure cleanup; successful
handoff leaves the trainer alive under a verified host/ancestor reaper.
[Linux wait](https://man7.org/linux/man-pages/man2/waitpid.2.html),
[subreapers](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html).

Validate ownership and service settings with a CPU-only probe on the actual host,
not the tool sandbox. Keep zombie/unrelated-child regression coverage. If the
trainer cannot survive worker exit, revise the launcher before GPU work; do not
add a hidden permanent Python supervisor.

Restore compares untouched state before release. Do not repair modes, adapters,
optimizer, or RNG; never load observation JSON as training state. References and
old `Trial` objects are verification inputs, not recovery dependencies.

## Failures, measurements, and checks

Bound all waits and apply one capture deadline through publication. Cancellation
may release a matching live request before dump starts. After dump may have
started, do not automatically resume unknown CPU/CUDA state. Inspection failure
never publishes continue. Cleanup targets only the identified failed attempt.

Record separate request, ready, dump, original-exit, post-exit GPU, payload-sync,
publication, restore, verified-release, and next-update events. Preserve IDs,
raw monotonic times, and CRIU time-namespace offsets; reject negative latencies.
Include launch/preflight/hash/sync/publication costs. Report job B, the intentional
gap between commands, and full diagnostic duration separately.

Keep focused checks for ownership/survival, request cancellation/exclusion,
publication ordering and injected failures, preservation of an older snapshot,
corrupt artifacts, and blocked continuation after mismatch. Test docstrings must
identify their acceptance/critical/regression/non-obvious purpose. Prefer real
small processes and targeted fault injection; do not mock an entire successful
CRIU lifecycle or claim firmware/power-loss verification.

After source changes, regenerate matching reference pairs. Compare model, Adam,
schedule, data cursor, RNG, modes, adapters, trainable flags, and dropout settings;
verify actual LoRA dropout execution and CUDA RNG advancement before and after
restore, then compare losses and subsequent updates. Preserve qualified
compatibility verdicts and the plan's full milestone matrix.
