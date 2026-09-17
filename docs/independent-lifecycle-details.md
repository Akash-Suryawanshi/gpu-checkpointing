# Implementation details: independent snapshot commands

**Proposed; awaiting approval.** Start with the
[overview and diagrams](independent-lifecycle-plan.md). This companion page is
for implementing the agreed behavior after the current PR is reviewed/merged and
a new branch is created. No runtime implementation is included in this revision.

## 1. Programs and files

| File | Responsibility |
| --- | --- |
| `experiments/finetuning/train.py` | One training loop, job registration, completed-update request acknowledgement, restored-state observation, and bounded waits. Never invokes CRIU. |
| `checkpoint.py` — new | Validate an existing trainer; request a pause; invoke dump; verify exit; execute the persistence protocol; exit. Never restores. |
| `restore.py` — new | Validate a completed snapshot; invoke restore; inspect untouched state; release only on a match; exit without killing the successfully restored trainer. |
| `control.py` — new | Small shared functions for records, request identities, exclusive operation ownership, and ordered snapshot publication. |
| `experiments/criu/session.py` | Privileged CRIU commands, bounded process observation, child reaping, and targeted failure cleanup. |
| `run.py`, `criu_pipeline.py`, `pipeline.py` | Experiment harness: launch the entrypoints as subprocesses, compare references, run job B, and collect evidence. Remove the replaced in-process lifecycle path. |
| `state.py`, `metrics.py`, `report.py` | State comparison, clock-aware metrics, and curated evidence. Include new executable sources in reference fingerprints. |

Keep the reference and application-checkpoint pipelines. Share existing preflight
checks without importing the entrypoint `run.py` into its own dependencies.
Workers must not initialize a model or a CUDA context. Use plain records, a
straight-through main sequence, and helpers only for meaningful shared operations.
Avoid a generic workflow engine, backend registry, or permanent Python supervisor.

Proposed command interface, **not yet available**:

```bash
# Use the pinned virtualenv interpreter and run from experiments/finetuning/.
python train.py --assets ASSETS --run-dir JOB --external-control --until 4
python checkpoint.py --run-dir JOB --snapshot SNAPSHOT --tools TOOLS
python restore.py --snapshot SNAPSHOT --tools TOOLS
```

The final examples must use explicit existing interpreter/tool paths. Automate
requests from progress observations: a four-update trainer may finish before a
person can type the capture command. Record the actual acknowledged update;
the external request must not depend on the trainer's old `--pause-at` setting.

```text
JOB/
  job.json                  Current job identity and required paths
  control/                  Unique requests, acknowledgements, and operation phase
  attempts/ID/              Worker events, CRIU logs, PID files, restored observations
  ...                       Existing trainer observations/logs
SNAPSHOT/                   New private directory, one per capture
  images/                   Immutable CRIU image payload
  before.json               Diagnostic observation, never loaded as training state
  manifest.json             Version, identities, dependencies, payload hashes
  COMPLETE                  Manifest digest and capture ID; published last
```

Mutable attempt logs stay outside the immutable snapshot; otherwise a final
publication event or restore log would change already-hashed files. The first
version keeps captured absolute model/asset/control paths and required external
file contents in place on the same host. Images do not roll back external files;
this is not a relocatable archive. Reuse existing filenames when compatible.

Records must contain stable job ID; unique capture/request and restore-attempt
IDs; actual update; PID plus process start and host boot identity; expected UID
and job path; schema version; interpreter/package/source/tool fingerprints;
GPU/driver and workload/model/data identity; required paths; and image inventory,
sizes, and hashes. Refresh live registration after restore. A reused numeric PID
is not the old process generation.

## 2. Exact local storage protocol

Choose a fresh private snapshot directory on the validated persistent filesystem;
record its mount/filesystem and backing volume. Reject a RAM-only destination
such as `tmpfs`. Available space must cover the new image while retaining the
previous completed snapshot. Do not overwrite the previous snapshot in place.
No remote/page-server/lazy-page image dependency is added in this phase.

The worker holds the per-job operation lock through capture and publication.
Once CRIU returns successfully, image writers must have finished before hashing
or synchronization begins. Inventory every file required by the selected CRIU
route; `inventory.img` and `pstree.img` alone do not describe the whole payload.

| Order | Planned operation | Reason |
| --- | --- | --- |
| 1 | Validate CRIU return status and plugin evidence; enumerate and hash the finalized image payload and `before.json`. | Establish a complete capture and an integrity inventory. A hash is not a storage flush. |
| 2 | For each required payload file, flush any writer's user-space buffer, then call `os.fsync()` on its open file descriptor. Fail on errors. | Ask Linux to complete outstanding writes and wait for the storage acknowledgement. CRIU's own buffered writers have already closed at successful exit. |
| 3 | `fsync()` affected image subdirectories, deepest first. | Persist the filenames used to locate the payload. |
| 4 | Write `manifest.json.tmp`, flush, and `fsync()` it; close it; rename to `manifest.json` in the same directory. | Publish a complete manifest that describes the fixed payload. |
| 5 | `fsync()` the snapshot directory and its parent, including any newly created ancestors up to an already-persisted directory. | Persist the manifest name and the new snapshot directory's link from its parent. |
| 6 | Write `COMPLETE.tmp` with capture ID and manifest digest; flush and `fsync()`; close; rename to `COMPLETE`; `fsync()` the snapshot directory again. | Publish and persist completion only after its prerequisites are persisted. |
| 7 | Persist the successful terminal operation phase using an atomic record write plus file/directory sync; emit `local_snapshot_published`, then return success. | Distinguish durable publication from dump completion and from an unacknowledged attempt. The phase record is part of the documented control-path dependency. |

A **file descriptor** is the small integer handle Linux returns when a process
opens a file or directory. `flush()` moves a language/library buffer toward Linux;
`fsync()` requests storage completion. Rename makes the final name visible as one
operation on the same filesystem; it does not replace synchronization.
[Linux write](https://man7.org/linux/man-pages/man2/write.2.html) and
[fsync](https://man7.org/linux/man-pages/man2/fsync.2.html) define these boundaries.

Do not add sleeps as a substitute for completion. Do not silently skip a failed
or unsupported directory sync. Reuse this publication helper for genuinely shared
record writes, without giving ephemeral readiness markers an unnecessary bulk
storage flush. Keep checksum, flush, directory, and completion costs in timings.

A failure before completion publication leaves an ineligible candidate. A failure
or worker interruption around the last rename/directory sync can leave a visible
`COMPLETE` without an acknowledged successful operation. Report failure, retain
phase/error evidence, and withdraw/quarantine the candidate where possible.
Restore checks the operation phase as well as the completion record and hashes;
an ambiguous attempt needs explicit revalidation before it becomes eligible.
Never promise that a failing disk will reliably persist an additional error file.
A missing success acknowledgement can conservatively reject a usable snapshot;
it must not release a trainer from an unverified candidate.

A newly started restore worker must take the operation lock, validate the manifest
digest and every required payload hash, verify the documented environment/assets,
and reject unknown schemas, partial/failed phases, old control markers, or an
already-active job before invoking CRIU. Saved observation JSON is comparison
input only; it is never a source of replacement model/optimizer/RNG values.

The current implementation has a filesystem-wide `sync -f` after dump in
[pipeline.py](../experiments/finetuning/pipeline.py). Linux waits for the relevant
writeback in `syncfs`; see [Linux sync semantics](https://man7.org/linux/man-pages/man2/sync.2.html).
The new explicit file/directory protocol defines publication per snapshot and
must be validated before replacing that barrier. A normal file reread can use
page cache, so matching hashes after process exit do not independently prove
power-loss survival. No reboot/power-cut test is authorized by this plan.

## 3. Request and restoration boundaries

A request arriving during an update becomes pending. Acknowledge only after at
least one update has initialized Adam, and after forward/backward, optimizer and
scheduler steps, gradient clearing, temporary cleanup, counter/data-cursor
updates, and `torch.cuda.synchronize()`. GPU synchronization waits for submitted
GPU work; it does not flush checkpoint files to storage.

Publish boundary evidence and readiness with the request ID and actual update.
While waiting, do not fetch the next example or consume training randomness.
The worker uses the pinned terminating dump; the CUDA plugin owns NVIDIA
lock/checkpoint/restore/unlock exactly once. No manual second restore is appended.

After successful capture/publication, a new restore command reconstructs the
trainer at its saved inspection wait. Use an attempt-specific PID file and logs,
and compare untouched restored state with the captured observation before writing
continue. Do not call `model.train()` or reload state to repair a mismatch.
The harness compares the boundary and later updates with fresh uninterrupted
references; its objects and reference directory are not recovery dependencies.

Use unique capture and restore-attempt IDs, not just an update number or PID.
Support a sequential lineage: capture A → restore A → capture B → restore B.
Reject leftover inspect/continue markers. Replaying an older snapshot after later
execution, or automatically retrying a partially completed restore, would require
external-file rollback rules and remains outside this phase.

## 4. Who owns exited and restored processes

A **parent** is the process that created another process. After its child exits,
the parent collects the exit status so Linux can remove the remaining record;
this is **reaping**. Our current `session.reap()` assumes child ownership. A later
capture worker is not automatically the original trainer's parent.
[Linux wait](https://man7.org/linux/man-pages/man2/waitpid.2.html).

The original launch shell/service owns and reaps the trainer. Capture only observes
that identified process's exit and removal. Use a stable process handle where
supported for targeted signaling, and recheck identity before passing a numeric
PID to CRIU. Keep the existing zombie and unrelated-live-child regression checks.

The restore worker may temporarily adopt its descendant for failed-attempt cleanup.
After a successful handoff it exits without killing the trainer. A functioning
host/ancestor reaper must then adopt the trainer and later collect its status.
A subreaper adopts orphaned descendants; it does not acquire arbitrary unrelated
processes. [Linux subreaper semantics](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html).

Prove this arrangement with a tiny CPU-only lifecycle in the actual approved host
context first. Record the real parent/reaper and service settings; do not infer
host behavior from the tool sandbox's PID namespace. The environment must not kill
the trainer when its restore worker exits. If this gate fails, document the result
and revise the launch recipe before GPU work. Existing host supervision is an
environment dependency, not a requirement to preserve the old Python controller.

## 5. Failures and measurements

One OS-managed exclusive lock per job covers capture/restore; its handle belongs
to the worker, not the captured trainer. Worker death releases the lock, but a
persisted operation phase prevents an ambiguous partial operation from being
silently repeated. A lock alone does not establish that the previous work succeeded.

All waits are bounded, including trainer control waits. A caller deadline covers
the capture request through publication; report the phase that exhausted it.
Explicit cancellation before dump starts can release a still-live trainer from
that request. Once dump may have started, do not automatically resume unknown
CPU/CUDA state. A failed inspection never creates a continue marker. Cleanup may
only affect the positively identified failed attempt and must preserve previous
completed snapshots.

Workers write separate events with capture/attempt IDs, raw monotonic timestamps,
and clock-domain evidence. Keep request receipt, trainer readiness, dump completion,
original exit, GPU observation after exit, payload persistence, completion
publication, restore request/return, verified release, and next-update completion
separate. Preserve the existing CRIU time-namespace conversion; reject negative
cross-process latencies.

Report request-to-ready, request-to-local-publication, request-to-post-exit GPU
availability, and restore-request-to-next-update. Include worker launch, preflight,
hashing, file/directory synchronization, and publication in their proper intervals.
Exclude the intentional gap between commands and report job B separately. Full
diagnostic-trial duration is another measurement. Remote publication would need
its own completion event in a later experiment.

## 6. Acceptance checks worth keeping

Use the writing-tests discipline: every added test docstring identifies a
**Critical contract**, **Acceptance criterion**, **Non-obvious correctness**, or
**Regression**. Prefer real small processes and temporary directories; use focused
failure injection for write/sync/rename errors rather than mocking a whole CRIU
success. Parameterize related failure points instead of one test per field.

- Prove unrelated-process observation, final reaping, and trainer survival after
  the restore worker exits. Keep targeted cleanup checks.
- Prove request identity/cancellation and exclusion of simultaneous operations.
- Verify publication ordering and successful completion only after the final
  required directory sync; inject failures at payload, manifest, and completion
  stages. An interrupted/ambiguous publication is rejected pending revalidation;
  a previous completed snapshot remains unchanged.
- Reject missing/corrupt files and mismatched restored state without releasing
  another update. An independent worker must use only saved artifacts and the
  specified environment, not the old `Trial` object.

These tests validate our protocol and error handling; they do not emulate drive
firmware or establish abrupt host-loss durability. Record the filesystem/storage
assumptions alongside results rather than claiming physical persistence from
mocks, process termination, or checksum reads.

GPU acceptance retains named model/Adam/schedule/data/RNG comparisons, module
training modes, active adapters, trainable flags, and dropout probabilities.
Require the actual LoRA dropout path to run in training mode and its used CUDA
generator to advance in original and restored updates. Establish fresh matching
reference pairs after executable-source changes, then check subsequent losses
and updates. Run the full milestone matrix at dropout 0 and 0.1, both restoration
routes, and two sequential capture generations. Keep numerical, lifecycle, and
compatibility verdicts separate; warnings about shared-memory/resource ownership
remain qualifications.
