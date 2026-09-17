# Plan: independently triggered training, capture, and restore

**Status: proposed; awaiting review.** This document changes the plan, not runtime
behavior. Add it to the current `feat/lora-snapshot` review. After approval, the
user merges that branch, creates a fresh implementation branch from updated
`main`, and implementation proceeds in the milestones below. Do not merge or
begin implementation as part of this planning change.

The outcome is three independently runnable entrypoints: `train.py`,
`checkpoint.py`, and `restore.py`. A capture command can finish and disappear;
a later restore command uses saved artifacts and the documented environment to
resume the trainer. No original controller object or long-lived capture worker
is required. Keep the code small and readable for a CS student without OS
knowledge, following the [teaching plan](readability-plan.md).

## Scope and existing behavior

Keep the pinned same-host, single-GPU FP32 LoRA workload and CRIU CUDA plugin.
Use four optimizer updates, with a one-update capture as the first GPU gate.
Target 2–3 minutes for one warm diagnostic trial; installation, asset downloads,
reference generation, and the complete acceptance matrix are separate costs.
This is a turnaround target, not a promised result.

Today, [criu_pipeline.py](../experiments/finetuning/criu_pipeline.py) calls launch,
dump, handoff, restore, and comparison within one controller run.
[train.py](../experiments/finetuning/train.py) receives predetermined
`--pause-at` updates. The reference/application/CRIU pipeline files are separate
experiments, not separate capture and restore commands.

This change adds external triggering and explicit process ownership. It does not
add an AWS warning listener, object-store upload, instance provisioning, cross-host
restore, arbitrary-point capture, distributed training, a new backend, or an
inference experiment. Manual/scripted triggers exercise the same command boundary
that a future warning receiver or scheduler can call. Existing compatibility
qualifications remain in force.

## Entrypoints and shared code

A program file contains instructions; a process is a running instance with its
own memory. Each entrypoint below must work as a separate process, not merely as
a function imported by the old controller.

| File | Responsibility after the change |
| --- | --- |
| `experiments/finetuning/train.py` | Train; publish job identity; accept a request and acknowledge a completed-update boundary; wait for inspection/release. It does not invoke CRIU. |
| `experiments/finetuning/checkpoint.py` — new | Validate the existing trainer, request capture, invoke CRIU dump, verify exit and image completion, publish the completed snapshot, and exit. It does not restore. |
| `experiments/finetuning/restore.py` — new | Validate a completed snapshot and its dependencies, invoke CRIU restore, inspect the untouched restored state, permit continuation on a match, and exit while training continues. |
| `experiments/finetuning/control.py` — new, small shared module | Read/write the job registration, request acknowledgements, and snapshot manifest. Share only real protocol operations; use plain records and functions. |
| `experiments/criu/session.py` | Keep privileged CRIU calls, bounded process observation, child reaping, and targeted failure cleanup. Separate observing an unrelated process from waiting for an owned child. |
| `run.py`, `criu_pipeline.py`, `pipeline.py` | Remain the experiment harness. Launch the new commands as subprocesses; compare references and continuation, run job B, and report evidence. Remove duplicate capture/restore orchestration once replaced. |
| `state.py`, `metrics.py`, `report.py` | Retain state comparisons; include new source files in fingerprints and adapt event reporting to independent workers. |

Keep reference and application-checkpoint pipelines usable. Extract existing
preflight/identity checks only where both standalone commands need them; avoid
import cycles through `run.py`. The worker command path should not initialize a
model or CUDA context. Prefer one readable main sequence per entrypoint, shared
functions for meaningful operations, and grouped comments for non-obvious steps.
No generic workflow engine, plugin registry, queue service, or class hierarchy.

The proposed interface, to implement on the later branch, is:

```bash
# Run in separate terminals or through the experiment harness.
python train.py --assets ASSETS --run-dir JOB --external-control --until 4
python checkpoint.py --run-dir JOB --snapshot SNAPSHOT --tools TOOLS
python restore.py --snapshot SNAPSHOT --tools TOOLS
```

These are design examples, not commands supported by the current checkout.
Use the existing isolated interpreter in the eventual runnable examples. The
four-update trainer can finish before a human types the next command, so the
reproducible demonstration must automate requests from observable progress.

## Files connect the processes

A **manifest** is a small description of a saved snapshot. It records what was
captured and which external files and environment are still required. A
**marker** is a file that communicates readiness or permission; it is not model
state. Reuse the existing file handshake instead of adding sockets or a server.

```text
JOB/
  job.json                   Current trainer identity and required paths
  control/                   Requests and acknowledgements, each with a unique ID
  ...                        Existing trainer observations and logs
SNAPSHOT/                    A fresh directory for one capture
  manifest.json              Versioned description and image inventory
  images/                    CRIU process/GPU images
  before.json                Diagnostic boundary observation; never loaded as state
  capture-events.jsonl        Capture worker's own events
  COMPLETE                   Published last, after required local persistence
JOB/restore-attempts/ID/      Fresh logs, PID filename, and observations per attempt
```

This is the logical layout; retain existing filenames where compatible. Store
absolute control/model/asset paths in the manifest. The first implementation
requires those paths and their external file contents to remain available on the
same host. A process image does not roll back external files. Do not advertise a
self-contained portable archive or support relocation in this phase.

The records must establish:

- **Identity:** stable job ID, unique capture/request ID, acknowledged update,
  trainer PID, process start identity and host boot identity, plus expected UID
  and job path. A PID alone can be reused for another process. Refresh live
  registration after restore; do not confuse a reused numeric PID with the old
  process generation.
- **Compatibility:** schema version, interpreter/package and tool fingerprints,
  GPU/driver information, workload/source fingerprints, model/data identity,
  and required external paths. Reuse the existing manifest checks.
- **Completeness:** image inventory and integrity hashes, captured observation,
  successful CRIU/plugin evidence, exit verification, and local flush outcome.
  Keep integrity hashing visible in costs; it must not silently disappear from
  capture-to-publication or restore-preflight timing.

Publish each small record by writing a temporary file and renaming it into place
on the same filesystem, so another process sees a complete record. Flush required
files and directories before publishing `COMPLETE`, then persist that publication.
File visibility and durable directory entries are distinct; see the
[Linux fsync documentation](https://man7.org/linux/man-pages/man2/fsync.2.html).
Keep completed snapshot payloads immutable; restore logs and PID files belong to
fresh attempt directories. Partial snapshots are never eligible for restore, and
a failed newer capture must not overwrite a previous completed one.

Use one OS-managed exclusive operation lock per job across capture/restore, with
its handle held by the worker, not the captured trainer. This prevents simultaneous
workers without leaving a permanently held lock after worker death. Also retain
a durable phase record: losing the lock does not mean an interrupted operation
is safe to repeat. The first version refuses ambiguous retries and explains the
required cleanup rather than attempting automatic recovery of a partial dump.

## Runtime sequence

```mermaid
sequenceDiagram
    participant T as Trainer
    participant C as Checkpoint command
    participant D as Snapshot storage
    participant R as Restore command (started later)
    Note over T: Independently running training process
    C-->>T: Request ID: pause after a complete update
    T-->>C: Ready: request ID + actual completed update
    Note over T,C: CRIU + CUDA plugin capture; original exits
    C->>D: Write images, manifest, and boundary evidence
    Note over C: Verify original exit and GPU availability; flush
    C->>D: Publish COMPLETE last
    Note over C: Checkpoint command exits
    D->>R: Read completed snapshot and required-path description
    Note over T,R: CRIU reconstructs trainer at its saved wait
    R-->>T: Inspect saved state before any training changes
    T-->>R: Restored observation
    R-->>T: Continue only after comparison passes
    Note over R: Restore command exits; trainer continues
```

Dashed arrows represent control messages; solid arrows represent saved files
being written/read. CRIU/NVIDIA helper processes and the OS parent/reaper are
omitted from the diagram, but their responsibilities are explicit below.

1. **Register and request.** Trainer publishes its identity and listens at update
   boundaries. Checkpoint validates that identity and publishes one unique
   request. Only acknowledge after at least one update has initialized Adam.
   Record the actual acknowledged update; do not encode it as the request ID.
2. **Finish the whole update.** Complete forward/backward, optimizer and scheduler,
   gradient clearing, temporary cleanup, counters/data cursor, and CUDA
   synchronization. Publish diagnostic evidence and readiness, then wait without
   fetching another example or consuming training randomness. A request arriving
   mid-update takes effect at this boundary. A request arriving after training
   finishes fails clearly rather than pretending to capture.
3. **Capture and publish.** Use the pinned terminating CRIU dump with its CUDA
   plugin as the only owner of CUDA transitions. Require successful tool completion,
   the expected image inventory, verified original exit/reaping, and GPU observation
   after exit. Persist the manifest/images and publish `COMPLETE`; return success
   and exit. Job B remains a separate experiment run after capture and before
   restore; its duration is not snapshot overhead.
4. **Restore and inspect.** Start a new process that receives the snapshot path,
   tools, and documented environment. Validate completion, integrity, compatibility,
   paths, and absence of an already-active job before invoking CRIU. Use a fresh
   restore-attempt ID and PID filename. Reconstruct the trainer, release only its
   inspection wait, compare its untouched state with `before.json`, and then
   permit continuation. No application checkpoint, model initialization, or
   training-state repair is used in this route.
5. **Continue independently.** On successful handoff, the restore command exits
   without running failure cleanup on the trainer. The trainer performs subsequent
   updates and remains observable through files and its refreshed identity.
   Uninterrupted-reference comparisons remain the harness's responsibility;
   restore does not require the old harness or its `Trial` object.

Use a unique capture ID and a separate unique restore-attempt ID. Before the
single supported restore of a capture, reject unexpected old inspect/continue
markers. Keep the initial scope to a sequential lineage: capture A → restore A →
capture B → restore B. Replaying an older snapshot after later execution or
retrying a partially completed restore requires external-file rollback semantics
and is not silently supported.

## Process ownership is the first implementation gate

A parent is the process that created another process. After a child exits, its
parent collects its status; Linux can then remove its remaining process record.
This is **reaping**. The current controller is the original trainer's parent and
adopts restored descendants, so its `Popen.wait()`/`os.waitpid()` assumptions work.
A new capture worker is not automatically the existing trainer's parent.
[Linux wait](https://man7.org/linux/man-pages/man2/waitpid.2.html) and
[subreapers](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)
describe this distinction.

For the standalone demonstration, the original launch shell/service owns and
reaps the original trainer. The short-lived restore worker can temporarily adopt
its restored descendant for failed-attempt cleanup. On success, that trainer
must outlive the worker and pass to the existing host/ancestor reaper. Verify
which process actually adopts it in the approved host execution context; do not
assume that the tool sandbox has the host's PID namespace or working PID 1.
Record the documented launcher/reaper arrangement and any service settings that
would terminate descendants when the worker exits. Existing host supervision is
an environment dependency, not saved recovery state.

Prove this with a tiny CPU-only process lifecycle before modifying GPU orchestration.
The new observation helper waits for an identified unrelated process to exit;
only its actual parent calls `wait`. Do not report the original PID reusable
until the old process record has been removed. Retain the existing regression
protection for zombies and unrelated live children. Use a stable process handle
where supported for targeted signaling, and recheck identity before CRIU receives
its numeric PID. If the launch environment cannot provide the required reaping
and survival behavior, stop this gate with evidence and revise the launch recipe;
do not introduce a hidden permanent Python supervisor.

## Failures, time, and evidence

Every request, CRIU phase, inspection, and exit observation has a bounded wait.
Keep a single caller deadline for the capture operation and report which phase
exhausted it; the current per-command timeouts are not a warning-window promise.
Before dump starts, an explicit cancellation can release the still-live trainer
from a matching request. Once dump may have started, failure leaves a recorded
failed/unknown phase and no `COMPLETE`; do not automatically resume or publish an
ambiguous CUDA/process state. A trainer's control wait also times out visibly.

Reject stale identities, concurrent requests, incomplete/corrupt images,
unsupported manifests, missing assets, and mismatched environments before the
relevant destructive operation. A failed inspection never writes the continue
marker. Preserve evidence and clean up only the positively identified failed
attempt; do not signal a recycled PID or delete earlier completed snapshots.

Each worker writes its own events with capture/attempt IDs, raw monotonic times,
and clock-domain evidence. Record request receipt, readiness, dump completion,
original exit, subsequent GPU observation, local snapshot publication, restore
request/return, verified release, and next completed update. Preserve the existing
CRIU time-namespace conversion when comparing trainer and worker timestamps.

Report request-to-ready, request-to-local-publication, request-to-post-exit GPU
availability, and restore-request-to-next-update separately. Include worker launch,
preflight, hashing, and sync in the appropriate end-to-end interval; keep job B
and deliberate time between the two commands separate. Full diagnostic trial
time remains a separate measure. Local flushing does not establish survival of
instance deletion; future remote publication must add its own completion event.

## Implementation milestones and acceptance

Commit each milestone on the **new implementation branch** after its gate passes.
Do not retroactively relabel historical runs as validating the new commands.

| Milestone | Concrete change | Acceptance gate |
| --- | --- | --- |
| 1. Ownership and protocol | Add minimal control records/publication; split process observation from child reaping; document the standalone launcher. | CPU lifecycle proves an unrelated worker can observe exit and a detached process survives its worker, with eventual reaping. Identity, exclusive-operation, and incomplete-publication failures are covered. |
| 2. External trainer pause | Add `--external-control`; consume requests only at completed updates; preserve existing reference/application modes. | A request during work is acknowledged after a complete update with initialized Adam, synchronized CUDA, and unchanged RNG while waiting. Late/cancelled/stale requests have bounded, explicit outcomes. |
| 3. Standalone capture | Add `checkpoint.py`, fresh snapshot directories, integrity inventory, and durable completion publication. | Capture an independently started one-update LoRA trainer; original exit and post-exit GPU availability are proven; checkpoint command exits; job B succeeds. No restore is performed by this command. |
| 4. Standalone restore | Add `restore.py`, attempt-specific files, untouched-state comparison, refreshed identity, and successful ownership handoff. | With the capture worker and its launcher gone, a fresh restore process reconstructs the trainer from images and required files, exits, and the next update matches the reference. This completes the first new GPU restore gate. |
| 5. Harness and repeated lifecycle | Make the CRIU experiment invoke the entrypoints; remove the replaced in-process path; adapt event/metric collection. | Fresh matching reference pairs and both application/CRIU routes pass with dropout 0 and active 0.1; two sequential capture/restore generations pass without stale markers/PID files. |
| 6. Evidence and teaching | Update both READMEs, code walkthrough, results, and curated evidence; explain requests, manifests, parent ownership, publication, and clocks beside their use. | A reader can reproduce the independent commands; one warm trial is measured against the 2–3-minute target; one timing smoke per route verifies metric wiring. New comparative performance claims require fresh paired repetitions. |

Keep tests necessary and small. Each added test docstring must name its purpose
under the writing-tests discipline: **Critical contract**, **Acceptance criterion**,
**Non-obvious correctness**, or **Regression**. Prefer real small child processes
and temporary directories over mocks of CRIU success. Parameterize closely related
invalid-artifact cases rather than adding one test per field.

The load-bearing checks are independent-worker survival/exit ownership; request
identity and cancellation; single-operation exclusion; rejection of partial or
corrupt publication; and refusal to continue after a mismatched restore.
Exercise interrupted capture and failed restore publication/cleanup paths without
sacrificing the previous valid snapshot. Do not add tests for trivial wrappers
or exact explanatory wording.

GPU acceptance retains full named state comparisons: model tensors, Adam,
schedule, data position, CPU/CUDA RNG, training/evaluation modes, enabled adapters,
trainable flags, and dropout settings. Require the actual LoRA dropout path to run
in training mode and its used CUDA generator to advance during both original and
restored updates. Inspect restored settings before any code can repair them;
matching configuration alone is insufficient. Follow boundary equality with
matching losses and subsequent updates. Numerical, lifecycle, and compatibility
verdicts remain separate; unresolved sharing and resource warnings remain visible.
