# Reusable inference image contract

The [cold-start plan](inference-cold-start-plan.md) requires repeated activation
from one image. Training keeps its single-use protocol; inference images published
with `--contract reusable-inference-v1` follow the versioned contract below.

```text
immutable snapshot + COMPLETE -> attempt A -> serve -> exit/reap
              same image ----> attempt B -> serve -> exit/reap
```

The image is reusable; an activation attempt is not. Scope: sequential activations,
saved PID available, same host/boot and captured paths; no concurrent clones.

## Versioned inference contract

Keep training's current schema and single-use checks unchanged. Add explicit
inference publication state separate from mutable activation state; never reset
the legacy phase to `published` to bypass its consumed-image check.

Under the existing operation lock:

1. Require the previous worker/helper gone and the saved PID free. A conflicting
   unrelated process causes admission failure; never terminate it or auto-retry.
2. Create a fresh attempt directory containing phase, host registration,
   inspection, continuation, requests, replies, and logs.
3. Atomically write `control/activation.json` with snapshot/attempt IDs and an
   attempt-relative request path. Constrain that path beneath the attempt root.
   Write before CRIU starts; the versioned worker boundary reads it before any
   saved inspection/continuation marker.
4. Preserve captured external logs as immutable baseline copies. Materialize
   their required original paths before restore, then redirect worker output to
   attempt logs before serving. Preserve the external-file hash checks and count
   materialization in activation time.
5. Register process identity from the host clock domain. Require the worker's
   matching attempt acknowledgement and inspection before releasing it to serve.
   Cleanup targets the current attempt, never a captured/previous identity.

## Implemented flow

```text
publish: manifest.contract + external/ baseline log copies + COMPLETE   (phase.json: published, never moves)
activate (under lock):
  validate -> previous activation terminal? saved PID free? -> materialize logs from external/
          -> drop consumed inspect marker, reset job.json to the captured registration
  activation.json{restoring, attempt, requests=attempts/<id>/requests} -> CRIU -> inspect/continue
  worker: boundary() returns the attempt -> reads activation.json -> binds to attempts/<id>/requests
          -> redirects stdout/stderr to attempts/<id>/worker.log -> acknowledged.json -> ready.json
release: stop.json -> completed.json -> reap -> activation.json{released}
```

Arrows show control order. `control.validate()` owns admission, `restore.py` writes
activation states instead of phases for this contract, and `worker.attempt_paths()`
rejects any request path outside the attempt that released the process.

## Files and acceptance

Changes belong in `experiments/finetuning/{control,checkpoint,restore}.py`,
`experiments/inference/worker.py`, and `lifecycle.restored_candidate()`; reuse
`experiments/criu/session.py` for process ownership and fresh PID files.

Acceptance: restore, serve, unload, then restore the same image again with exact
outputs and unchanged payload hashes. Reject stale messages, PID conflicts,
changed required files, concurrent activation, and failed/ambiguous cleanup.
Failed attempts stay recorded and cannot authorize continuation or another retry.
