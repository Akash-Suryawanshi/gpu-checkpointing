"""One owned model worker: activation, request dispatch, unload, and cleanup.

The trial CLI and the HTTP adapter share these functions so an endpoint measures
the same worker protocol the CLI validated. Nothing here holds CUDA state.
"""

import json
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid

import lifecycle
import park
from worker import control
from control import session

HERE = Path(__file__).resolve().parent
CACHE_KEYS = ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH")


def comparison_key(deps, manifest, env, **settings):
    """Everything that must match between a diagnostic and the runs it admits."""
    return {"dependencies": deps, "assets": manifest, "bundle": None, "loader": "transformers",
            "validation_order": "payload-first", "validation_workers": 1, "data_cache": "uncontrolled",
            # Version the boundary so HTTP and CLI evidence never pool with historical runs.
            "boundary_schema": 2, "integrity_policy": "strict-v1",
            "inspection_policy": "full-diagnostic-post-response-timing-v1",
            "cache_paths": {k: env[k] for k in CACHE_KEYS if k in env}, **settings}


def admit(output, action, memory, deadline, identity=None):
    """Record every admission decision, including refusals, before acting."""
    pid = identity["pid"] if identity else None
    observed = park.resources(deadline, pid)
    record = {"observed": observed, "memory": memory, "action": action}
    try:
        record["requirements"] = park.admission(observed, memory, action, pid)
    except Exception as error:
        record["refusal"] = str(error)
        raise
    finally:
        control.write(output / f"admission-{action}.json", record)
    return observed


def dispatch(requests, request, identity, deadline, poll, on_token=None):
    """Publish one request and return its observation in the controller's clock."""
    tokens = requests / f"tokens-{request['index']}.jsonl"
    body = {k: v for k, v in request.items() if k != "index"}
    request_start = time.monotonic_ns()
    control.write(requests / f"request-{request['index']}.json", body)
    published = time.monotonic_ns()
    received, seen = None, 0
    while True:
        if tokens.exists():
            with tokens.open() as stream:
                lines = stream.read().splitlines()
            for line in lines[seen:]:
                event = json.loads(line)
                if event["run_id"] != body["run_id"] or event["request_id"] != body["request_id"]:
                    raise ValueError("Token identity mismatch")
                if received is None:
                    received = time.monotonic_ns()
                if on_token is not None:
                    on_token(event)
            seen = len(lines)
        response_path = requests / f"response-{request['index']}.json"
        if response_path.exists() and received is not None:
            response = control.read(response_path)
            break
        control.remaining(deadline)
        if identity is not None and not session.matches(identity):
            raise RuntimeError("Identified trainer exited while waiting")
        time.sleep(poll)
    if any(response[k] != body[k] for k in ("run_id", "request_id")):
        raise ValueError("Response identity mismatch")
    first = control.read(requests / f"first-{request['index']}.json")
    return {"request": body, "first": first, "response": response, "request_start_ns": request_start,
            "published_ns": published, "first_ns": received, "completed_ns": time.monotonic_ns()}


def stop_worker(requests, run_id, identity, process, deadline, poll):
    """Ask the worker to exit, collect it, and require the GPU to be free afterwards."""
    control.write(requests / "stop.json", {"run_id": run_id})
    control.wait(requests / "completed.json", deadline, identity, poll)
    code = session.reap(identity["pid"], process, timeout=control.remaining(deadline))
    if park.resources(deadline)["compute_pids"]:
        raise RuntimeError("GPU compute process remains")
    return code


class Runtime:
    """Serve one model through an owned worker; states: ABSENT, ACTIVATING, READY, FAILED."""

    def __init__(self, assets, tools, root, route, *, snapshot_run=None, poll_ms=5.0, sample_ms=100.0,
                 activation_timeout=1800.0):
        self.assets, self.tools, self.root = Path(assets).resolve(), Path(tools).resolve(), Path(root).resolve()
        self.route, self.snapshot_run = route, Path(snapshot_run).resolve() if snapshot_run else None
        if (route == "snapshot") != (self.snapshot_run is not None):
            raise ValueError("Snapshot route requires a published run directory")
        self.poll, self.sample_ms, self.activation_timeout = poll_ms / 1000, sample_ms, activation_timeout
        self.helper = self.tools / "cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint"
        self.env = session.child_environment(self.helper)
        self.worker_script = HERE / "worker.py"
        self.lock = threading.Lock()
        self.state, self.activation_id, self.last_error = "ABSENT", None, None
        self.active = None  # {"dir", "requests", "run_id", "identity", "process", "index", "sampler"}
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        session.adopt_restored_children()

    def status(self):
        return {"state": self.state, "activation_id": self.activation_id, "last_error": self.last_error,
                "route": self.route, "busy": self.lock.locked()}

    # -- activation -------------------------------------------------------------------
    def ensure_ready(self, deadline):
        if self.state == "READY":
            return self.activation_id
        if self.state != "ABSENT":
            raise RuntimeError(f"Model is {self.state}; unload or clean up first")
        activation_id = uuid.uuid4().hex
        folder = self.root / "activations" / activation_id
        folder.mkdir(mode=0o700, parents=True)
        self.state, self.activation_id, self.last_error = "ACTIVATING", activation_id, None
        events = lambda name, **values: control.event(folder, name, activation_id=activation_id, **values)
        try:
            events("activation_started", route=self.route)
            if self.route == "fresh":
                self.active = self._launch(folder, activation_id, deadline, events)
            else:
                self.active = self._restore(folder, activation_id, deadline, events)
            control.wait(self.active["requests"] / "ready.json", deadline, self.active["identity"], self.poll)
            events("worker_ready_observed")
            self.state = "READY"
            return activation_id
        except BaseException as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self.state = "FAILED"
            events("activation_failed", reason=self.last_error)
            self._cleanup()
            raise

    def _key(self, deadline, **settings):
        from prepare_assets import validate_assets
        manifest = validate_assets(self.assets)
        deps = control.dependencies({"assets": str(self.assets), "python": sys.executable}, self.tools, deadline)
        return manifest, comparison_key(deps, manifest, self.env, poll_ms=self.poll * 1000, sample_ms=self.sample_ms,
                                        transport="http", **settings)

    def _launch(self, folder, activation_id, deadline, events):
        manifest, key = self._key(deadline)
        run_id = uuid.uuid4().hex
        control.write(folder / "run.json", {"run_id": run_id, "route": "fresh", "kind": "timing",
                                            "activation_id": activation_id, "key": key})
        for source, target in (("reference.json", "reference.json"), ("tokens.json", "tokens.json"),
                               ("audit.json", "prepared-audit.json")):
            shutil.copyfile(self.assets / source, folder / target)
        if park.resources(deadline)["compute_pids"]:
            raise ValueError("GPU must start without foreign compute processes")
        sampler = park.Sampler(folder, self.sample_ms / 1000, deadline)
        sampler.thread.start()
        process = session.launch(sys.executable, self.worker_script, ["--assets", self.assets, "--run-dir", folder,
            "--route", "fresh", "--kind", "timing", "--run-id", run_id], folder, self.env)
        identity = session.identity(process.pid)
        sampler.pid = process.pid
        events("worker_launched", identity=identity)
        active = {"dir": folder, "requests": folder, "run_id": run_id, "identity": identity,
                  "process": process, "index": 0, "sampler": sampler, "run": folder}
        control.wait(folder / "idle.json", deadline, identity, self.poll)
        admit(folder, "loaded", control.read(folder / "memory.json"), deadline, identity)
        return active

    def _restore(self, folder, activation_id, deadline, events):
        run = self.snapshot_run
        snapshot = control.read(run / "snapshot/manifest.json")
        if snapshot.get("contract") != control.CONTRACT:
            raise ValueError("Snapshot was not published with the reusable contract")
        saved = control.read(run / "run.json")
        control.write(folder / "run.json", {"run_id": saved["run_id"], "route": "snapshot", "kind": saved["kind"],
                                            "activation_id": activation_id, "snapshot_run": str(run), "key": saved["key"]})
        for name in ("reference.json", "tokens.json", "prepared-audit.json"):
            shutil.copyfile(run / name, folder / name)
        admit(folder, "restore", snapshot["pre_staging_memory"], deadline)
        sampler = park.Sampler(folder, self.sample_ms / 1000, deadline)
        sampler.thread.start()
        events("restore_command_started")
        identity = lifecycle.restore(run, self.tools, self.env, deadline)
        sampler.pid = identity["pid"]
        record = control.activation(run)
        requests = run / record["requests"]
        events("restore_command_exited", attempt_id=record["attempt_id"])
        acknowledged = control.wait(run / "attempts" / record["attempt_id"] / "acknowledged.json",
                                    deadline, identity, self.poll)
        if acknowledged["attempt_id"] != record["attempt_id"] or acknowledged["pid"] != identity["pid"]:
            raise ValueError("Restored worker acknowledged another attempt")
        return {"dir": folder, "requests": requests, "run_id": saved["run_id"], "identity": identity,
                "process": None, "index": 0, "sampler": sampler, "run": run, "attempt": record}

    # -- requests ---------------------------------------------------------------------
    def generate(self, request, deadline, on_token=None):
        """Serve one request; the caller holds self.lock so the worker sees one at a time."""
        if self.state != "READY":
            raise RuntimeError(f"Model is {self.state}")
        active = self.active
        index = active["index"] + 1
        if index == 2:
            # The worker audits full weights after response one, outside that response's timing.
            audit = control.wait(active["requests"] / "audit-after.json", deadline, active["identity"], self.poll)
            if audit != control.read(active["dir"] / "prepared-audit.json"):
                self.state, self.last_error = "FAILED", "Immutable model state changed"
                raise ValueError(self.last_error)
        body = {"run_id": active["run_id"], "request_id": request.get("request_id") or uuid.uuid4().hex,
                "index": index, "max_new_tokens": request["max_new_tokens"]}
        body["prompt" if "prompt" in request else "input_ids"] = request.get("prompt", request.get("input_ids"))
        active["index"] = index
        try:
            observation = dispatch(active["requests"], body, active["identity"], deadline, self.poll, on_token)
        except BaseException as error:
            self.state, self.last_error = "FAILED", f"{type(error).__name__}: {error}"
            raise
        with (active["dir"] / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps({**observation, "activation_id": self.activation_id}) + "\n")
        return observation

    # -- unload / cleanup -------------------------------------------------------------
    def unload(self, deadline):
        """Stop the worker, collect it, and verify GPU release; then release the attempt."""
        if self.state == "ABSENT":
            return
        if self.state == "READY":
            active = self.active
            try:
                stop_worker(active["requests"], active["run_id"], active["identity"], active["process"], deadline, self.poll)
                active["identity"] = None
            except BaseException as error:
                self.state, self.last_error = "FAILED", f"{type(error).__name__}: {error}"
                raise
            finally:
                self._cleanup()
        else:
            self._cleanup()

    def _cleanup(self):
        """Kill and reap owned processes. Success reaches ABSENT; failure blocks reuse."""
        active, self.active = self.active, None
        try:
            if active is not None:
                lifecycle.cleanup(active["run"], active.get("identity"), active.get("process"))
                if active.get("sampler"):
                    active["sampler"].close()
                if active.get("attempt"):
                    self._release_attempt(active["run"], active["attempt"])
        except BaseException as error:
            self.state, self.last_error = "FAILED", f"cleanup failed: {error}"
            raise
        self.state = "ABSENT"

    def _release_attempt(self, run, attempt):
        """Only a completed attempt becomes terminal; restore's own failure record stays."""
        with control.lock(run):
            current = control.activation(run)
            if (current and current["attempt_id"] == attempt["attempt_id"]
                    and current["status"] in ("restoring", "verified", "restored")):
                control.activation_state(run, "released", **{k: attempt[k] for k in
                                         ("capture_id", "attempt_id", "snapshot", "requests")})
