"""Thin HTTP adapter over one Runtime; it holds no CUDA state and never queues.

POST /generate streams newline-delimited JSON events: token, done, or error.
TTFT for a client ends at its first token event, never at headers.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import select
import signal
import socket
import threading
import time
import uuid

from runtime import Runtime

MAX_NEW_TOKENS = 16


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # keep the server quiet; evidence lives in files
        pass

    @property
    def runtime(self):
        return self.server.runtime

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _client_gone(self):
        # A small write into a closed socket can still succeed; a readable socket
        # that yields EOF is the reliable sign the client has left.
        readable, _, _ = select.select([self.connection], [], [], 0)
        if not readable:
            return False
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _chunk(self, event):
        # Chunked transfer lets each event leave the process immediately.
        if self._client_gone():
            raise BrokenPipeError("client disconnected")
        data = (json.dumps(event) + "\n").encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/healthz":
            # Identifies this server, so a client cannot mistake an unrelated
            # listener on the same port for a ready model endpoint.
            return self._json(200, {"ok": True, "server_id": self.server.server_id,
                                    "model": self.server.model_id, "route": self.runtime.route})
        if self.path == f"/models/{self.server.model_id}":
            return self._json(200, self.runtime.status())
        self._json(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path == f"/admin/models/{self.server.model_id}/unload":
            return self._unload()
        if self.path != "/generate":
            return self._json(404, {"error": "unknown path"})
        try:
            request = validate(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))),
                               self.server.model_id)
        except ValueError as error:
            return self._json(400, {"error": str(error)})
        if not self.runtime.lock.acquire(blocking=False):
            return self._json(429, {"error": "busy: one request at a time"})
        try:
            self._generate(request)
        finally:
            self.runtime.lock.release()

    def _unload(self):
        if not self.runtime.lock.acquire(blocking=False):
            return self._json(409, {"error": "busy"})
        try:
            self.runtime.unload(time.monotonic() + self.server.request_timeout)
            self._json(200, self.runtime.status())
        except Exception as error:
            self._json(500, {"error": f"{type(error).__name__}: {error}", **self.runtime.status()})
        finally:
            self.runtime.lock.release()

    def _generate(self, request):
        deadline = time.monotonic() + self.server.request_timeout
        phases = {"received_ns": time.monotonic_ns()}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        request_id = request["request_id"]
        try:
            try:
                activation = self.runtime.ensure_ready(deadline)
                phases["ready_ns"] = time.monotonic_ns()
                observed = self.runtime.generate({**request, "request_id": request_id}, deadline,
                    on_token=lambda event: self._chunk({"event": "token", "request_id": request_id,
                        "activation_id": activation, "index": event["index"], "token_id": event["token_id"],
                        "text": event["text"]}))
                phases.update(first_token_ns=observed["first_ns"], done_ns=time.monotonic_ns())
                self._chunk({"event": "done", "request_id": request_id, "activation_id": activation,
                             "tokens": observed["response"]["tokens"], "text": observed["response"]["text"],
                             "server_phases_ns": phases})
            except (BrokenPipeError, ConnectionResetError):
                # The client left: stop and reap the worker rather than serve into the void.
                self.runtime.unload(time.monotonic() + self.server.request_timeout)
                raise
            except Exception as error:
                self._chunk({"event": "error", "request_id": request_id, "reason": f"{type(error).__name__}: {error}",
                             **self.runtime.status()})
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def validate(body, model_id):
    """Reject anything the greedy worker does not honor rather than silently ignoring it."""
    if not isinstance(body, dict) or body.get("model") != model_id:
        raise ValueError("unknown model")
    if body.get("temperature", 0) != 0:
        raise ValueError("only temperature 0 (greedy) is supported")
    count = body.get("max_new_tokens", MAX_NEW_TOKENS)
    if type(count) is not int or not 1 <= count <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be an integer in 1..{MAX_NEW_TOKENS}")
    if not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise ValueError("prompt must be a nonempty string")
    unknown = set(body) - {"model", "prompt", "max_new_tokens", "temperature", "request_id"}
    if unknown:
        raise ValueError(f"unsupported settings: {sorted(unknown)}")
    return {"prompt": body["prompt"], "max_new_tokens": count,
            "request_id": body.get("request_id") or uuid.uuid4().hex}


def install_shutdown(server):
    """Take over the termination signals, whatever disposition we inherited.

    A non-interactive shell sets background jobs to ignore SIGINT, and Python
    keeps an inherited SIG_IGN. Without this the server cannot be stopped by the
    campaign script, which then waits on a process that will never exit.
    """
    def stop(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)
    return stop


def serve(runtime, model_id, host, port, request_timeout):
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.runtime, server.model_id, server.request_timeout = runtime, model_id, request_timeout
    server.server_id = uuid.uuid4().hex
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "tools", "root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route", choices=("fresh", "snapshot"), required=True)
    parser.add_argument("--snapshot-run", type=Path, help="Run directory holding a reusable published snapshot")
    parser.add_argument("--model-id", default="qwen3-8b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--poll-ms", type=float, default=5)
    parser.add_argument("--sample-ms", type=float, default=100)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--integrity-policy", default="strict-v1")
    args = parser.parse_args()
    runtime = Runtime(args.assets, args.tools, args.root, args.route, snapshot_run=args.snapshot_run,
                      poll_ms=args.poll_ms, sample_ms=args.sample_ms, model_policy=args.integrity_policy)
    server = serve(runtime, args.model_id, args.host, args.port, args.request_timeout)
    install_shutdown(server)
    print(f"serving {args.route} on http://{args.host}:{args.port} server_id={server.server_id}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.unload(time.monotonic() + 60)
