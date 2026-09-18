"""HTTP boundary contracts: immediate streaming, strict settings, busy rejection, cancellation."""

import http.client
import http.server
import json
import threading
import time
import unittest

import api
import bench_http


class FakeRuntime:
    """Duck-typed Runtime: two tokens 150 ms apart; records unload calls."""
    def __init__(self):
        self.lock = threading.Lock()
        self.unloads, self.state = 0, "ABSENT"
        self.route = "fresh"

    def status(self):
        return {"state": self.state, "activation_id": "act", "last_error": None, "route": self.route}

    def ensure_ready(self, deadline):
        self.state = "READY"
        return "act"

    def generate(self, request, deadline, on_token=None):
        for index, token in enumerate((11, 12)):
            on_token({"index": index, "token_id": token, "text": chr(64 + token)})
            time.sleep(0.15)
        return {"response": {"tokens": [11, 12], "text": "KL"}, "first_ns": time.monotonic_ns()}

    def unload(self, deadline):
        self.unloads += 1
        self.state = "ABSENT"


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.server = api.serve(self.runtime, "m", "127.0.0.1", 0, request_timeout=5)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, body, path="/generate"):
        connection, response = bench_http.call(self.url, "POST", path, body)
        return connection, response

    def test_tokens_stream_before_done_and_unsupported_settings_are_rejected(self):
        """AC (endpoint boundary): the first token event reaches the client while generation
        continues, so TTFT ends at that event, not at headers or done; any setting the
        greedy worker does not honor is refused with 400 rather than ignored."""
        for bad in ({"model": "m", "prompt": "hi", "temperature": 0.5}, {"model": "other", "prompt": "hi"},
                    {"model": "m", "prompt": "hi", "max_new_tokens": 17}, {"model": "m", "prompt": "hi", "top_p": 0.9},
                    {"model": "m", "prompt": ""}):
            with self.subTest(bad=bad):
                connection, response = self.post(bad)
                self.assertEqual(response.status, 400, response.read())
                connection.close()
        observed = bench_http.request(self.url, "m", "hi", 16)
        events = [e["event"] for e in observed["events"]]
        self.assertEqual(events, ["token", "token", "done"])
        self.assertGreater(observed["done_ns"] - observed["first_ns"], 250_000_000, "first token arrived before done")
        self.assertEqual(observed["events"][-1]["tokens"], [11, 12])

    def test_concurrent_request_and_unload_are_rejected_while_busy_and_disconnect_unloads(self):
        """AC (no queue, cancellation): a second request during generation gets 429 and
        unload gets 409; a client that disconnects mid-stream causes the worker to be
        unloaded instead of finishing into a closed socket."""
        connection, response = self.post({"model": "m", "prompt": "hi"})
        first = response.readline()
        self.assertEqual(json.loads(first)["event"], "token")
        busy, busy_response = self.post({"model": "m", "prompt": "hi"})
        self.assertEqual(busy_response.status, 429)
        busy.close()
        unload, unload_response = self.post(None, "/admin/models/m/unload")
        self.assertEqual(unload_response.status, 409)
        unload.close()
        response.close()  # the response keeps its own socket file; both must close for FIN
        connection.close()
        deadline = time.monotonic() + 3
        while self.runtime.unloads == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.runtime.unloads, 1)
        self.assertFalse(self.runtime.lock.locked())
        unload, unload_response = self.post(None, "/admin/models/m/unload")
        self.assertEqual(unload_response.status, 200)
        self.assertEqual(json.loads(unload_response.read())["state"], "ABSENT")
        unload.close()


class ReadinessTests(unittest.TestCase):
    """A foreign listener on the chosen port must never pass as this server."""

    def setUp(self):
        class Foreign(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")  # plain text, exactly like an unrelated service

            def do_POST(self):
                self.send_response(404)
                self.send_header("Content-Length", "9")
                self.end_headers()
                self.wfile.write(b"Not found")

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Foreign)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_a_foreign_listener_answering_healthz_is_rejected_before_any_trial(self):
        """REGRESSION (EBS HTTP campaign, 2026-09-18): port 8090 was held by an unrelated
        service that answered /healthz with 200 'ok'. The bench script accepted it, the real
        server had died on bind, and the trial failed inside json.loads with no usable
        message. Readiness must identify this server, not merely find a listener."""
        with self.assertRaises(RuntimeError) as caught:
            bench_http.ready(self.url, "qwen3-8b", "fresh", attempts=1, delay=0)
        self.assertIn("did not identify", str(caught.exception))

    def test_readiness_accepts_only_the_expected_model_and_route(self):
        """Critical: two servers configured for different routes share this client. Serving a
        snapshot trial against a fresh server would silently mislabel the headline result."""
        runtime = FakeRuntime()
        runtime.route = "fresh"
        server = api.serve(runtime, "qwen3-8b", "127.0.0.1", 0, request_timeout=5)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            identity = bench_http.ready(url, "qwen3-8b", "fresh", attempts=20, delay=0.05)
            self.assertTrue(identity["server_id"])
            for model, route in (("other-model", "fresh"), ("qwen3-8b", "snapshot")):
                with self.subTest(model=model, route=route), self.assertRaises(RuntimeError):
                    bench_http.ready(url, model, route, attempts=1, delay=0)
        finally:
            server.shutdown()
            server.server_close()
