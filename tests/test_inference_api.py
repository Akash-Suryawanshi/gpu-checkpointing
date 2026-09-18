"""HTTP boundary contracts: immediate streaming, strict settings, busy rejection, cancellation."""

import http.client
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

    def status(self):
        return {"state": self.state, "activation_id": "act", "last_error": None}

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
