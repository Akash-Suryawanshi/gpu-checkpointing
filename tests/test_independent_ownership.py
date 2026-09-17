"""Host-process ownership checks; no GPU or CRIU emulation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import session


class OwnershipTests(unittest.TestCase):
    def test_unrelated_observer_does_not_reap_and_rejects_reused_identity(self):
        """AC1: a capture worker observes exit; the launch parent owns reaping."""
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.2)"])
        expected = session.identity(child.pid)
        try:
            code = "import json,sys,time,session; session.observe_exit(json.loads(sys.argv[1]),time.monotonic()+5)"
            subprocess.run([sys.executable, "-c", code, json.dumps(expected)], check=True)
            self.assertTrue(Path(f"/proc/{child.pid}").exists())
            self.assertFalse(session.alive(child.pid))
            with self.assertRaises(TimeoutError):
                session.observe_exit(expected, time.monotonic() + .01, removed=True)
            child.wait(timeout=5)
            session.observe_exit(expected, time.monotonic() + 1, removed=True)
            wrong = {**session.identity(os.getpid()), "start_ticks": -1}
            with self.assertRaisesRegex(RuntimeError, "reused"):
                session.observe_exit(wrong, time.monotonic() + 1)
        finally:
            child.kill()
            child.wait()

    def test_descendant_survives_worker_exit_and_is_adopted(self):
        """AC1: a detached trainer outlives its worker under a real ancestor reaper."""
        session.adopt_restored_children()
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid)"
        worker = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        pid = int(worker.stdout)
        expected = session.identity(pid)
        try:
            self.assertTrue(session.matches(expected))
            self.assertEqual(int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1]), os.getpid())
        finally:
            session.kill_identified(expected)
            session.reap(pid)

