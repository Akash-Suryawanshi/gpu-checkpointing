"""Concurrent validation preserves byte identities and failure propagation."""

from pathlib import Path
import tempfile
import time
import unittest

import control


class HashTests(unittest.TestCase):
    def test_concurrent_digests_match_serial_and_propagate_missing_files(self):
        """Critical: concurrency must not skip corrupt/missing files or alter snapshot identity."""
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / str(i) for i in range(5)]
            for i, path in enumerate(paths):
                path.write_bytes(bytes([i]) * 12345)
            deadline = time.monotonic() + 5
            expected = control.hash_files(paths, deadline, 1)
            self.assertEqual(control.hash_files(paths, deadline, 4), expected)
            paths[0].write_bytes(b"changed")
            self.assertNotEqual(control.hash_files(paths, deadline, 4), expected)
            paths[1].unlink()
            with self.assertRaises(FileNotFoundError):
                control.hash_files(paths, deadline, 4)
            with self.assertRaises(TimeoutError):
                control.hash_files(paths, time.monotonic() - 1, 4)
