"""Eviction must either reach zero resident pages or name the files that kept them."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cold_cache


class EvictionTests(unittest.TestCase):
    def test_a_transient_reader_is_retried_and_a_persistent_one_is_named(self):
        """REGRESSION (cold-nvme-loader-pipelined b3, 2026-09-18): a block failed with
        'Selected files are still cached' naming no file, so nothing could be diagnosed.
        A file repopulated between the advice and the check must be retried, and a file
        that stays resident must appear in the message with its page counts."""
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "weights.bin"
            target.write_bytes(b"x" * 8192)
            calls = []

            def flaky(path):
                calls.append(path)
                resident = 2 if len(calls) < 3 else 0
                return {"path": str(path), "bytes": 8192, "resident_before": 2, "resident_after": resident}

            with patch.object(cold_cache, "drop", side_effect=flaky):
                records = cold_cache.evict([target], attempts=5, delay=0)
            self.assertEqual(len(calls), 3, "retried until the file was actually cold")
            self.assertEqual(records[0]["resident_after"], 0)

            def stuck(path):
                return {"path": str(path), "bytes": 8192, "resident_before": 2, "resident_after": 1}

            with patch.object(cold_cache, "drop", side_effect=stuck):
                with self.assertRaises(RuntimeError) as caught:
                    cold_cache.evict([target], attempts=2, delay=0)
            message = str(caught.exception)
            self.assertIn("weights.bin", message)
            self.assertIn("kept 1 of 2 pages", message)
