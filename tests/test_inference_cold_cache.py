"""Cold-start evidence must describe actual disk residency, not a hint alone."""

import mmap
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cold_cache


class ColdCacheTests(unittest.TestCase):
    def test_eviction_verifies_real_pages_without_changing_the_file(self):
        """Acceptance criterion: a cold disk trial evicts only selected files and verifies residency."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data"
            data = b"snapshot" * (mmap.PAGESIZE * 2)
            path.write_bytes(data)
            evidence = cold_cache.evict([path, path])
            self.assertEqual(len(evidence), 1)
            self.assertGreater(evidence[0]["resident_before"], 0)
            self.assertEqual(evidence[0]["resident_after"], 0)
            self.assertEqual(path.read_bytes(), data)
            # fadvise is advisory: a pinned or concurrently read page must fail
            # admission rather than quietly become a supposedly cold result.
            with patch.object(cold_cache, "resident_pages", return_value=1):
                with self.assertRaisesRegex(RuntimeError, "still cached"):
                    cold_cache.evict([path])
