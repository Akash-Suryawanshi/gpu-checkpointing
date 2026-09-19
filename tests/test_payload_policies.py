"""What each snapshot payload policy still detects, and what it stops detecting."""

import os
from pathlib import Path
import tempfile
import unittest

import control


def payload(root, data=b"\xa5" * (3 * 1024 * 1024)):
    """The three names `payload_paths` requires, plus one file worth hashing."""
    (root / "images").mkdir(parents=True)
    (root / "before.json").write_text('{"weights": "untouched"}')
    for name in ("inventory.img", "pstree.img"):
        (root / "images" / name).write_bytes(b"x")
    (root / "images/pages-1.img").write_bytes(data)
    return root / "images/pages-1.img"


class ChunkedDigest(unittest.TestCase):
    def test_value_is_independent_of_worker_count(self):
        """Concurrency must never change identity, or a campaign cannot be compared."""
        with tempfile.TemporaryDirectory() as folder:
            target = payload(Path(folder))
            digests = {control.chunked_hash(target, None, n) for n in (1, 2, 8)}
            self.assertEqual(len(digests), 1)

    def test_a_single_flipped_byte_changes_the_digest(self):
        """The parallel policy is a different value, not a weaker guarantee."""
        with tempfile.TemporaryDirectory() as folder:
            target = payload(Path(folder))
            before = control.chunked_hash(target, None, 4)
            size = target.stat().st_size
            for offset in (0, size // 2, size - 1):
                with target.open("r+b") as stream:
                    stream.seek(offset)
                    original = stream.read(1)
                    stream.seek(offset)
                    stream.write(bytes([original[0] ^ 1]))
                self.assertNotEqual(control.chunked_hash(target, None, 4), before)
                with target.open("r+b") as stream:
                    stream.seek(offset)
                    stream.write(original)
            self.assertEqual(control.chunked_hash(target, None, 4), before)

    def test_chunk_boundaries_are_covered(self):
        """A file longer than one chunk must fold every chunk, in order."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = payload(root, os.urandom(1024))
            long = root / "images/pages-2.img"
            long.write_bytes(os.urandom(control.PAYLOAD_CHUNK + 4096))
            self.assertNotEqual(control.chunked_hash(long, None, 4),
                                control.chunked_hash(target, None, 4))
            data = bytearray(long.read_bytes())
            # Swapping two chunks keeps every byte and must still be rejected.
            head, tail = data[:control.PAYLOAD_CHUNK], data[control.PAYLOAD_CHUNK:]
            before = control.chunked_hash(long, None, 4)
            long.write_bytes(bytes(tail + head))
            self.assertNotEqual(control.chunked_hash(long, None, 4), before)


class PublicationOnly(unittest.TestCase):
    def test_structure_check_rejects_truncation_without_reading_contents(self):
        """No content read, but size and file set are still enforced."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = payload(root)
            sizes = control.payload_sizes(root)
            self.assertEqual(sizes["images/pages-1.img"], target.stat().st_size)
            with target.open("r+b") as stream:
                stream.truncate(1024)
            self.assertNotEqual(control.payload_sizes(root), sizes)

    def test_structure_check_misses_a_same_size_rewrite(self):
        """The trade this policy makes, stated as a test rather than a comment."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = payload(root)
            sizes = control.payload_sizes(root)
            target.write_bytes(b"\x5a" * target.stat().st_size)
            self.assertEqual(control.payload_sizes(root), sizes)
            self.assertNotEqual(control.inventory(root), sizes)

    def test_unknown_policy_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            target = payload(Path(folder))
            with self.assertRaisesRegex(ValueError, "payload policy"):
                control.payload_hash(target, "no-such-policy-v1")


if __name__ == "__main__":
    unittest.main()
