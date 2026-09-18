"""The disk-to-GPU path must never accept incomplete or corrupt tensor bytes."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import stream_weights


class StreamTests(unittest.TestCase):
    def test_direct_io_buffer_view_is_page_aligned_and_owned(self):
        """REGRESSION (direct-02): pinned allocation does not imply O_DIRECT alignment."""
        buffers = [stream_weights.aligned_buffer(17, False) for _ in range(4)]
        for index, buffer in enumerate(buffers):
            self.assertEqual(buffer.data_ptr() % 4096, 0)
            self.assertEqual(buffer.numel(), 17)
            buffer.fill_(index)
        self.assertEqual([buffer.tolist() for buffer in buffers], [[i] * 17 for i in range(4)])

    def test_verified_read_handles_short_reads_and_rejects_corruption(self):
        """Critical: chunk validation must cover every byte before it is copied to GPU."""
        class ShortReads(io.BytesIO):
            def readinto(self, target):
                return super().readinto(target[:3])
        data = b"one complete chunk"
        target = memoryview(bytearray(len(data)))
        digest = hashlib.sha256(data).hexdigest()
        stream_weights.verified_read(ShortReads(data), target, digest)
        self.assertEqual(target, data)
        for content in (data[:-1], b"x" + data[1:]):
            with self.assertRaises(ValueError):
                stream_weights.verified_read(ShortReads(content), target, digest)
        with tempfile.TemporaryFile() as stream:
            stream.write(b"prefix" + data)
            stream.flush()
            stream.seek(2)
            stream_weights.verified_read(stream, target, digest, offset=6)
            self.assertEqual(target, data)
            self.assertEqual(stream.tell(), 2, "Concurrent readers must not share a mutable cursor")

    def test_tensor_metadata_requires_dense_complete_bf16_layout(self):
        """Non-obvious correctness: sorted JSON names are not the physical tensor order."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "weights.bin").write_bytes(b"abcd")
            record = {"schema": 1, "bytes": 4, "chunk_bytes": stream_weights.CHUNK, "chunks": ["digest"],
                      "tensors": {"a": {"dtype": "BF16", "shape": [1], "offset": 2, "bytes": 2},
                                  "z": {"dtype": "BF16", "shape": [1], "offset": 0, "bytes": 2}}}
            (root / "manifest.json").write_text(json.dumps(record))
            stream_weights.metadata(root)
            record["tensors"]["a"]["offset"] = 0
            (root / "manifest.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                stream_weights.metadata(root)
