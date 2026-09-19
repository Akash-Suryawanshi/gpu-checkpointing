"""Physical read accounting must not be confused with logical hashing bytes."""

import unittest

import io_stats


class IoTests(unittest.TestCase):
    def test_diskstats_sectors_are_512_bytes_and_deltas_stay_on_one_device(self):
        """Non-obvious correctness: diskstats reports 512-byte sectors even on 4K devices,
        and a device-set change or counter wrap must invalidate the delta, not zero it."""
        text = " 259       1 nvme2n1 10 0 2048 5 0 0 0 0 0 0 0 0 0 0 0 0 0\n 252       0 dm-0 1 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        self.assertEqual(io_stats.read_bytes("259:1", text), 2048 * 512)
        self.assertEqual(io_stats.read_bytes("252:0", text), 4096)
        with self.assertRaises(ValueError):
            io_stats.read_bytes("8:0", text)
        before = [{"source": "/dev/a", "maj_min": "259:1", "read_bytes": 100, "monotonic_ns": 5}]
        after = [{"source": "/dev/a", "maj_min": "259:1", "read_bytes": 1124, "monotonic_ns": 1_000_000_005}]
        self.assertEqual(io_stats.delta(before, after)[0]["read_bytes"], 1024)
        self.assertAlmostEqual(io_stats.delta(before, after)[0]["seconds"], 1.0)
        for bad in ([{**after[0], "maj_min": "252:0"}], [{**after[0], "read_bytes": 99}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                io_stats.delta(before, bad)
