"""Which files a restored image needs decides whether activation-time hashing is required."""

import json
from pathlib import Path
import tempfile
import unittest

import image_files


class ImageFileTests(unittest.TestCase):
    def test_model_weights_are_separated_from_libraries_and_run_logs(self):
        """Non-obvious correctness: the venv ships a safetensors *extension module* whose name
        contains 'safetensors' but holds no weights, and the model directory holds the index
        JSON as well as shards. Misfiling either one would invert the audit's conclusion."""
        paths = ["/data/model/model-00001-of-00004.safetensors", "/data/model/model.safetensors.index.json",
                 "/venv/site-packages/safetensors/_safetensors_rust.abi3.so", "/venv/torch/lib/libtorch.so",
                 "/venv/nvidia/cublas/lib/libcublas.so.12", "/run/trainer.stderr", "/usr/bin/python3.10"]
        groups = image_files.classify(paths, "/data/model", "/run")
        self.assertEqual(groups["model_files"],
                         ["/data/model/model-00001-of-00004.safetensors", "/data/model/model.safetensors.index.json"])
        self.assertEqual(len(groups["libraries"]), 3)
        self.assertEqual(groups["run_files"], ["/run/trainer.stderr"])
        self.assertEqual(groups["other"], ["/usr/bin/python3.10"])

    def test_collected_paths_are_absolute_and_deduplicated(self):
        """Critical: CRIU logs collected paths without a leading slash and can repeat one;
        a relative or duplicated path would silently escape the model-directory comparison."""
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "restore.log"
            log.write_text("(00.1) Collected [data/model/a.safetensors] ID 0x1\n"
                           "(00.2) Collected [data/model/a.safetensors] ID 0x2\n"
                           "(00.3) Collected [/usr/lib/libc.so.6] ID 0x3\n"
                           "(00.4) Opening something unrelated\n")
            self.assertEqual(image_files.collected(log), ["/data/model/a.safetensors", "/usr/lib/libc.so.6"])
