"""Asset and child-environment contracts that do not need CUDA."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import control
import prepare_assets
import session


class AssetTests(unittest.TestCase):
    def test_partial_and_changed_assets_are_not_reused(self):
        """AC3: only complete, unchanged preparations may be reused."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                prepare_assets.validate_assets(root)
            (root / "weights").write_bytes(b"original")
            control.write(root / "manifest.json", {"model_path": str(root),
                "identity": {"files": {"weights": control.file_hash(root / "weights")}}, "artifacts": {}})
            prepare_assets.validate_assets(root)
            (root / "weights").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Model file changed"):
                prepare_assets.validate_assets(root)

    def test_only_allowed_cache_variables_reach_children(self):
        """Critical: data-volume cache paths propagate without inheriting secrets."""
        allowed = {name: "/data/" + name for name in
                   ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH")}
        with patch.dict(os.environ, {**allowed, "UNRELATED_SECRET": "private"}):
            env = session.child_environment(Path("/tools/cuda-checkpoint"))
        self.assertTrue(all(env[k] == v for k, v in allowed.items()))
        self.assertNotIn("UNRELATED_SECRET", env)

    def test_direct_preparation_sets_deterministic_cublas_before_loading(self):
        """REGRESSION (8B preparation v1): direct loading needs the child cuBLAS policy."""
        import worker
        from unittest.mock import MagicMock
        torch, transformers = MagicMock(), MagicMock()
        def check(*args, **kwargs):
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            return MagicMock()
        transformers.AutoModelForCausalLM.from_pretrained.side_effect = check
        with patch.dict(os.environ, {}, clear=True), patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            worker.load("model")
