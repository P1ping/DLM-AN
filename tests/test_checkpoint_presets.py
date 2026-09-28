"""Tests for checkpoint preset selection without network access."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from checkpoint_presets import PRESETS, resolve_checkpoints


class CheckpointPresetsTests(unittest.TestCase):
    def test_each_backend_downloads_its_matching_bundle(self):
        for backend in PRESETS:
            with self.subTest(backend=backend), patch(
                "checkpoint_presets.hf_hub_download",
                side_effect=lambda *, repo_id, filename: f"/cache/{filename}",
            ) as download:
                bundle = resolve_checkpoints(backend, checkpoint_repo="owner/model")
                expected = [
                    name
                    for name in (
                        "lm_checkpoint",
                        "quantizer_checkpoint",
                        "flow_checkpoint",
                        "hift_checkpoint",
                    )
                    if PRESETS[backend][name] is not None
                ]
                self.assertEqual(download.call_count, len(expected))
                self.assertEqual(bundle.lm_config, PRESETS[backend]["lm_config"])
                self.assertEqual(
                    bundle.lm_checkpoint,
                    Path(f"/cache/{PRESETS[backend]['lm_checkpoint']}"),
                )
                self.assertEqual(
                    bundle.quantizer_checkpoint,
                    Path(f"/cache/{PRESETS[backend]['quantizer_checkpoint']}"),
                )
                self.assertEqual(bundle.hift_checkpoint, Path("/cache/shared/hift.pt"))
                self.assertEqual(bundle.flow_checkpoint is None, backend == "joint-ctx")

    def test_local_overrides_skip_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.touch()
            with patch("checkpoint_presets.hf_hub_download") as download:
                bundle = resolve_checkpoints(
                    "km-flow",
                    lm_checkpoint=path,
                    quantizer_checkpoint=path,
                    flow_checkpoint=path,
                    hift_checkpoint=path,
                )
            download.assert_not_called()
            self.assertEqual(bundle.lm_checkpoint, path)

    def test_missing_local_path_does_not_fall_back_to_download(self):
        with patch("checkpoint_presets.hf_hub_download") as download:
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoints("km-flow", lm_checkpoint="/nonexistent/dlman.pt")
            download.assert_not_called()

    def test_joint_rejects_extra_flow(self):
        with self.assertRaisesRegex(ValueError, "joint-ctx"):
            resolve_checkpoints("joint-ctx", flow_checkpoint="unused.pt")

    def test_download_failure_explains_local_override(self):
        with patch(
            "checkpoint_presets.hf_hub_download",
            side_effect=FileNotFoundError("repo missing"),
        ):
            with self.assertRaisesRegex(RuntimeError, "supply --lm_checkpoint"):
                resolve_checkpoints("km-flow")


if __name__ == "__main__":
    unittest.main()
