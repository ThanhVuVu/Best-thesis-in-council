from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.daeac_dataset import DAEACDataset, DAEACPseudoLabeledDataset, DAEACTargetUnlabeledDataset
from src.data.daeac_preprocess import compute_rr_features_from_diffs
from src.models.daeac_paper import DAEACNetwork, LateFusionClassifierH
from src.training.train_daeac_paper import build_daeac_model
from src.utils.io import load_config


class DAEACRRLateFusionTests(unittest.TestCase):
    def test_rr_features_follow_fcba_definition(self) -> None:
        rr_diffs = np.arange(1, 13, dtype=np.float64) * 360.0

        features = compute_rr_features_from_diffs(rr_diffs, beat_index=5, target_fs=360.0)

        self.assertIsNotNone(features)
        assert features is not None
        rr_seconds = rr_diffs / 360.0
        rr_avg = rr_seconds.mean()
        rr_anterior = rr_seconds[4]
        rr_posterior = rr_seconds[5]
        rr_local = np.concatenate([rr_seconds[0:5], rr_seconds[5:10]]).mean()
        expected = np.asarray(
            [
                rr_anterior - rr_avg,
                rr_posterior - rr_avg,
                rr_local - rr_avg,
                rr_anterior / rr_avg,
                rr_posterior / rr_avg,
                rr_local / rr_avg,
                rr_anterior / rr_posterior,
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(features, expected))
        self.assertIsNone(compute_rr_features_from_diffs(rr_diffs, beat_index=4, target_fs=360.0))

    def test_dataset_returns_morphology_rr_and_label_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            _write_sample_npz(path)

            ds = DAEACDataset(path, return_rr_features=True, morphology_only=True)
            try:
                x, rr_features, y = ds[0]

                self.assertEqual(tuple(x.shape), (1, 1, 128))
                self.assertEqual(tuple(rr_features.shape), (7,))
                self.assertEqual(int(y), 0)
                self.assertEqual(tuple(ds.rr_features.shape), (3, 7))
            finally:
                ds.close()

    def test_target_and_pseudo_datasets_preserve_rr_features_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            _write_sample_npz(path)

            target = DAEACTargetUnlabeledDataset(path, return_rr_features=True, morphology_only=True)
            try:
                x, rr_features, index = target[1]
                self.assertEqual(tuple(x.shape), (1, 1, 128))
                self.assertEqual(tuple(rr_features.shape), (7,))
                self.assertEqual(int(index), 1)

                pseudo = DAEACPseudoLabeledDataset(target, torch.tensor([1]), torch.tensor([2]))
                px, prr, py, conf, entropy = pseudo[0]
                self.assertEqual(tuple(px.shape), (1, 1, 128))
                self.assertEqual(tuple(prr.shape), (7,))
                self.assertEqual(int(py), 2)
                self.assertEqual(float(conf), 1.0)
                self.assertEqual(float(entropy), 0.0)
            finally:
                target.close()

    def test_late_fusion_network_forward_shapes(self) -> None:
        model = DAEACNetwork(
            attention_type="fcba",
            input_rows=1,
            late_fusion=True,
            rr_dim=7,
            late_fusion_fc1_dim=128,
            late_fusion_fc2_dim=64,
        )
        x = torch.randn(2, 1, 1, 128)
        rr_features = torch.randn(2, 7)

        output = model(x, rr_features=rr_features, return_dict=True)
        features, logits, probs = model(x, rr_features=rr_features, return_logits=True)

        self.assertIsInstance(model.classifier, LateFusionClassifierH)
        self.assertEqual(model.feature_dim, 128)
        self.assertEqual(tuple(output["features"].shape), (2, 128))
        self.assertEqual(tuple(features.shape), (2, 128))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(probs.shape), (2, 4))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6))

    def test_late_fusion_config_builds_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "phase6_daeac_fcba_latefusion_rr.yaml")

        model = build_daeac_model(config, torch.device("cpu"))

        self.assertIsInstance(model.classifier, LateFusionClassifierH)
        self.assertEqual(model.feature_dim, 128)
        self.assertTrue(config["data"]["return_rr_features"])
        self.assertTrue(config["data"]["morphology_only"])


def _write_sample_npz(path: Path) -> None:
    np.savez_compressed(
        path,
        x=np.random.randn(3, 1, 3, 128).astype(np.float32),
        rr_features=np.random.randn(3, 7).astype(np.float32),
        y=np.asarray([0, 1, 2], dtype=np.int64),
        record=np.asarray(["r1", "r1", "r1"], dtype=object),
        class_names=np.asarray(["N", "S", "V", "F"], dtype=object),
    )


if __name__ == "__main__":
    unittest.main()
