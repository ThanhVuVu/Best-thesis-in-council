from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.daeac_paper import DAEACNetwork, DualLateFusionClassifierH
from src.training.train_daeac_paper import (
    ReliablePseudoLabelSelector,
    build_daeac_model,
    load_daeac_checkpoint,
    save_daeac_checkpoint,
)
from scripts.phase6_daeac_paper.common import load_phase1_config


class DAEACLateFusionDualHeadTests(unittest.TestCase):
    def test_late_fusion_dual_head_forward_shapes_and_average_logits(self) -> None:
        model = _dual_late_model()
        x = torch.randn(4, 1, 1, 128)
        rr_features = torch.randn(4, 7)

        output = model(x, rr_features=rr_features, return_dict=True)
        features, logits, probs = model(x, rr_features=rr_features, return_logits=True)
        expected = 0.5 * (output["logits_1"] + output["logits_2"])

        self.assertIsInstance(model.classifier, DualLateFusionClassifierH)
        self.assertEqual(model.feature_dim, 128)
        self.assertEqual(tuple(output["features"].shape), (4, 128))
        self.assertEqual(tuple(features.shape), (4, 128))
        self.assertEqual(tuple(output["logits"].shape), (4, 3))
        self.assertEqual(tuple(output["logits_1"].shape), (4, 3))
        self.assertEqual(tuple(output["logits_2"].shape), (4, 3))
        self.assertTrue(torch.allclose(output["logits"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))
        self.assertTrue(torch.allclose(probs, torch.softmax(expected, dim=1), atol=1e-6))
        self.assertIsNot(model.classifier.fc2.weight, model.classifier.fc2_b.weight)
        self.assertIsNot(model.classifier.fc3.weight, model.classifier.fc3_b.weight)

    def test_single_head_late_fusion_checkpoint_initializes_dual_head(self) -> None:
        root = Path(__file__).resolve().parents[1]
        single_config = load_phase1_config(str(root / "configs" / "phase6_daeac_fcba_latefusion_rr_nsv.yaml"))
        dual_config = load_phase1_config(str(root / "configs" / "phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_ds2.yaml"))
        device = torch.device("cpu")
        single = build_daeac_model(single_config, device)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single_late_fusion.pt"
            save_daeac_checkpoint(single, single_config, path, 0, {})
            dual = load_daeac_checkpoint(path, dual_config, device)

        self.assertIsInstance(dual.classifier, DualLateFusionClassifierH)
        self.assertTrue(torch.allclose(dual.classifier.fc2.weight, dual.classifier.fc2_b.weight))
        self.assertTrue(torch.allclose(dual.classifier.fc2.bias, dual.classifier.fc2_b.bias))
        self.assertTrue(torch.allclose(dual.classifier.fc3.weight, dual.classifier.fc3_b.weight))
        self.assertTrue(torch.allclose(dual.classifier.fc3.bias, dual.classifier.fc3_b.bias))

    def test_dhrel_configs_build_late_fusion_dual_head_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_names = [
            "phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_ds2.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_incart.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_svdb.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dhrel_mitbih_incart.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dhrel_mitbih_svdb.yaml",
        ]
        for name in config_names:
            with self.subTest(name=name):
                config = load_phase1_config(str(root / "configs" / name))
                model = build_daeac_model(config, torch.device("cpu"))

                self.assertEqual(config["data"]["num_classes"], 3)
                self.assertEqual(config["data"]["rr_mode"], "real")
                self.assertTrue(config["data"]["morphology_only"])
                self.assertTrue(config["data"]["return_rr_features"])
                self.assertTrue(config["model"]["late_fusion"]["enabled"])
                self.assertTrue(config["rtd_daeac"]["dual_head"]["enabled"])
                self.assertTrue(config["rtd_daeac"]["reliable_pseudo"]["enabled"])
                self.assertEqual(float(config["rtd_daeac"]["reliable_pseudo"]["confidence_thresholds"]["S"]), 0.97)
                self.assertEqual(float(config["adaptation"]["pseudo_thresholds"]["S"]), 0.97)
                self.assertIsInstance(model.classifier, DualLateFusionClassifierH)

    def test_reliable_selector_accepts_late_fusion_dual_head_with_rr_features(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_phase1_config(str(root / "configs" / "phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_ds2.yaml"))
        model = build_daeac_model(config, torch.device("cpu"))
        source = TensorDataset(
            torch.randn(6, 1, 1, 128),
            torch.randn(6, 7),
            torch.tensor([0, 0, 1, 1, 2, 2]),
        )

        selector = ReliablePseudoLabelSelector.from_source(
            model,
            DataLoader(source, batch_size=3, shuffle=False),
            config,
            torch.device("cpu"),
        )
        aux_classifier = copy.deepcopy(model.classifier).eval()
        selected = selector.select_batch(model, aux_classifier, torch.randn(2, 1, 1, 128), rr_features=torch.randn(2, 7))

        self.assertEqual(tuple(selected["labels"].shape), (2,))
        self.assertEqual(tuple(selected["confidence"].shape), (2,))
        self.assertEqual(tuple(selected["head_discrepancy"].shape), (2,))
        self.assertIn("distance", selected["masks"])


def _dual_late_model() -> DAEACNetwork:
    return DAEACNetwork(
        num_classes=3,
        attention_type="fcba",
        input_rows=1,
        late_fusion=True,
        dual_head=True,
        rr_dim=7,
        late_fusion_fc1_dim=128,
        late_fusion_fc2_dim=64,
    )


if __name__ == "__main__":
    unittest.main()
