from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.daeac_paper import ClassifierH, DAEACNetwork, DualClassifierH
from src.training.train_daeac_paper import (
    _source_classification_loss,
    build_daeac_model,
    load_daeac_checkpoint,
    save_daeac_checkpoint,
)
from src.utils.io import load_config


class RTDDAEACPhase2ADualHeadTests(unittest.TestCase):
    def test_single_head_behavior_is_unchanged(self) -> None:
        model = DAEACNetwork()
        x = torch.randn(2, 1, 3, 128)

        features, logits, probs = model(x, return_logits=True)
        output = model(x, return_dict=True)

        self.assertIsInstance(model.classifier, ClassifierH)
        self.assertNotIn("logits_1", output)
        self.assertEqual(tuple(features.shape), (2, 256))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(probs.shape), (2, 4))

    def test_dual_head_output_shapes_and_independent_parameters(self) -> None:
        model = DAEACNetwork(dual_head=True)
        x = torch.randn(3, 1, 3, 128)

        output = model(x, return_dict=True)

        self.assertIsInstance(model.classifier, DualClassifierH)
        self.assertEqual(tuple(output["features"].shape), (3, 256))
        self.assertEqual(tuple(output["logits"].shape), (3, 4))
        self.assertEqual(tuple(output["logits_1"].shape), (3, 4))
        self.assertEqual(tuple(output["logits_2"].shape), (3, 4))
        self.assertIsNot(model.classifier.fc.weight, model.classifier.fc2.weight)
        self.assertIsNot(model.classifier.fc.bias, model.classifier.fc2.bias)

    def test_dual_head_averaged_logits_are_public_logits(self) -> None:
        model = DAEACNetwork(dual_head=True)
        x = torch.randn(2, 1, 3, 128)

        output = model(x, return_dict=True)
        _, logits, probs = model(x, return_logits=True)
        expected = 0.5 * (output["logits_1"] + output["logits_2"])

        self.assertTrue(torch.allclose(output["logits"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))
        self.assertTrue(torch.allclose(probs, torch.softmax(expected, dim=1), atol=1e-6))

    def test_dual_head_source_loss_is_finite(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "phase6_daeac_paper_dualhead.yaml")
        model = DAEACNetwork(dual_head=True)
        output = model(torch.randn(4, 1, 3, 128), return_dict=True)
        labels = torch.tensor([0, 1, 2, 3])

        loss = _source_classification_loss(output, labels, None, None, config)

        self.assertTrue(torch.isfinite(loss))

    def test_old_checkpoint_initializes_dual_head_from_c1(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base_config = load_config(root / "configs" / "phase6_daeac_paper.yaml")
        dual_config = load_config(root / "configs" / "phase6_daeac_paper_dualhead.yaml")
        device = torch.device("cpu")
        single = build_daeac_model(base_config, device)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.pt"
            save_daeac_checkpoint(single, base_config, path, 0, {})
            dual = load_daeac_checkpoint(path, dual_config, device)

        self.assertIsInstance(dual.classifier, DualClassifierH)
        self.assertTrue(torch.allclose(dual.classifier.fc.weight, dual.classifier.fc2.weight))
        self.assertTrue(torch.allclose(dual.classifier.fc.bias, dual.classifier.fc2.bias))

    def test_aux_snapshot_preserves_both_heads(self) -> None:
        model = DAEACNetwork(dual_head=True)
        snapshot = copy.deepcopy(model.classifier).eval()

        self.assertIsInstance(snapshot, DualClassifierH)
        self.assertIn("fc2.weight", snapshot.state_dict())
        self.assertTrue(torch.allclose(snapshot.fc.weight, model.classifier.fc.weight))
        self.assertTrue(torch.allclose(snapshot.fc2.weight, model.classifier.fc2.weight))

    def test_dual_head_config_loads_and_keeps_real_rr(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "phase6_daeac_paper_dualhead.yaml")
        model = build_daeac_model(config, torch.device("cpu"))

        self.assertEqual(config["data"]["rr_mode"], "real")
        self.assertTrue(config["rtd_daeac"]["dual_head"]["enabled"])
        self.assertEqual(float(config["rtd_daeac"]["dual_head"]["consistency_weight"]), 0.0)
        self.assertIsInstance(model.classifier, DualClassifierH)


if __name__ == "__main__":
    unittest.main()
