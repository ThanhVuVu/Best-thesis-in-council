from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.models.daeac_paper import (
    ClassifierH,
    DAEACNetwork,
    FrequencyConvolutionBlockAttention2D,
    SELayer2D,
)
from src.training.train_daeac_paper import build_daeac_model
from src.utils.io import load_config


class DAEACFCBATests(unittest.TestCase):
    def test_fcba_preserves_shape_and_outputs_finite_values(self) -> None:
        module = FrequencyConvolutionBlockAttention2D(channels=16, reduction=4, frequency_modes=4)
        x = torch.randn(2, 16, 1, 128)

        y = module(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_fcba_network_forward_shapes(self) -> None:
        model = DAEACNetwork(attention_type="fcba")
        x = torch.randn(2, 1, 3, 128)

        features, logits, probs = model(x, return_logits=True)

        self.assertEqual(tuple(features.shape), (2, 256))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(probs.shape), (2, 4))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertIsInstance(model.feature_extractor.aspp_se_1.se, FrequencyConvolutionBlockAttention2D)
        self.assertIsInstance(model.classifier, ClassifierH)

    def test_default_network_still_uses_se_attention(self) -> None:
        model = DAEACNetwork()

        self.assertIsInstance(model.feature_extractor.aspp_se_1.se, SELayer2D)
        self.assertIsInstance(model.feature_extractor.aspp_se_2.se, SELayer2D)
        self.assertIsInstance(model.feature_extractor.final_aspp_se.se, SELayer2D)

    def test_fcba_config_builds_non_dual_head_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "phase6_daeac_paper_fcba.yaml")

        model = build_daeac_model(config, torch.device("cpu"))

        self.assertEqual(config["model"]["attention"], "fcba")
        self.assertFalse(config["rtd_daeac"]["dual_head"]["enabled"])
        self.assertIsInstance(model.feature_extractor.aspp_se_1.se, FrequencyConvolutionBlockAttention2D)
        self.assertIsInstance(model.classifier, ClassifierH)


if __name__ == "__main__":
    unittest.main()
