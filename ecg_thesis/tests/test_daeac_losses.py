from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.daeac_losses import (
    ClassBalancedFocalLoss,
    CustomFocalLoss,
    WeightedCrossEntropyByBatchSize,
    build_daeac_classification_loss,
    weighted_cross_entropy_from_logits,
)


class CustomFocalLossTest(unittest.TestCase):
    def test_weighted_ce_divides_by_batch_size_as_algorithm_one(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
        per_sample = F.cross_entropy(logits, labels, weight=weights, reduction="none")
        expected = per_sample.sum() / len(labels)

        functional = weighted_cross_entropy_from_logits(logits, labels, weights)
        module = WeightedCrossEntropyByBatchSize(weights)

        self.assertTrue(torch.allclose(functional, expected))
        self.assertTrue(torch.allclose(module(logits, labels), expected))
        self.assertFalse(torch.allclose(expected, F.cross_entropy(logits, labels, weight=weights)))

    def test_gamma_zero_without_alpha_matches_cross_entropy(self) -> None:
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.3]], dtype=torch.float32)
        labels = torch.tensor([0, 2], dtype=torch.long)

        focal = CustomFocalLoss(alpha=None, gamma=0.0)

        self.assertTrue(torch.allclose(focal(logits, labels), F.cross_entropy(logits, labels)))

    def test_alpha_weights_true_class_loss(self) -> None:
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.3]], dtype=torch.float32)
        labels = torch.tensor([0, 2], dtype=torch.long)
        alpha = torch.tensor([1.6, 1.8, 0.8], dtype=torch.float32)
        gamma = 2.35

        focal = CustomFocalLoss(alpha=alpha, gamma=gamma, reduction="none")
        ce = F.cross_entropy(logits, labels, reduction="none")
        expected = alpha[labels] * ((1.0 - torch.exp(-ce)) ** gamma) * ce

        self.assertTrue(torch.allclose(focal(logits, labels), expected))

    def test_alpha_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "focal_alpha length"):
            build_daeac_classification_loss(
                {"source_loss": "focal", "focal_alpha": [1.0, 1.0], "focal_gamma": 2.0},
                num_classes=3,
                class_weights=None,
            )

    def test_standard_focal_has_no_alpha_when_class_weights_disabled(self) -> None:
        loss = build_daeac_classification_loss(
            {"source_loss": "focal", "focal_alpha": None, "focal_gamma": 2.0},
            num_classes=3,
            class_weights=None,
        )

        self.assertIsInstance(loss, CustomFocalLoss)
        self.assertIsNone(loss.alpha)

    def test_class_balanced_focal_alpha_uses_effective_number(self) -> None:
        counts = torch.tensor([100.0, 10.0, 1.0, 50.0], dtype=torch.float32)
        beta = 0.9

        loss = ClassBalancedFocalLoss(counts, beta=beta, gamma=2.35)
        raw = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
        expected = raw * (float(counts.numel()) / raw.sum())

        self.assertTrue(torch.allclose(loss.alpha, expected))

    def test_class_balanced_focal_is_finite_for_extreme_logits(self) -> None:
        logits = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0]], dtype=torch.float32)
        labels = torch.tensor([0, 0], dtype=torch.long)

        loss = ClassBalancedFocalLoss([2.0, 2.0], beta=0.9999, gamma=2.35)(logits, labels)

        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss))

    def test_class_balanced_focal_beta_zero_gamma_zero_matches_cross_entropy(self) -> None:
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.3]], dtype=torch.float32)
        labels = torch.tensor([0, 2], dtype=torch.long)

        loss = ClassBalancedFocalLoss([100.0, 10.0, 1.0], beta=0.0, gamma=0.0)

        self.assertTrue(torch.allclose(loss(logits, labels), F.cross_entropy(logits, labels)))

    def test_class_balanced_focal_downweights_confident_correct_predictions(self) -> None:
        logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)

        focal = ClassBalancedFocalLoss([1.0, 1.0], beta=0.0, gamma=2.0)
        ce = F.cross_entropy(logits, labels)

        self.assertLess(float(focal(logits, labels)), float(ce))

    def test_build_selects_class_balanced_focal_from_losses_config(self) -> None:
        loss = build_daeac_classification_loss(
            {"losses": {"source_cls_loss": "class_balanced_focal", "class_balanced_focal": {"beta": 0.9, "gamma": 2.35}}},
            num_classes=4,
            class_weights=None,
            class_counts=torch.tensor([100.0, 10.0, 1.0, 50.0]),
        )

        self.assertIsInstance(loss, ClassBalancedFocalLoss)
        self.assertAlmostEqual(loss.gamma, 2.35)

    def test_class_balanced_focal_requires_source_class_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "class_counts"):
            build_daeac_classification_loss(
                {"losses": {"source_cls_loss": "class_balanced_focal"}},
                num_classes=4,
                class_weights=None,
            )


if __name__ == "__main__":
    unittest.main()
