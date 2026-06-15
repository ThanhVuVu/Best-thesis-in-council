from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.daeac_losses import CustomFocalLoss, build_daeac_classification_loss


class CustomFocalLossTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

