from __future__ import annotations

import unittest

import torch

from src.training.mcc_loss import minimum_class_confusion_loss


class MinimumClassConfusionLossTests(unittest.TestCase):
    def test_uniform_predictions_have_expected_off_diagonal_mass(self) -> None:
        logits = torch.zeros(8, 4)
        loss, diagnostics = minimum_class_confusion_loss(logits, temperature=1.0, return_diagnostics=True)

        self.assertAlmostEqual(float(loss), 0.75, places=5)
        self.assertEqual(diagnostics["pred_counts"], [8, 0, 0, 0])
        self.assertTrue(torch.isfinite(diagnostics["soft_confusion"]).all())

    def test_separated_predictions_have_near_zero_confusion(self) -> None:
        logits = torch.full((8, 4), -20.0)
        logits[torch.arange(8), torch.arange(8) % 4] = 20.0
        loss = minimum_class_confusion_loss(logits, temperature=1.0)

        self.assertLess(float(loss), 1.0e-6)

    def test_loss_is_differentiable_without_target_labels(self) -> None:
        logits = torch.randn(16, 4, requires_grad=True)
        loss = minimum_class_confusion_loss(logits, temperature=1.0)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_target_batch_is_finite_and_differentiable(self) -> None:
        logits = torch.empty(0, 4, requires_grad=True)
        loss = minimum_class_confusion_loss(logits, temperature=1.0)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(tuple(logits.grad.shape), (0, 4))


if __name__ == "__main__":
    unittest.main()
