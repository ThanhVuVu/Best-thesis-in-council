from __future__ import annotations

import unittest

import torch

from src.training.dev_validation import DEVDiscriminator, dev_control_variate_risk


class DeepEmbeddedValidationTests(unittest.TestCase):
    def test_control_variate_matches_hand_calculation(self) -> None:
        losses = torch.tensor([1.0, 2.0])
        weights = torch.tensor([1.0, 3.0])

        result = dev_control_variate_risk(losses, weights)

        self.assertAlmostEqual(result["dev_weighted_loss_mean"], 3.5, places=7)
        self.assertAlmostEqual(result["dev_eta"], -2.5, places=7)
        self.assertAlmostEqual(result["dev_risk"], 1.0, places=7)

    def test_constant_weights_fall_back_to_zero_eta(self) -> None:
        losses = torch.tensor([0.25, 0.75])
        weights = torch.ones(2)

        result = dev_control_variate_risk(losses, weights)

        self.assertEqual(result["dev_eta"], 0.0)
        self.assertAlmostEqual(result["dev_risk"], 0.5, places=7)

    def test_two_layer_discriminator_shape(self) -> None:
        discriminator = DEVDiscriminator(feature_dim=8, hidden_dim=4)
        logits = discriminator(torch.randn(5, 8))

        self.assertEqual(tuple(logits.shape), (5,))
        linear_layers = [layer for layer in discriminator.modules() if isinstance(layer, torch.nn.Linear)]
        self.assertEqual(len(linear_layers), 2)

    def test_mismatched_arrays_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dev_control_variate_risk(torch.ones(2), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
