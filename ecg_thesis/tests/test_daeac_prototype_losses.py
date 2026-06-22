from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.daeac_prototype_losses import (
    build_margin_matrix,
    directed_target_alignment_loss,
    linear_ramp,
    sample_prototype_margin_loss,
    source_compactness_loss,
    target_compactness_loss,
    target_reliability_weights,
    validate_prototype_loss_config,
)


CLASS_NAMES = ["N", "S", "V", "F"]
PAIR_MARGIN = {
    "N": {"N": 0.0, "S": 7.5, "V": 5.0, "F": 7.5},
    "S": {"N": 7.5, "S": 0.0, "V": 5.0, "F": 5.0},
    "V": {"N": 5.0, "S": 5.0, "V": 0.0, "F": 5.0},
    "F": {"N": 7.5, "S": 5.0, "V": 5.0, "F": 0.0},
}


class PrototypeReplacementLossTests(unittest.TestCase):
    def test_source_and_weighted_target_compactness(self) -> None:
        prototypes = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        valid = torch.tensor([True, True])
        source = torch.tensor([[1.0, 0.0], [2.0, 2.0]], requires_grad=True)
        source_loss, source_diag = source_compactness_loss(source, torch.tensor([0, 1]), prototypes, valid)
        self.assertAlmostEqual(float(source_loss.detach()), 1.5)
        self.assertEqual(float(source_diag["active_samples"]), 2.0)

        target = torch.tensor([[1.0, 0.0], [2.0, 2.0]], requires_grad=True)
        weights = torch.tensor([1.0, 3.0])
        target_loss, _ = target_compactness_loss(target, torch.tensor([0, 1]), prototypes, valid, weights)
        self.assertAlmostEqual(float(target_loss.detach()), 1.75)
        (source_loss + target_loss).backward()
        self.assertGreater(float(source.grad.abs().sum()), 0.0)
        self.assertGreater(float(target.grad.abs().sum()), 0.0)
        self.assertIsNone(prototypes.grad)

    def test_target_weights_are_detached_and_rejected_sample_has_no_gradient(self) -> None:
        confidence = torch.tensor([0.9, 0.8], requires_grad=True)
        entropy = torch.tensor([0.1, 0.5], requires_grad=True)
        weights = target_reliability_weights(confidence, entropy)
        self.assertFalse(weights.requires_grad)
        self.assertTrue(torch.allclose(weights, torch.tensor([0.81, 0.4])))
        all_features = torch.tensor([[1.0, 0.0], [9.0, 9.0]], requires_grad=True)
        accepted = all_features[:1]
        loss, _ = target_compactness_loss(
            accepted,
            torch.tensor([0]),
            torch.zeros(2, 2),
            torch.tensor([True, True]),
            weights[:1],
        )
        loss.backward()
        self.assertGreater(float(all_features.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(all_features.grad[1].abs().sum()), 0.0)

    def test_directed_alignment_reaches_target_not_source_anchor(self) -> None:
        target = torch.tensor([[2.0, 0.0], [4.0, 0.0]], requires_grad=True)
        source = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
        loss, diag = directed_target_alignment_loss(
            target,
            torch.tensor([0, 0]),
            torch.ones(2),
            source,
            torch.tensor([True, True]),
            min_target_count=2,
        )
        loss.backward()
        self.assertEqual(float(diag["active_classes"]), 1.0)
        self.assertGreater(float(target.grad.abs().sum()), 0.0)
        self.assertIsNone(source.grad)

    def test_pair_margin_values_and_sample_level_gradient(self) -> None:
        matrix = build_margin_matrix(
            {"use_pair_margin": True, "pair_margin": PAIR_MARGIN}, CLASS_NAMES, torch.device("cpu"), torch.float32
        )
        self.assertEqual(float(matrix[1, 0]), 7.5)
        self.assertEqual(float(matrix[3, 0]), 7.5)
        self.assertEqual(float(matrix[2, 0]), 5.0)
        features = torch.tensor([[0.5, 0.0]], requires_grad=True)
        prototypes = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        loss, diag = sample_prototype_margin_loss(
            features,
            torch.tensor([0]),
            prototypes,
            torch.tensor([True, True]),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        )
        self.assertAlmostEqual(float(loss.detach()), 1.0)
        self.assertAlmostEqual(float(diag["violation_ratio"]), 1.0)
        loss.backward()
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_empty_and_invalid_prototype_cases_are_finite_graph_zeros(self) -> None:
        empty = torch.empty(0, 2, requires_grad=True)
        loss, diag = sample_prototype_margin_loss(
            empty,
            torch.empty(0, dtype=torch.long),
            torch.zeros(2, 2),
            torch.tensor([False, False]),
            torch.zeros(2, 2),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(diag["active_samples"]), 0.0)
        loss.backward()
        self.assertIsNotNone(empty.grad)

    def test_ramp_and_config_validation(self) -> None:
        self.assertEqual(validate_prototype_loss_config({}, CLASS_NAMES)["mode"], "legacy")
        self.assertEqual(
            validate_prototype_loss_config({"prototype_losses": {"enabled": False, "mode": "legacy"}}, CLASS_NAMES)["mode"],
            "legacy",
        )
        self.assertEqual(linear_ramp(0, 10), 0.0)
        self.assertEqual(linear_ramp(5, 10), 0.5)
        self.assertEqual(linear_ramp(20, 10), 1.0)
        cfg = {
            "adaptation": {"distance": "l2"},
            "prototype_losses": {
                "enabled": True,
                "mode": "replacement",
                "use_sep_margin": True,
                "use_pair_margin": True,
                "target_weight_mode": "confidence_x_inverse_entropy",
                "lambda_proto_align": 0.1,
                "lambda_comp_source": 0.1,
                "lambda_comp_target": 0.05,
                "lambda_sep_margin": 0.05,
                "rampup_epochs": {"proto_align": 10, "comp_source": 1, "comp_target": 10, "sep_margin": 10},
                "uniform_margin": 5.0,
                "pair_margin": PAIR_MARGIN,
            },
        }
        validated = validate_prototype_loss_config(cfg, CLASS_NAMES)
        self.assertEqual(validated["mode"], "replacement")
        broken = {**cfg, "prototype_losses": {**cfg["prototype_losses"], "pair_margin": {**PAIR_MARGIN, "F": {**PAIR_MARGIN["F"], "N": 6.0}}}}
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_prototype_loss_config(broken, CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
