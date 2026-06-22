from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.daeac_prototype_bank import ReliabilityWeightedPrototypeBank, dense_batch_prototypes, pseudo_distribution_flags
from src.training.daeac_pseudo_filter import (
    filter_target_pseudolabels,
    normalized_entropy,
    pseudo_safety_reason,
    update_pseudo_safety_state,
    validate_pseudo_filter_config,
)


class DAEACPseudoFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = torch.tensor([0.999, 0.99, 0.99, 0.99])

    def _filter(self, probabilities, mode, global_threshold=0.99, entropy_threshold=0.05):
        return filter_target_pseudolabels(
            torch.tensor(probabilities, dtype=torch.float32),
            mode=mode,
            global_confidence_threshold=global_threshold,
            class_confidence_thresholds=self.thresholds,
            max_normalized_entropy=entropy_threshold,
        )

    def test_entropy_shapes_bounds_and_known_values(self) -> None:
        values = normalized_entropy(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.25] * 4]))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertAlmostEqual(float(values[0]), 0.0, places=6)
        self.assertAlmostEqual(float(values[1]), 1.0, places=6)

    def test_all_four_modes_and_strict_confidence_boundaries(self) -> None:
        probabilities = [[0.99, 0.01, 0.0, 0.0], [0.998, 0.002, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]
        self.assertEqual(self._filter(probabilities, "none").accepted_mask.tolist(), [True, True, True])
        self.assertEqual(self._filter(probabilities, "confidence_global").accepted_mask.tolist(), [False, True, False])
        class_specific = self._filter(probabilities, "class_specific", entropy_threshold=1.0)
        self.assertEqual(class_specific.accepted_mask.tolist(), [False, False, False])
        entropy_sample = [0.991, 0.009, 0.0, 0.0]
        entropy_boundary = float(normalized_entropy(torch.tensor([entropy_sample])).item())
        accepted = self._filter([entropy_sample], "confidence_entropy", entropy_threshold=entropy_boundary)
        self.assertTrue(bool(accepted.accepted_mask[0]))

    def test_rejection_reasons_are_disjoint(self) -> None:
        result = self._filter(
            [[0.97, 0.01, 0.01, 0.01], [0.99, 0.01, 0.0, 0.0], [0.7, 0.1, 0.1, 0.1]],
            "confidence_entropy",
            entropy_threshold=0.03,
        )
        total = result.accepted_mask.to(torch.long)
        total += result.rejected_confidence_mask
        total += result.rejected_entropy_mask
        total += result.rejected_both_mask
        self.assertEqual(total.tolist(), [1, 1, 1])

    def test_rejected_samples_do_not_update_target_or_receive_gradient(self) -> None:
        features = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
        accepted = torch.tensor([True, False])
        selected = features[accepted]
        prototypes, counts = dense_batch_prototypes(selected, torch.tensor([0]), 2)
        bank = ReliabilityWeightedPrototypeBank(2, 2, min_target_count=1)
        bank.initialize_source(torch.eye(2), torch.ones(2, dtype=torch.long))
        source, source_counts = dense_batch_prototypes(torch.eye(2), torch.tensor([0, 1]), 2)
        candidates = bank.candidates(source, source_counts, prototypes, counts)
        candidates.target.sum().backward()
        self.assertGreater(float(features.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(features.grad[1].abs().sum()), 0.0)
        self.assertEqual(candidates.target_update_mask.tolist(), [True, False])

    def test_distribution_distinguishes_empty_all_n_and_near_all_n(self) -> None:
        empty = pseudo_distribution_flags(torch.zeros(4, dtype=torch.long))
        all_n = pseudo_distribution_flags(torch.tensor([3, 0, 0, 0]))
        near = pseudo_distribution_flags(torch.tensor([95, 5, 0, 0]))
        self.assertEqual(empty["accepted_total"], 0)
        self.assertFalse(empty["all_n"])
        self.assertTrue(all_n["all_n"])
        self.assertTrue(near["near_all_n"])

    def test_safety_patience_tracks_empty_and_all_n_independently(self) -> None:
        empty = update_pseudo_safety_state(
            torch.zeros(4, dtype=torch.long), previous_empty_streak=1, previous_all_n_streak=2,
            near_all_n_ratio=0.95,
        )
        self.assertEqual(empty["empty_acceptance_streak"], 2)
        self.assertEqual(empty["all_n_streak"], 0)
        self.assertEqual(
            pseudo_safety_reason(empty, fail_on_empty=True, fail_on_all_n=True, patience=2),
            "no_target_pseudo_labels_accepted",
        )
        all_n = update_pseudo_safety_state(
            torch.tensor([8, 0, 0, 0]), previous_empty_streak=3, previous_all_n_streak=1,
            near_all_n_ratio=0.95,
        )
        self.assertEqual(all_n["empty_acceptance_streak"], 0)
        self.assertEqual(all_n["all_n_streak"], 2)
        self.assertEqual(
            pseudo_safety_reason(all_n, fail_on_empty=True, fail_on_all_n=True, patience=2),
            "all_accepted_target_pseudo_labels_are_N",
        )

    def test_config_requires_complete_class_order(self) -> None:
        config = {"pseudo_filter": {
            "enabled": True, "mode": "class_specific", "global_confidence_threshold": 0.99,
            "class_confidence_thresholds": {"N": 0.999, "S": 0.99, "V": 0.99},
            "max_normalized_entropy": 0.05, "near_all_n_ratio": 0.95, "safety_patience_epochs": 2,
        }}
        with self.assertRaisesRegex(ValueError, r"missing=\['F'\]"):
            validate_pseudo_filter_config(config, ["N", "S", "V", "F"])


if __name__ == "__main__":
    unittest.main()
