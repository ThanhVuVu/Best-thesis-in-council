from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.daeac_prototype_bank import (
    ReliabilityWeightedPrototypeBank,
    dense_batch_prototypes,
    pseudo_distribution_flags,
)


class ReliabilityWeightedPrototypeBankTest(unittest.TestCase):
    def test_state_is_buffer_only_with_expected_shapes(self) -> None:
        bank = self._bank()
        self.assertEqual(list(bank.parameters()), [])
        self.assertEqual(tuple(bank.source_prototypes.shape), (4, 3))
        self.assertEqual(tuple(bank.target_prototypes.shape), (4, 3))
        self.assertEqual(tuple(bank.global_prototypes.shape), (4, 3))
        self.assertEqual(tuple(bank.target_reliability.shape), (4,))
        self.assertEqual(tuple(bank.beta.shape), (4,))
        self.assertIn("source_prototypes", dict(bank.named_buffers()))

    def test_source_ema_target_first_update_and_min_count_gate(self) -> None:
        bank = self._bank()
        source_initial = torch.ones(4, 3)
        bank.initialize_source(source_initial, torch.tensor([10, 10, 10, 10]))
        source_local = torch.full((4, 3), 3.0)
        target_local = torch.full((4, 3), 5.0)
        candidates = bank.candidates(
            source_local,
            torch.tensor([2, 2, 2, 2]),
            target_local,
            torch.tensor([4, 3, 4, 0]),
        )
        self.assertTrue(torch.allclose(candidates.source, torch.full((4, 3), 1.2)))
        self.assertTrue(torch.allclose(candidates.target[0], torch.full((3,), 5.0)))
        self.assertTrue(torch.allclose(candidates.target[1], torch.zeros(3)))
        self.assertEqual(candidates.target_update_mask.tolist(), [True, False, True, False])
        bank.commit(candidates)
        second = bank.candidates(
            source_local,
            torch.zeros(4, dtype=torch.long),
            torch.full((4, 3), 7.0),
            torch.tensor([4, 4, 0, 0]),
        )
        self.assertTrue(torch.allclose(second.target[0], torch.full((3,), 5.02), atol=1e-6))
        self.assertTrue(torch.allclose(second.target[1], torch.full((3,), 7.0)))

    def test_invalid_target_falls_back_to_source_and_reliability_controls_beta(self) -> None:
        bank = self._bank()
        source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        bank.initialize_source(source, torch.ones(4, dtype=torch.long))
        self.assertTrue(torch.equal(bank.global_prototypes, source))
        self.assertTrue(torch.equal(bank.beta, torch.zeros(4)))
        candidates = bank.candidates(
            source,
            torch.zeros(4, dtype=torch.long),
            source + 10.0,
            torch.tensor([4, 0, 0, 0]),
        )
        bank.commit(candidates)
        stats = bank.update_reliability(
            torch.tensor([10, 10, 0, 0]),
            torch.tensor([5, 0, 0, 0]),
            torch.tensor([4.5, 0.0, 0.0, 0.0]),
            epoch=5,
        )
        self.assertAlmostEqual(float(stats["observed_reliability"][0]), 0.45, places=6)
        self.assertAlmostEqual(float(bank.target_reliability[0]), 0.045, places=6)
        self.assertAlmostEqual(float(bank.beta[0]), 0.00675, places=6)
        self.assertEqual(bank.beta[1:].tolist(), [0.0, 0.0, 0.0])
        self.assertLessEqual(float(bank.beta.max()), bank.beta_max)

    def test_candidate_graph_backpropagates_then_commit_detaches(self) -> None:
        bank = ReliabilityWeightedPrototypeBank(2, 2, min_target_count=1)
        bank.initialize_source(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([1, 1]))
        source_local = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        target_local = torch.tensor([[0.5, 0.5], [-0.5, 0.5]], requires_grad=True)
        candidates = bank.candidates(source_local, torch.tensor([1, 1]), target_local, torch.tensor([1, 1]))
        loss = torch.linalg.vector_norm(candidates.source - candidates.target, dim=1).sum()
        loss.backward()
        self.assertGreater(float(source_local.grad.abs().sum()), 0.0)
        self.assertGreater(float(target_local.grad.abs().sum()), 0.0)
        bank.commit(candidates)
        self.assertFalse(bank.source_prototypes.requires_grad)
        self.assertIsNone(bank.source_prototypes.grad_fn)

    def test_state_dict_round_trip_and_cpu_move(self) -> None:
        bank = self._bank()
        bank.initialize_source(torch.randn(4, 3), torch.tensor([2, 2, 2, 2]))
        candidate = bank.candidates(
            torch.randn(4, 3),
            torch.ones(4, dtype=torch.long),
            torch.randn(4, 3),
            torch.full((4,), 4, dtype=torch.long),
        )
        bank.commit(candidate)
        stream = io.BytesIO()
        torch.save(bank.state_dict(), stream)
        stream.seek(0)
        restored = self._bank().cpu()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        for key, value in bank.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), restored.state_dict()[key]))

    def test_dense_prototypes_and_all_n_flags(self) -> None:
        features = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], requires_grad=True)
        prototypes, counts = dense_batch_prototypes(features, torch.tensor([0, 0, 1]), num_classes=3)
        self.assertEqual(counts.tolist(), [2, 1, 0])
        self.assertTrue(torch.equal(prototypes[0], torch.tensor([2.0, 0.0])))
        prototypes.sum().backward()
        self.assertIsNotNone(features.grad)
        flags = pseudo_distribution_flags(torch.tensor([20, 0, 0, 0]), near_all_n_ratio=0.95)
        self.assertTrue(flags["all_n"])
        self.assertTrue(flags["near_all_n"])
        near = pseudo_distribution_flags(torch.tensor([95, 5, 0, 0]), near_all_n_ratio=0.95)
        self.assertFalse(near["all_n"])
        self.assertTrue(near["near_all_n"])

    @staticmethod
    def _bank() -> ReliabilityWeightedPrototypeBank:
        return ReliabilityWeightedPrototypeBank(
            num_classes=4,
            feature_dim=3,
            source_momentum=0.9,
            target_momentum=0.99,
            reliability_momentum=0.9,
            min_target_count=4,
            beta_max=0.3,
            rampup_epochs=10,
        )


if __name__ == "__main__":
    unittest.main()
