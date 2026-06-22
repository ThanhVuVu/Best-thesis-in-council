from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.daeac_paper import DAEACNetwork
from src.training.daeac_losses import compacting_loss, l2_distance, separating_loss
from src.training.daeac_prototype_bank import ReliabilityWeightedPrototypeBank, candidate_lists
from src.training.train_daeac_paper import CenterMemory, load_daeac_checkpoint
from src.training.train_daeac_prototype_bank import _centers_for_usage, validate_prototype_bank_config


class DAEACPrototypeBankTrainingTest(unittest.TestCase):
    def test_weighted_global_losses_are_finite_and_reach_batch_features(self) -> None:
        bank = ReliabilityWeightedPrototypeBank(3, 2, source_momentum=0.9, target_momentum=0.9, min_target_count=1)
        bank.initialize_source(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), torch.ones(3, dtype=torch.long))
        bank.target_valid.fill_(True)
        bank.target_prototypes.copy_(torch.tensor([[0.8, 0.1], [0.1, 0.8], [-0.8, 0.1]]))
        bank.target_reliability.fill_(0.8)
        bank.beta.fill_(0.2)
        source_local = torch.tensor([[1.2, 0.0], [0.0, 1.2], [-1.2, 0.0]], requires_grad=True)
        target_local = torch.tensor([[0.7, 0.2], [0.2, 0.7], [-0.7, 0.2]], requires_grad=True)
        candidates = bank.candidates(source_local, torch.ones(3, dtype=torch.long), target_local, torch.ones(3, dtype=torch.long))
        global_centers = candidate_lists(candidates.global_, candidates.global_valid)
        features = torch.stack([source_local[0], source_local[1], target_local[2]])
        labels = torch.tensor([0, 1, 2])
        loss = separating_loss(global_centers, margin=2.0, distance_fn=l2_distance, device=torch.device("cpu"))
        loss = loss + compacting_loss(features, labels, global_centers, l2_distance, torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(source_local.grad.abs().sum()), 0.0)
        self.assertGreater(float(target_local.grad.abs().sum()), 0.0)

    def test_logging_only_selects_legacy_centers(self) -> None:
        bank = ReliabilityWeightedPrototypeBank(2, 2, min_target_count=1)
        bank.initialize_source(torch.eye(2), torch.ones(2, dtype=torch.long))
        candidates = bank.candidates(torch.eye(2), torch.ones(2, dtype=torch.long), torch.eye(2), torch.ones(2, dtype=torch.long))
        legacy = CenterMemory(2, 2, torch.device("cpu"))
        legacy.source = [torch.tensor([10.0, 0.0]), torch.tensor([0.0, 10.0])]
        legacy.target = [torch.tensor([8.0, 0.0]), torch.tensor([0.0, 8.0])]
        legacy.refresh_mixed()
        z_s = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        z_t = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        source, target, mixed = _centers_for_usage(
            "logging_only",
            candidates,
            legacy,
            z_s,
            torch.tensor([0, 1]),
            z_t,
            torch.tensor([0, 1]),
            gamma=0.1,
            num_classes=2,
        )
        self.assertTrue(torch.allclose(source[0], torch.tensor([9.2, 0.0])))
        self.assertTrue(torch.allclose(target[0], torch.tensor([7.3, 0.0])))
        self.assertTrue(torch.allclose(mixed[0], torch.tensor([8.25, 0.0])))

    def test_checkpoint_with_bank_state_remains_evaluator_compatible(self) -> None:
        config = {
            "model": {
                "num_classes": 4,
                "input_channels": 1,
                "initial_channels": 4,
                "feature_dim": 256,
                "dilations": [1, 6, 12, 18],
                "se_reduction": 16,
                "dropout": 0.0,
            }
        }
        model = DAEACNetwork()
        bank = ReliabilityWeightedPrototypeBank(4, 256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "prototype_bank_state_dict": bank.state_dict(),
                    "epoch": 0,
                },
                path,
            )
            loaded = load_daeac_checkpoint(path, config, torch.device("cpu"))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))

    def test_config_rejects_unknown_usage(self) -> None:
        with self.assertRaisesRegex(ValueError, "prototype_bank.usage"):
            validate_prototype_bank_config(
                {"prototype_bank": {"enabled": True, "usage": "mystery", "reliability_rule": "coverage_x_confidence"}}
            )


if __name__ == "__main__":
    unittest.main()
