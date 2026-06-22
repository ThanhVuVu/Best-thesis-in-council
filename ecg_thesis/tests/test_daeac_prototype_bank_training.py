from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.daeac_paper import ClassifierH, DAEACNetwork
from src.training.daeac_losses import compacting_loss, l2_distance, separating_loss
from src.training.daeac_prototype_bank import ReliabilityWeightedPrototypeBank, candidate_lists
from src.training.train_daeac_paper import CenterMemory, load_daeac_checkpoint
from src.training.train_daeac_prototype_bank import (
    _centers_for_usage,
    _validate_resume_execution_compatibility,
    configure_adaptation_batchnorm,
    forward_target_for_pseudolabels,
    paired_adaptation_batches,
    set_adaptation_train_mode,
    validate_prototype_bank_config,
)


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

    def test_freeze_all_batchnorm_preserves_buffers_and_affine_parameters(self) -> None:
        model = DAEACNetwork()
        configure_adaptation_batchnorm(model, "freeze_all")
        set_adaptation_train_mode(model, "freeze_all")
        batchnorm = [
            module for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        before = [(module.running_mean.clone(), module.running_var.clone()) for module in batchnorm]
        with torch.no_grad():
            model.extract_features(torch.randn(4, 1, 3, 128))
        self.assertTrue(batchnorm)
        for module, (mean, variance) in zip(batchnorm, before):
            self.assertFalse(module.training)
            self.assertFalse(module.weight.requires_grad)
            self.assertFalse(module.bias.requires_grad)
            self.assertTrue(torch.equal(module.running_mean, mean))
            self.assertTrue(torch.equal(module.running_var, variance))

    def test_target_once_epoch_never_cycles_target_loader(self) -> None:
        source = DataLoader(TensorDataset(torch.arange(10)), batch_size=2, shuffle=False)
        target = DataLoader(TensorDataset(torch.arange(3)), batch_size=1, shuffle=False)
        pairs = list(paired_adaptation_batches(source, target, epoch_driver="target_once"))
        target_values = [int(target_batch[0].item()) for _, target_batch in pairs]
        self.assertEqual(len(pairs), len(target))
        self.assertEqual(target_values, [0, 1, 2])

    def test_single_target_forward_is_consistent_and_keeps_feature_gradient(self) -> None:
        class CountingExtractor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)
                self.calls = 0

            def extract_features(self, inputs):
                self.calls += 1
                return self.layer(inputs)

        model = CountingExtractor()
        classifier = ClassifierH(feature_dim=2, num_classes=2)
        features, probabilities = forward_target_for_pseudolabels(
            model,
            classifier,
            torch.randn(4, 2),
            mode="single",
        )
        self.assertEqual(model.calls, 1)
        self.assertTrue(features.requires_grad)
        self.assertFalse(probabilities.requires_grad)
        features[:2].sum().backward()
        self.assertIsNotNone(model.layer.weight.grad)

    def test_old_adaptation_checkpoint_cannot_resume_corrected_semantics(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible adaptation training semantics"):
            _validate_resume_execution_compatibility(
                {"config": {"adaptation": {}}},
                {
                    "training_semantics_version": 2,
                    "batchnorm_mode": "freeze_all",
                    "target_forward_mode": "single",
                    "epoch_driver": "target_once",
                },
            )


if __name__ == "__main__":
    unittest.main()
