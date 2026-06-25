from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.models.daeac_paper import ClassifierH, DualClassifierH
from src.training.train_daeac_paper import (
    PseudoLabelBank,
    ReliablePseudoLabelSelector,
    batch_centers,
    build_daeac_model,
    build_pseudo_labeled_target_dataset,
    compute_global_pseudo_target_centers,
)
from src.utils.io import load_config


class _ToyDualModel(nn.Module):
    def __init__(self, feature_dim: int = 5, num_classes: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.classifier = DualClassifierH(feature_dim=feature_dim, num_classes=num_classes)
        with torch.no_grad():
            self.classifier.fc.weight.zero_()
            self.classifier.fc.bias.zero_()
            self.classifier.fc2.weight.zero_()
            self.classifier.fc2.bias.zero_()
            self.classifier.fc.weight[:, :num_classes].copy_(5.0 * torch.eye(num_classes))
            self.classifier.fc2.weight[:, :num_classes].copy_(5.0 * torch.eye(num_classes))
            self.classifier.fc2.weight[1, feature_dim - 1] = 5.0

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()

    def forward(self, x: torch.Tensor, return_logits: bool = False, return_dict: bool = False):
        features = self.extract_features(x)
        logits_1, logits_2 = self.classifier.forward_head_logits(features)
        logits = 0.5 * (logits_1 + logits_2)
        probs = torch.softmax(logits, dim=1)
        if return_dict:
            return {
                "features": features,
                "logits": logits,
                "probabilities": probs,
                "logits_1": logits_1,
                "logits_2": logits_2,
            }
        if return_logits:
            return features, logits, probs
        return features, probs


class _ToySingleModel(_ToyDualModel):
    def __init__(self, feature_dim: int = 5, num_classes: int = 4):
        nn.Module.__init__(self)
        self.feature_dim = feature_dim
        self.classifier = ClassifierH(feature_dim=feature_dim, num_classes=num_classes)


class _ForbiddenLabelTarget(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx]

    @property
    def y(self):
        raise AssertionError("Target labels must not be accessed.")


class RTDDAEACPhase2BReliablePseudoTests(unittest.TestCase):
    def test_selector_accepts_only_samples_passing_all_three_gates(self) -> None:
        model = _ToyDualModel()
        selector = ReliablePseudoLabelSelector(
            source_centers=torch.eye(4, 5),
            source_radii=torch.ones(4),
            discrepancy_threshold=torch.tensor(1.0),
            confidence_thresholds=torch.full((4,), 0.8),
            class_names=["N", "S", "V", "F"],
            min_samples_per_class=2,
        )
        target_x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],  # accepted
                [0.1, 0.0, 0.0, 0.0, 0.0],  # low confidence
                [3.0, 0.0, 0.0, 0.0, 0.0],  # too far from source center
                [1.0, 0.0, 0.0, 0.0, 1.0],  # high head discrepancy
            ]
        )
        dataset = _ForbiddenLabelTarget(target_x)
        snapshot = build_pseudo_labeled_target_dataset(
            model,
            model.classifier,
            dataset,
            DataLoader(dataset, batch_size=4, shuffle=False),
            torch.full((4,), 0.8),
            torch.device("cpu"),
            selector=selector,
        )

        self.assertEqual(snapshot.positions.tolist(), [0])
        self.assertEqual(snapshot.labels.tolist(), [0])
        diag = snapshot.reliable_diagnostics
        self.assertEqual(diag["reliable_confidence_pass_total"], 3)
        self.assertEqual(diag["reliable_distance_pass_total"], 3)
        self.assertEqual(diag["reliable_discrepancy_pass_total"], 3)
        self.assertEqual(diag["reliable_all_gates_pass_total"], 1)
        self.assertEqual(diag["reliable_accepted_count_N"], 1)
        self.assertEqual(diag["reliable_accepted_count_S"], 0)
        self.assertEqual(diag["reliable_accepted_count_V"], 0)
        self.assertEqual(diag["reliable_accepted_count_F"], 0)

    def test_bank_stores_values_by_target_index(self) -> None:
        bank = PseudoLabelBank(target_size=5, num_classes=4)
        bank.update(
            torch.tensor([2, 4]),
            torch.tensor([1, 3]),
            torch.tensor([0.91, 0.97]),
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.01, 0.02]),
            torch.tensor([0.1, 0.1]),
            {
                "confidence": torch.tensor([True, True]),
                "distance": torch.tensor([True, False]),
                "discrepancy": torch.tensor([True, True]),
            },
        )

        self.assertEqual(bank.accepted_positions().tolist(), [2])
        self.assertEqual(int(bank.labels[2]), 1)
        self.assertEqual(int(bank.labels[4]), 3)
        self.assertFalse(bool(bank.accepted[4]))

    def test_source_thresholds_are_percentile_based_and_finite(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "phase6_daeac_paper_dualhead_reliable.yaml")
        config["rtd_daeac"]["reliable_pseudo"]["distance_percentile"] = 50
        config["rtd_daeac"]["reliable_pseudo"]["discrepancy_percentile"] = 50
        source_x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.5, 0.0],
            ]
        )
        source_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        selector = ReliablePseudoLabelSelector.from_source(
            _ToyDualModel(),
            DataLoader(TensorDataset(source_x, source_y), batch_size=4, shuffle=False),
            config,
            torch.device("cpu"),
        )

        self.assertTrue(torch.isfinite(selector.source_radii).all())
        self.assertTrue(torch.isfinite(selector.discrepancy_threshold))
        self.assertAlmostEqual(float(selector.source_radii[0]), 0.25, places=6)

    def test_min_samples_skip_target_centers(self) -> None:
        features = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])

        centers = batch_centers(features, labels, num_classes=2, min_samples_per_class=2)

        self.assertIsNotNone(centers[0])
        self.assertIsNone(centers[1])

    def test_global_pseudo_target_centers_respect_min_samples(self) -> None:
        model = nn.Module()
        model.feature_dim = 2
        model.extract_features = lambda x: x.float()
        features = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(features, labels, torch.ones(3), torch.zeros(3)), batch_size=2)

        centers = compute_global_pseudo_target_centers(
            model, loader, torch.device("cpu"), num_classes=2, min_samples_per_class=2
        )

        self.assertIsNotNone(centers[0])
        self.assertIsNone(centers[1])

    def test_reliable_mode_requires_dual_head(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "phase6_daeac_paper_dualhead_reliable.yaml")
        source = TensorDataset(torch.randn(4, 5), torch.tensor([0, 1, 2, 3]))

        with self.assertRaisesRegex(ValueError, "dual_head"):
            ReliablePseudoLabelSelector.from_source(
                _ToySingleModel(),
                DataLoader(source, batch_size=4),
                config,
                torch.device("cpu"),
            )

    def test_confidence_only_flow_is_unchanged_when_selector_is_disabled(self) -> None:
        target = TensorDataset(torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]]))
        loader = DataLoader(target, batch_size=2, shuffle=False)
        model = nn.Module()
        model.extract_features = lambda x: x
        h = ClassifierH(feature_dim=2, num_classes=2)
        with torch.no_grad():
            h.fc.weight.copy_(torch.eye(2))
            h.fc.bias.zero_()

        snapshot = build_pseudo_labeled_target_dataset(
            model, h, target, loader, torch.tensor([0.8, 0.8]), torch.device("cpu")
        )

        self.assertEqual(snapshot.positions.tolist(), [0, 1, 2])
        self.assertEqual(snapshot.labels.tolist(), [0, 1, 0])
        self.assertFalse(hasattr(snapshot, "reliable_diagnostics"))

    def test_reliable_config_loads_and_keeps_real_rr(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "phase6_daeac_paper_dualhead_reliable.yaml")
        model = build_daeac_model(config, torch.device("cpu"))

        self.assertEqual(config["data"]["rr_mode"], "real")
        self.assertTrue(config["rtd_daeac"]["dual_head"]["enabled"])
        self.assertTrue(config["rtd_daeac"]["reliable_pseudo"]["enabled"])
        self.assertIsInstance(model.classifier, DualClassifierH)


if __name__ == "__main__":
    unittest.main()
