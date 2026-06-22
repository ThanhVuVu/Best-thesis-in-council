from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.daeac_protocol import audit_daeac_disjoint, create_daeac_after_time_split, create_daeac_before_time_split
from src.models.daeac_paper import ClassifierH
from src.training.daeac_losses import cluster_aligning_loss, compacting_loss, l2_distance, separating_loss
from src.training.train_daeac_paper import build_pseudo_labeled_target_dataset


class _IdentityFeatureModel(nn.Module):
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DAEACAdaptationSpecificationTests(unittest.TestCase):
    def test_pseudo_snapshot_visits_all_target_and_is_immutable(self) -> None:
        target = TensorDataset(torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]]))
        loader = DataLoader(target, batch_size=2, shuffle=False)
        model = _IdentityFeatureModel()
        h = ClassifierH(feature_dim=2, num_classes=2)
        with torch.no_grad():
            h.fc.weight.copy_(torch.eye(2))
            h.fc.bias.zero_()
        snapshot = build_pseudo_labeled_target_dataset(
            model, h, target, loader, torch.tensor([0.8, 0.8]), torch.device("cpu")
        )
        self.assertEqual(snapshot.positions.tolist(), [0, 1, 2])
        self.assertEqual(snapshot.labels.tolist(), [0, 1, 0])
        with torch.no_grad():
            h.fc.weight.mul_(-1)
        self.assertEqual(snapshot.labels.tolist(), [0, 1, 0])

    def test_cluster_losses_use_mathematical_sum(self) -> None:
        device = torch.device("cpu")
        source = [torch.tensor([0.0]), torch.tensor([4.0])]
        target = [torch.tensor([1.0]), torch.tensor([6.0])]
        self.assertEqual(float(cluster_aligning_loss(source, target, l2_distance, device)), 3.0)
        mixed = [torch.tensor([0.0]), torch.tensor([2.0])]
        self.assertEqual(float(separating_loss(mixed, 3.0, l2_distance, device)), 2.0)
        features = torch.tensor([[1.0], [2.0], [5.0]])
        labels = torch.tensor([0, 0, 1])
        self.assertEqual(float(compacting_loss(features, labels, mixed, l2_distance, device)), 6.0)

    def test_classifier_receives_only_classification_gradient(self) -> None:
        feature_extractor = nn.Linear(2, 2, bias=False)
        classifier = nn.Linear(2, 2, bias=False)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([0, 1])
        z = feature_extractor(x)
        loss_cls = nn.functional.cross_entropy(classifier(z), y)
        loss_cluster = z.square().sum()
        grad_h_total = torch.autograd.grad(loss_cls + loss_cluster, classifier.weight, retain_graph=True)[0]
        grad_h_cls = torch.autograd.grad(loss_cls, classifier.weight, retain_graph=True)[0]
        grad_f_total = torch.autograd.grad(loss_cls + loss_cluster, feature_extractor.weight)[0]
        self.assertTrue(torch.allclose(grad_h_total, grad_h_cls))
        self.assertGreater(float(grad_f_total.abs().sum()), 0.0)

    def test_scheduler_decays_on_exact_200th_iteration(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=0.005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
        for _ in range(199):
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.005)
        optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.005 * 0.99)

    def test_first5_and_after5_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, before, after = root / "full.npz", root / "before.npz", root / "after.npz"
            np.savez_compressed(
                full,
                x=np.zeros((3, 1, 3, 128), dtype=np.float32),
                y=np.zeros(3, dtype=np.int64),
                record=np.asarray(["a", "a", "a"]),
                r_peak_sample=np.asarray([1, 2, 3]),
                r_peak_time_sec=np.asarray([10.0, 299.9, 300.0]),
                class_names=np.asarray(["N", "S", "V", "F"], dtype=object),
            )
            create_daeac_before_time_split(full, before)
            create_daeac_after_time_split(full, after)
            self.assertTrue(audit_daeac_disjoint(before, after)["disjoint"])
            with np.load(before) as data:
                self.assertEqual(len(data["x"]), 2)
            with np.load(after) as data:
                self.assertEqual(len(data["x"]), 1)


if __name__ == "__main__":
    unittest.main()
