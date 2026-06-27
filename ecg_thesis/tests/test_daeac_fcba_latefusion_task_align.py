from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_daeac_paper.common import load_phase1_config
from src.models.daeac_paper import LateFusionClassifierH
from src.training.daeac_losses import cluster_aligning_loss, distance_from_name, task_positive_features_from_logits
from src.training.train_daeac_paper import (
    _task_align_logits_from_features,
    batch_centers,
    build_daeac_model,
    build_pseudo_labeled_target_dataset,
)


class DAEACFCBALateFusionTaskAlignTests(unittest.TestCase):
    def test_task_positive_features_match_late_fusion_shapes_and_are_finite(self) -> None:
        model = _late_fusion_model()
        x, rr_features, labels = _source_batch(batch_size=6)

        output = model(x, rr_features=rr_features, return_dict=True)
        task_logits = _task_align_logits_from_features(model, output, output["features"], rr_features)
        z_pos = task_positive_features_from_logits(task_logits, output["features"], labels)

        self.assertEqual(tuple(x.shape), (6, 1, 1, 128))
        self.assertEqual(tuple(rr_features.shape), (6, 7))
        self.assertEqual(tuple(output["logits"].shape), (6, 3))
        self.assertEqual(tuple(output["features"].shape), (6, 128))
        self.assertEqual(tuple(z_pos.shape), (6, 128))
        self.assertTrue(torch.isfinite(z_pos).all())
        self.assertGreater(float(torch.linalg.vector_norm(z_pos.detach(), dim=1).min()), 1.0e-8)
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(z_pos, dim=1),
                torch.linalg.vector_norm(output["features"], dim=1),
                rtol=1.0e-4,
                atol=1.0e-6,
            )
        )

    def test_task_alignment_loss_is_finite_and_backward_works_with_detached_mask(self) -> None:
        model = _late_fusion_model()
        x, rr_features, labels = _source_batch(batch_size=6)
        target_features = torch.randn(6, 128)
        target_labels = torch.tensor([0, 1, 2, 0, 1, 2])
        distance_fn = distance_from_name("l2")

        output = model(x, rr_features=rr_features, return_dict=True)
        task_logits = _task_align_logits_from_features(model, output, output["features"], rr_features)
        z_pos = task_positive_features_from_logits(
            task_logits,
            output["features"],
            labels,
            detach_task_mask=True,
        )
        source_centers = batch_centers(z_pos, labels, num_classes=3)
        target_centers = batch_centers(target_features, target_labels, num_classes=3)
        loss_align = cluster_aligning_loss(source_centers, target_centers, distance_fn, torch.device("cpu"), reduction="mean")
        loss_total = F.cross_entropy(output["logits"], labels) + 0.1 * loss_align

        loss_total.backward()

        self.assertTrue(torch.isfinite(loss_align))
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_target_labels_are_not_accessed_when_pseudo_labeling(self) -> None:
        model = _late_fusion_model()
        target = _ForbiddenLabelTarget(torch.randn(4, 1, 1, 128), torch.randn(4, 7))

        pseudo = build_pseudo_labeled_target_dataset(
            model,
            model.classifier,
            target,
            DataLoader(target, batch_size=2, shuffle=False),
            thresholds=torch.zeros(3),
            device=torch.device("cpu"),
        )

        self.assertEqual(len(pseudo), 4)
        self.assertEqual(tuple(pseudo[0][0].shape), (1, 1, 128))
        self.assertEqual(tuple(pseudo[0][1].shape), (7,))

    def test_disabled_task_align_preserves_raw_center_alignment(self) -> None:
        features = torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        target_centers = [torch.tensor([1.5, 0.0]), torch.tensor([0.0, 2.5])]
        distance_fn = distance_from_name("l2")

        local_source = batch_centers(features, labels, num_classes=2)
        disabled_source_for_loss = local_source

        old_loss = cluster_aligning_loss(local_source, target_centers, distance_fn, torch.device("cpu"), reduction="mean")
        disabled_loss = cluster_aligning_loss(
            disabled_source_for_loss,
            target_centers,
            distance_fn,
            torch.device("cpu"),
            reduction="mean",
        )

        self.assertTrue(torch.allclose(disabled_loss, old_loss))

    def test_task_align_configs_load_for_all_five_late_fusion_scenarios(self) -> None:
        config_names = [
            "phase6_daeac_fcba_latefusion_rr_nsv_taskalign_ds1_ds2.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_taskalign_ds1_incart.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_taskalign_ds1_svdb.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_taskalign_mitbih_incart.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_taskalign_mitbih_svdb.yaml",
        ]
        root = Path(__file__).resolve().parents[1]
        for name in config_names:
            with self.subTest(name=name):
                config = load_phase1_config(str(root / "configs" / name))
                model = build_daeac_model(config, torch.device("cpu"))
                task_align = config["rtd_daeac"]["task_align"]
                thresholds = config["adaptation"]["pseudo_thresholds"]

                self.assertEqual(config["data"]["num_classes"], 3)
                self.assertEqual(config["data"]["rr_mode"], "real")
                self.assertTrue(config["data"]["morphology_only"])
                self.assertTrue(config["data"]["return_rr_features"])
                self.assertEqual(config["model"]["attention"], "fcba")
                self.assertTrue(config["model"]["late_fusion"]["enabled"])
                self.assertTrue(task_align["enabled"])
                self.assertEqual(task_align["feature_key"], "features")
                self.assertTrue(task_align["source_positive_only"])
                self.assertFalse(task_align["target_positive_if_reliable"])
                self.assertTrue(task_align["detach_task_mask"])
                self.assertEqual(float(thresholds["N"]), 0.999)
                self.assertEqual(float(thresholds["S"]), 0.90)
                self.assertEqual(float(thresholds["V"]), 0.97)
                self.assertIsInstance(model.classifier, LateFusionClassifierH)
                self.assertEqual(model.feature_dim, 128)


class _ForbiddenLabelTarget(Dataset):
    def __init__(self, x: torch.Tensor, rr_features: torch.Tensor):
        self.x = x
        self.rr_features = rr_features

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.rr_features[idx]

    @property
    def y(self):
        raise AssertionError("Target labels must not be accessed.")


def _late_fusion_model():
    torch.manual_seed(123)
    model = build_daeac_model(
        {
            "model": {
                "num_classes": 3,
                "input_channels": 1,
                "input_rows": 1,
                "initial_channels": 4,
                "feature_dim": 256,
                "dilations": [1, 6, 12, 18],
                "se_reduction": 16,
                "attention": "fcba",
                "fcba": {"frequency_modes": 4, "spatial_kernel_size": 7},
                "late_fusion": {"enabled": True, "rr_dim": 7, "fc1_dim": 128, "fc2_dim": 64},
                "dropout": 0.0,
            },
            "rtd_daeac": {"dual_head": {"enabled": False}},
        },
        torch.device("cpu"),
    )
    model.train()
    return model


def _source_batch(batch_size: int):
    torch.manual_seed(321)
    x = torch.randn(batch_size, 1, 1, 128)
    rr_features = torch.randn(batch_size, 7)
    labels = torch.arange(batch_size) % 3
    return x, rr_features, labels


if __name__ == "__main__":
    unittest.main()
