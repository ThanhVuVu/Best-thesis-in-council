from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase6_daeac_paper.common import load_phase1_config
from src.models.daeac_paper import LateFusionClassifierH
from src.training.daeac_losses import DynamicWeightController
from src.training.train_daeac_paper import build_daeac_model, build_pseudo_labeled_target_dataset


class DAEACFCBALateFusionDynamicWeightTests(unittest.TestCase):
    def test_late_fusion_nsv_shapes_match_dynamic_weight_inputs(self) -> None:
        model = _late_fusion_model()
        x, rr_features, labels = _source_batch(batch_size=6)
        output = model(x, rr_features=rr_features, return_dict=True)
        z_t = torch.randn(6, 128)

        state = DynamicWeightController(beta1=0.1, beta2=0.1).update(
            output["features"],
            z_t,
            labels,
            epoch=1,
        )

        self.assertEqual(tuple(x.shape), (6, 1, 1, 128))
        self.assertEqual(tuple(rr_features.shape), (6, 7))
        self.assertEqual(tuple(output["logits"].shape), (6, 3))
        self.assertEqual(tuple(output["features"].shape), (6, 128))
        self.assertTrue(torch.isfinite(torch.tensor(list(state.as_dict().values()))).all())

    def test_tau_and_lambdas_are_finite_bounded_and_clipped(self) -> None:
        z_s = torch.randn(9, 128)
        z_t = torch.randn(9, 128) + 10.0
        y_s = torch.arange(9) % 3

        controller = DynamicWeightController(beta1=10.0, beta2=10.0, rampup_epochs=1, clip_min=0.0, clip_max=0.1)
        controller.update(z_s, z_t, y_s, epoch=1)
        state = controller.update(z_s + 1.0, z_t - 1.0, y_s, epoch=2)

        self.assertGreaterEqual(state.tau, 0.0)
        self.assertLessEqual(state.tau, 1.0)
        self.assertGreaterEqual(state.lambda_align, 0.0)
        self.assertLessEqual(state.lambda_align, 0.1)
        self.assertGreaterEqual(state.lambda_sep, 0.0)
        self.assertLessEqual(state.lambda_sep, 0.1)
        self.assertGreaterEqual(state.lambda_comp, 0.0)
        self.assertLessEqual(state.lambda_comp, 0.1)

    def test_first_batch_equal_min_max_has_no_nan(self) -> None:
        z_s = torch.randn(6, 128)
        z_t = torch.randn(6, 128)
        y_s = torch.tensor([0, 0, 1, 1, 2, 2])

        state = DynamicWeightController(beta1=0.1, beta2=0.1).update(z_s, z_t, y_s, epoch=1)

        values = torch.tensor(list(state.as_dict().values()))
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(state.tau, 0.0)

    def test_missing_classes_and_zero_within_scatter_are_finite(self) -> None:
        z_s = torch.ones(6, 128)
        z_t = torch.zeros(6, 128)
        y_s = torch.zeros(6, dtype=torch.long)

        state = DynamicWeightController(beta1=0.1, beta2=0.1).update(z_s, z_t, y_s, epoch=1)

        self.assertTrue(torch.isfinite(torch.tensor(list(state.as_dict().values()))).all())

    def test_ema_smooths_across_updates(self) -> None:
        y_s = torch.tensor([0, 1, 2, 0])
        z_s = torch.zeros(4, 128)
        z_t_one = torch.ones(4, 128)
        z_t_two = torch.full((4, 128), 2.0)
        raw_one = float(torch.linalg.vector_norm(z_s.mean(dim=0) - z_t_one.mean(dim=0)))
        raw_two = float(torch.linalg.vector_norm(z_s.mean(dim=0) - z_t_two.mean(dim=0)))

        controller = DynamicWeightController(beta1=0.1, beta2=0.1, ema_momentum=0.5)
        first = controller.update(z_s, z_t_one, y_s, epoch=1)
        second = controller.update(z_s, z_t_two, y_s, epoch=2)

        self.assertAlmostEqual(first.mmd, raw_one)
        self.assertGreater(second.mmd, raw_one)
        self.assertLess(second.mmd, raw_two)

    def test_disabled_fixed_state_preserves_old_beta_formula(self) -> None:
        loss_cls = torch.tensor(2.0)
        loss_align = torch.tensor(3.0)
        loss_sep = torch.tensor(5.0)
        loss_comp = torch.tensor(7.0)
        state = DynamicWeightController(beta1=0.1, beta2=0.2).fixed_state()

        dynamic_formula = loss_cls + state.lambda_align * loss_align + state.lambda_sep * loss_sep + state.lambda_comp * loss_comp
        old_formula = loss_cls + 0.1 * loss_align + 0.2 * (loss_sep + loss_comp)

        self.assertTrue(torch.allclose(dynamic_formula, old_formula))

    def test_enabled_dynamic_state_replaces_fixed_betas(self) -> None:
        z_s = torch.zeros(6, 128)
        z_t = torch.ones(6, 128)
        y_s = torch.tensor([0, 1, 2, 0, 1, 2])

        state = DynamicWeightController(beta1=0.1, beta2=0.1, rampup_epochs=10).update(z_s, z_t, y_s, epoch=1)

        self.assertNotEqual(state.lambda_sep, 0.1)
        self.assertNotEqual(state.lambda_comp, 0.1)

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

    def test_dynamic_configs_load_for_all_five_late_fusion_scenarios(self) -> None:
        config_names = [
            "phase6_daeac_fcba_latefusion_rr_nsv_dynamic_ds1_ds2.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dynamic_ds1_incart.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dynamic_ds1_svdb.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dynamic_mitbih_incart.yaml",
            "phase6_daeac_fcba_latefusion_rr_nsv_dynamic_mitbih_svdb.yaml",
        ]
        root = Path(__file__).resolve().parents[1]
        for name in config_names:
            with self.subTest(name=name):
                config = load_phase1_config(str(root / "configs" / name))
                model = build_daeac_model(config, torch.device("cpu"))
                dynamic_cfg = config["rtd_daeac"]["dynamic_weight"]

                self.assertEqual(config["data"]["num_classes"], 3)
                self.assertEqual(config["data"]["rr_mode"], "real")
                self.assertTrue(config["data"]["morphology_only"])
                self.assertTrue(config["data"]["return_rr_features"])
                self.assertEqual(config["model"]["attention"], "fcba")
                self.assertTrue(config["model"]["late_fusion"]["enabled"])
                self.assertTrue(dynamic_cfg["enabled"])
                self.assertEqual(dynamic_cfg["signal"], "mean_feature_distance")
                self.assertEqual(float(dynamic_cfg["clip_max"]), 0.1)
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
    model.eval()
    return model


def _source_batch(batch_size: int):
    torch.manual_seed(321)
    x = torch.randn(batch_size, 1, 1, 128)
    rr_features = torch.randn(batch_size, 7)
    labels = torch.arange(batch_size) % 3
    return x, rr_features, labels


if __name__ == "__main__":
    unittest.main()
