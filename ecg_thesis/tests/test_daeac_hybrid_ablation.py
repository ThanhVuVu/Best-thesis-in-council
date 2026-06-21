from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.daeac_protocol import audit_daeac_disjoint, create_daeac_after_time_split
from src.training.daeac_hybrid_ablation_losses import (
    class_balanced_conditional_mkmmd_loss,
    minority_weighted_mcc_loss,
    safe_topk_pseudolabel_mask,
    source_f_prototype_contrastive_loss,
)
from src.training.dan_mkmmd import make_mkmmd_gammas
from src.training.mcc_loss import minimum_class_confusion_loss


class DAEACHybridAblationLossTest(unittest.TestCase):
    def test_conditional_mkmmd_is_finite_and_backpropagates(self) -> None:
        source = torch.randn(8, 5, requires_grad=True)
        target = torch.randn(8, 5, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        probabilities = torch.softmax(torch.randn(8, 4), dim=1)
        gammas = make_mkmmd_gammas(2.0, kernel_num=3, kernel_mul=2.0, gamma_min=1.0e-6)
        loss, per_class, diagnostics = class_balanced_conditional_mkmmd_loss(
            source, target, labels, probabilities, gammas, torch.ones(3) / 3, torch.tensor([1.0, 1.0, 1.0, 2.0]), min_target_mass=0.1
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(per_class), {"0", "1", "2", "3"})
        self.assertEqual(float(diagnostics["active_3"]), 1.0)
        loss.backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)

    def test_safe_topk_never_selects_unsafe_or_exceeds_quota(self) -> None:
        pseudo = torch.tensor([3, 3, 3, 3, 0])
        confidence = torch.tensor([0.99, 0.95, 0.89, 0.98, 0.999])
        margin = torch.tensor([0.20, 0.05, 0.30, 0.15, 0.50])
        names = ["N", "S", "V", "F"]
        mask, diag = safe_topk_pseudolabel_mask(
            pseudo,
            confidence,
            margin,
            names,
            {"N": 0.995, "S": 0.95, "V": 0.98, "F": 0.90},
            {"N": 0.0, "S": 0.05, "V": 0.05, "F": 0.10},
            {"N": 64, "S": 32, "V": 32, "F": 1},
        )
        self.assertEqual(mask.nonzero().flatten().tolist(), [0, 4])
        self.assertEqual(diag["selected_F"], 1.0)

    def test_uniform_weighted_mcc_matches_standard_mcc(self) -> None:
        logits = torch.tensor([[2.0, 0.5, -0.3, 0.1], [0.2, 1.4, 0.7, -0.5], [0.1, -0.4, 2.1, 0.3]])
        standard = minimum_class_confusion_loss(logits, temperature=2.5)
        weighted = minority_weighted_mcc_loss(logits, torch.ones(4), temperature=2.5)
        self.assertTrue(torch.allclose(standard, weighted, atol=1.0e-6))

    def test_prototype_loss_prefers_correct_geometry(self) -> None:
        prototypes = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 0.5]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([-1.0, 0.0]),
        ]
        labels = torch.tensor([0, 2, 3])
        good = torch.stack([prototypes[0], prototypes[2], prototypes[3]]).requires_grad_()
        bad = torch.stack([prototypes[3], prototypes[3], prototypes[0]]).requires_grad_()
        good_loss, _ = source_f_prototype_contrastive_loss(good, labels, prototypes, ["N", "S", "V", "F"])
        bad_loss, _ = source_f_prototype_contrastive_loss(bad, labels, prototypes, ["N", "S", "V", "F"])
        self.assertLess(float(good_loss.detach()), float(bad_loss.detach()))
        good_loss.backward()
        self.assertIsNotNone(good.grad)


class DAEACAfter5ProtocolTest(unittest.TestCase):
    def test_after5_split_is_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full.npz"
            first = root / "first.npz"
            after = root / "after.npz"
            common = {
                "class_names": np.asarray(["N", "S", "V", "F"], dtype=object),
                "config_json": np.asarray("{}", dtype=object),
            }
            np.savez_compressed(
                full,
                x=np.zeros((4, 1, 3, 128), dtype=np.float32),
                y=np.asarray([0, 1, 2, 3]),
                record=np.asarray(["100"] * 4),
                r_peak_sample=np.asarray([10, 20, 30, 40]),
                r_peak_time_sec=np.asarray([10.0, 299.9, 300.0, 301.0]),
                **common,
            )
            np.savez_compressed(
                first,
                x=np.zeros((2, 1, 3, 128), dtype=np.float32),
                y=np.asarray([0, 1]),
                record=np.asarray(["100", "100"]),
                r_peak_sample=np.asarray([10, 20]),
                r_peak_time_sec=np.asarray([10.0, 299.9]),
                **common,
            )
            summary = create_daeac_after_time_split(full, after, threshold_sec=300.0)
            audit = audit_daeac_disjoint(first, after)
            self.assertEqual(summary["samples"], 2)
            self.assertTrue(audit["disjoint"])
            with np.load(after, allow_pickle=True) as data:
                self.assertEqual(data["class_names"].tolist(), ["N", "S", "V", "F"])


if __name__ == "__main__":
    unittest.main()
