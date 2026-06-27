from __future__ import annotations

import sys
import copy
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.daeac_paper import LateFusionClassifierH
from src.training.train_daeac_paper import _unpack_input_batch, _unpack_source_batch
from src.training.train_daeac_hybrid_mkmmd_mcc import _apply_pseudo_filter, _pseudo_collapse_guard


class HybridMkmmdMccPseudoGuardTest(unittest.TestCase):
    def test_late_fusion_batches_keep_rr_features(self) -> None:
        device = torch.device("cpu")
        x = torch.zeros(2, 1, 1, 128)
        rr = torch.ones(2, 7)
        y = torch.tensor([0, 1])

        x_s, rr_s, y_s = _unpack_source_batch((x, rr, y), device)
        x_t, rr_t = _unpack_input_batch((x, rr, torch.tensor([3, 4])), device)

        self.assertEqual(tuple(x_s.shape), (2, 1, 1, 128))
        self.assertEqual(tuple(rr_s.shape), (2, 7))
        self.assertEqual(y_s.tolist(), [0, 1])
        self.assertEqual(tuple(x_t.shape), (2, 1, 1, 128))
        self.assertEqual(tuple(rr_t.shape), (2, 7))

    def test_late_fusion_aux_classifier_snapshot_loads_state_dict(self) -> None:
        classifier = LateFusionClassifierH(feature_dim=256, num_classes=3, rr_dim=7, fc1_dim=128, fc2_dim=64)
        aux_classifier = copy.deepcopy(classifier).eval()

        incompatible = aux_classifier.load_state_dict(copy.deepcopy(classifier.state_dict()))

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_pseudo_filter_is_noop_when_disabled(self) -> None:
        confident = torch.tensor([True, True, True])
        pseudo = torch.tensor([0, 0, 1])
        confidence = torch.tensor([0.9, 0.8, 0.7])
        margin = torch.tensor([0.5, 0.4, 0.3])

        keep = _apply_pseudo_filter(confident, pseudo, confidence, margin, {}, ["N", "S", "V"], epoch=0)

        self.assertEqual(keep.tolist(), [True, True, True])

    def test_max_per_class_keeps_highest_confidence_examples(self) -> None:
        confident = torch.tensor([True, True, True, True, True])
        pseudo = torch.tensor([0, 0, 0, 1, 2])
        confidence = torch.tensor([0.91, 0.99, 0.95, 0.93, 0.92])
        margin = torch.ones(5)
        cfg = {"pseudo_filter": {"enabled": True, "max_per_class": {"N": 2}}}

        keep = _apply_pseudo_filter(confident, pseudo, confidence, margin, cfg, ["N", "S", "V"], epoch=0)

        self.assertEqual(keep.tolist(), [False, True, True, True, True])

    def test_collapse_guard_triggers_on_dominant_class_and_low_s(self) -> None:
        keep = torch.tensor([True] * 9 + [False])
        pseudo = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 2, 1])
        cfg = {
            "pseudo_collapse_guard": {
                "enabled": True,
                "max_class_ratio": 0.85,
                "min_class_ratio": {"S": 0.05},
                "scale": 0.0,
                "mcc_scale": 0.25,
            }
        }

        diag = _pseudo_collapse_guard(keep, pseudo, cfg, ["N", "S", "V"])

        self.assertEqual(diag["triggered"], 1.0)
        self.assertAlmostEqual(diag["align_scale"], 0.0)
        self.assertAlmostEqual(diag["comp_scale"], 0.0)
        self.assertAlmostEqual(diag["mcc_scale"], 0.25)

    def test_collapse_guard_keeps_full_scale_when_balanced(self) -> None:
        keep = torch.tensor([True, True, True, True])
        pseudo = torch.tensor([0, 0, 1, 2])
        cfg = {
            "pseudo_collapse_guard": {
                "enabled": True,
                "max_class_ratio": 0.85,
                "min_class_ratio": {"S": 0.05},
                "scale": 0.0,
            }
        }

        diag = _pseudo_collapse_guard(keep, pseudo, cfg, ["N", "S", "V"])

        self.assertEqual(diag["triggered"], 0.0)
        self.assertAlmostEqual(diag["align_scale"], 1.0)
        self.assertAlmostEqual(diag["comp_scale"], 1.0)
        self.assertAlmostEqual(diag["mcc_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
