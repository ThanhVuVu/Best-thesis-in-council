from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_daeac_prototype_bank import validate_prototype_bank_config


VARIANT_FLAGS = {
    "legacy_control": (),
    "align_only": ("use_proto_align",),
    "comp_source_only": ("use_comp_source",),
    "comp_target_only": ("use_comp_target",),
    "sep_uniform_only": ("use_sep_margin",),
    "sep_pair_only": ("use_sep_margin", "use_pair_margin"),
    "full_uniform": ("use_proto_align", "use_comp_source", "use_comp_target", "use_sep_margin"),
    "full_pair": ("use_proto_align", "use_comp_source", "use_comp_target", "use_sep_margin", "use_pair_margin"),
}


class PrototypeLossWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "scripts/phase6_daeac_proto_loss/common.py"
        spec = importlib.util.spec_from_file_location("plan3_test_common", path)
        cls.common = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.common)

    def test_all_variants_resolve_to_unique_outputs_and_expected_flags(self) -> None:
        outputs = set()
        prefixes = set()
        flag_names = ("use_proto_align", "use_comp_source", "use_comp_target", "use_sep_margin", "use_pair_margin")
        for variant, expected in VARIANT_FLAGS.items():
            config = self.common.load_phase1_config(str(ROOT / f"configs/phase6_daeac_proto_loss_{variant}.yaml"))
            validate_prototype_bank_config(config)
            outputs.add(config["paths"]["output_dir"])
            prefixes.add(config["adaptation"]["checkpoint_prefix"])
            active = tuple(name for name in flag_names if config["prototype_losses"].get(name, False))
            self.assertEqual(active, expected)
            self.assertEqual(config["data"]["class_names"], ["N", "S", "V", "F"])
            self.assertEqual(config["pseudo_filter"]["mode"], "class_specific")
            self.assertEqual(config["adaptation"]["batchnorm_mode"], "freeze_all")
            self.assertEqual(config["adaptation"]["target_forward_mode"], "single")
            self.assertEqual(config["adaptation"]["epoch_driver"], "target_once")
            self.assertEqual(config["adaptation"]["training_semantics_version"], 3)
            self.assertEqual(float(config["adaptation"]["lr"]), 0.005)
            self.assertEqual(config["adaptation"]["cluster_loss_reduction"], "sum")
        self.assertEqual(len(outputs), len(VARIANT_FLAGS))
        self.assertEqual(len(prefixes), len(VARIANT_FLAGS))

    def test_kaggle_notebook_is_clean_and_covers_every_domain_pair(self) -> None:
        path = ROOT / "notebooks/phase6_daeac_proto_loss_kaggle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
        for pair in ("ds1_ds2", "ds1_incart", "ds1_svdb", "mitbih_incart", "mitbih_svdb"):
            self.assertIn(repr(pair), code)
            self.assertTrue((ROOT / f"configs/phase6_daeac_pair_{pair}.yaml").exists())
        self.assertIn("/kaggle/working", code)
        self.assertIn("RUN_FULL = False", code)
        self.assertIn("DOMAIN_PAIR = 'ds1_ds2'", code)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_domain_pair_configs_use_original_daeac_losses(self) -> None:
        for pair in ("ds1_ds2", "ds1_incart", "ds1_svdb", "mitbih_incart", "mitbih_svdb"):
            config = self.common.load_phase1_config(str(ROOT / f"configs/phase6_daeac_pair_{pair}.yaml"))
            self.assertEqual(config["prototype_bank"]["usage"], "logging_only")
            self.assertEqual(config["prototype_losses"]["mode"], "legacy")
            self.assertFalse(config["prototype_losses"]["enabled"])
            self.assertEqual(config["pseudo_filter"]["max_normalized_entropy"], 1.0)
            self.assertEqual(config["adaptation"]["lr"], 0.005)
            if pair == "ds1_ds2":
                self.assertEqual(config["data"]["target_protocol"], "first5_adapt_full_test")
                self.assertEqual(config["data"]["target_test"], config["data"]["target_full_transductive"])
            else:
                self.assertEqual(config["data"]["target_protocol"], "full_target_transductive")
                self.assertEqual(config["data"]["target_unlabeled"], config["data"]["target_test"])
                self.assertEqual(config["data"]["target_test"], config["data"]["target_full_transductive"])


if __name__ == "__main__":
    unittest.main()
