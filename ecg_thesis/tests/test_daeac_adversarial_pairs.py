from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from src.utils.io import load_config


ROOT = Path(__file__).resolve().parents[1]
PAIR_NAMES = ("ds1_ds2", "ds1_incart", "ds1_svdb", "mitbih_incart", "mitbih_svdb")


def _workflow_module():
    script_dir = ROOT / "scripts" / "phase6_daeac_adversarial"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("phase6_adversarial_workflow_test", script_dir / "workflow.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_domain_pair_protocols_and_unique_outputs() -> None:
    workflow = _workflow_module()
    for method, config_name in (
        ("dann", "phase6_daeac_dann.yaml"),
        ("adda", "phase6_daeac_adda.yaml"),
        ("cdan", "phase6_daeac_cdan.yaml"),
    ):
        outputs = set()
        for pair in PAIR_NAMES:
            config = load_config(ROOT / "configs" / config_name)
            workflow.apply_domain_pair(config, pair, method)
            outputs.add(config["paths"]["output_dir"])
            assert config["domain_pair"] == pair
            if pair == "ds1_ds2":
                assert "first5" in config["data"]["target_unlabeled"]
                assert config["data"]["target_unlabeled"] != config["data"]["target_test"]
            else:
                assert config["data"]["target_unlabeled"] == config["data"]["target_test"]
            if pair.startswith("mitbih_"):
                assert "mitdb_all" in config["data"]["source_train"]
        assert len(outputs) == len(PAIR_NAMES)


def test_original_adversarial_loss_weights_are_preserved() -> None:
    dann = load_config(ROOT / "configs" / "phase6_daeac_dann.yaml")
    cdan = load_config(ROOT / "configs" / "phase6_daeac_cdan.yaml")
    assert dann["dann"]["alpha"] == 1.0
    assert "loss_balance" not in dann["dann"]
    assert cdan["cdan"]["lambda_base"] == 0.2
    assert "loss_balance" not in cdan["cdan"]


def test_adversarial_notebooks_cover_all_pairs_and_are_clean_json() -> None:
    for method in ("dann", "adda", "cdan"):
        path = ROOT / "notebooks" / f"phase6_daeac_{method}_kaggle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        for pair in PAIR_NAMES:
            assert pair in code
        assert "--domain-pair" in code
        assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code")
