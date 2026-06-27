from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from src.models.daeac_paper import LateFusionClassifierH
from src.training.train_daeac_adversarial import build_daeac_cdan_model
from scripts.phase6_daeac_paper.common import load_phase1_config


def test_cdan_builds_fcba_rr_late_fusion_model() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_phase1_config(str(root / "configs" / "phase6_daeac_fcba_latefusion_rr_nsv_cdan_ds1_ds2.yaml"))

    model = build_daeac_cdan_model(config, torch.device("cpu"), init_checkpoint=None)

    assert isinstance(model.classifier, LateFusionClassifierH)
    assert model.feature_dim == 128
    assert model.num_classes == 3
    assert config["data"]["return_rr_features"]
    assert config["data"]["morphology_only"]


def test_cdan_late_fusion_forward_and_domain_shapes() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_phase1_config(str(root / "configs" / "phase6_daeac_fcba_latefusion_rr_nsv_cdan_ds1_ds2.yaml"))
    model = build_daeac_cdan_model(config, torch.device("cpu"), init_checkpoint=None)
    x = torch.randn(4, 1, 1, 128)
    rr_features = torch.randn(4, 7)

    logits, features = model(x, rr_features=rr_features, return_embedding=True)
    domain_logits = model.forward_domain_from_features(features, logits, lambd=1.0)

    assert tuple(features.shape) == (4, 128)
    assert tuple(logits.shape) == (4, 3)
    assert tuple(domain_logits.shape) == (4, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(domain_logits).all()


def test_cdan_method_override_preserves_scenario_checkpoint_prefix() -> None:
    root = Path(__file__).resolve().parents[1]
    script_dir = root / "scripts" / "phase6_daeac_adversarial"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("phase6_train_cdan_test", script_dir / "01_train_cdan.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._should_override_cdan_prefix(None, None, "daeac_cdan_e")
    assert not module._should_override_cdan_prefix(None, None, "daeac_fcba_rr_nsv_cdan_ds1_ds2")
    assert not module._should_override_cdan_prefix("custom_prefix", None, "daeac_cdan_e")
