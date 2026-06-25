from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import cfg_path, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, inspect_daeac_npz, load_daeac_source_fit_val
from src.models.daeac_paper import DAEACNetwork


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_paper.yaml")
    parser.add_argument("--check-files", action="store_true", default=True)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    rr_mode = str(config["data"].get("rr_mode", "real"))
    class_names = list(config["data"]["class_names"])

    model = DAEACNetwork(num_classes=int(config["data"]["num_classes"]))
    features, logits, probs = model(torch.zeros(2, 1, 3, 128), return_logits=True)
    assert tuple(features.shape) == (2, 256), features.shape
    assert tuple(logits.shape) == (2, 4), logits.shape
    assert tuple(probs.shape) == (2, 4), probs.shape
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    print("DAEAC model forward OK: features=(2,256), logits=(2,4), probs=(2,4)")

    for key, require_labels in (
        ("source_train", True),
        ("source_eval", True),
        ("target_unlabeled", False),
        ("target_test", True),
    ):
        if key not in config["data"]:
            continue
        summary = inspect_daeac_npz(
            cfg_path(config, "data", key),
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=require_labels,
            rr_mode=rr_mode,
        )
        print(f"{key}: {summary}")
        _assert_real_rr_rows(key, summary, rr_mode)

    for name, value in dict(config["data"].get("external_targets", {})).items():
        path = Path(value)
        if not path.is_absolute():
            path = Path(config["_base_dir"]) / path
        if not path.exists():
            print(f"external_targets.{name}: skipped missing file {path}")
            continue
        summary = inspect_daeac_npz(
            path,
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=True,
            rr_mode=rr_mode,
        )
        print(f"external_targets.{name}: {summary}")
        _assert_real_rr_rows(f"external_targets.{name}", summary, rr_mode)

    source_train_path = cfg_path(config, "data", "source_train")
    source_eval_path = cfg_path(config, "data", "source_eval")
    if source_train_path.resolve() == source_eval_path.resolve():
        _, _, split_summary = load_daeac_source_fit_val(
            source_train_path,
            source_eval_path,
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            rr_mode=rr_mode,
            full_source_fit=str(config["data"].get("source_usage", "full")).lower() == "full",
        )
        print(f"source_train/source_eval share one file; source split: {split_summary}")

    target = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        rr_mode=rr_mode,
    )
    item = target[0]
    if not (isinstance(item, tuple) and len(item) == 2):
        raise AssertionError("Target unlabeled dataset must return only (x, index).")
    source = DAEACDataset(cfg_path(config, "data", "source_train"), input_key=input_key, label_key=label_key, class_names=class_names, rr_mode=rr_mode)
    print(f"Target unlabeled safety OK. Source samples={len(source)}, target_unlabeled samples={len(target)}")


def _assert_real_rr_rows(name: str, summary: dict, rr_mode: str) -> None:
    stats = summary["row_stats"]
    print(
        f"{name} row stats: "
        f"row0 mean={stats['row0_morphology']['mean']:.6f} std={stats['row0_morphology']['std']:.6f}; "
        f"row1 pre_rr mean={stats['row1_pre_rr_ratio']['mean']:.6f} std={stats['row1_pre_rr_ratio']['std']:.6f}; "
        f"row2 near_pre_rr mean={stats['row2_near_pre_rr_ratio']['mean']:.6f} std={stats['row2_near_pre_rr_ratio']['std']:.6f}"
    )
    if str(rr_mode).lower() == "real" and bool(summary["rr_rows_neutralized"]):
        raise AssertionError(f"{name}: rr_mode='real' but Row 1/2 are neutralized to constant 1.0.")


if __name__ == "__main__":
    main()
