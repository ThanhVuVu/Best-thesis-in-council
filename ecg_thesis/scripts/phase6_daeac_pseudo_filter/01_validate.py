from __future__ import annotations

import argparse

import torch

from common import cfg_path, load_phase1_config
from src.data.daeac_dataset import DAEACTargetUnlabeledDataset
from src.data.daeac_protocol import audit_daeac_disjoint, inspect_daeac_time_split
from src.training.daeac_pseudo_filter import (
    class_threshold_tensor,
    filter_target_pseudolabels,
    validate_pseudo_filter_config,
)
from src.training.train_daeac_prototype_bank import validate_prototype_bank_config
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    if args.init_checkpoint:
        config["adaptation"]["init_checkpoint"] = args.init_checkpoint
    usage = validate_prototype_bank_config(config)
    names = list(config["data"]["class_names"])
    filter_cfg = validate_pseudo_filter_config(config, names)
    target = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=str(config["data"].get("input_key", "auto")),
        label_key=str(config["data"].get("label_key", "y")),
        class_names=names,
    )
    if not (isinstance(target[0], tuple) and len(target[0]) == 2):
        raise AssertionError("Target adaptation dataset must return (x, index), never y.")
    overlap = audit_daeac_disjoint(
        cfg_path(config, "data", "target_unlabeled"), cfg_path(config, "data", "target_test")
    )
    threshold = float(config["data"]["target_split_seconds"])
    protocol = str(config["data"].get("target_protocol", "first5_adapt_after5_test"))
    target_adaptation_time = inspect_daeac_time_split(cfg_path(config, "data", "target_unlabeled"), threshold)
    target_test_time = inspect_daeac_time_split(cfg_path(config, "data", "target_test"), threshold)
    checkpoint = cfg_path(config, "adaptation", "init_checkpoint")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing source-selected base checkpoint: {checkpoint}")
    probabilities = torch.full((2, len(names)), 1.0 / len(names))
    result = filter_target_pseudolabels(
        probabilities,
        mode=str(filter_cfg["mode"]),
        global_confidence_threshold=float(filter_cfg["global_confidence_threshold"]),
        class_confidence_thresholds=class_threshold_tensor(filter_cfg, names, torch.device("cpu")),
        max_normalized_entropy=float(filter_cfg["max_normalized_entropy"]),
    )
    if bool(((result.normalized_entropy < 0) | (result.normalized_entropy > 1)).any()):
        raise AssertionError("Normalized entropy escaped [0,1].")
    if args.strict and protocol == "first5_adapt_after5_test" and not overlap["disjoint"]:
        raise ValueError(f"Strict protocol failed: {overlap['overlap_count']} target samples overlap.")
    if protocol.startswith("first5_adapt"):
        if target_adaptation_time["time_max_sec"] is not None and target_adaptation_time["time_max_sec"] >= threshold:
            raise ValueError(f"Target adaptation includes a sample at/after {threshold}s.")
    elif protocol != "full_target_transductive":
        raise ValueError(f"Unknown data.target_protocol: {protocol}")
    report = {
        "config": args.config,
        "prototype_usage": usage,
        "pseudo_filter": filter_cfg,
        "init_checkpoint": str(checkpoint),
        "target_loader_returns_label": False,
        "target_protocol": protocol,
        "target_overlap": overlap,
        "target_adaptation_time": target_adaptation_time,
        "target_test_time": target_test_time,
        "clinical_consistency": "deferred_missing_morphology_keys",
    }
    output = cfg_path(config, "paths", "output_dir") / "diagnostics" / "protocol_validation.json"
    write_json(report, output)
    print(f"Pseudo-filter validation passed for {filter_cfg['mode']}: {output}")


if __name__ == "__main__":
    main()
