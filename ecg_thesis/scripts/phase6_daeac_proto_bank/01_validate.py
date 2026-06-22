from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, inspect_daeac_npz, split_daeac_source_fit_val
from src.data.daeac_protocol import audit_daeac_disjoint, inspect_daeac_time_split
from src.training.train_daeac_prototype_bank import build_prototype_bank, validate_prototype_bank_config
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
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    datasets = {}
    for key in ("source_train", "source_eval", "target_unlabeled", "target_test", "target_full_transductive"):
        path = cfg_path(config, "data", key)
        if not path.exists():
            raise FileNotFoundError(f"Missing data.{key}: {path}")
        datasets[key] = inspect_daeac_npz(
            path,
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=key != "target_unlabeled",
        )
    source = DAEACDataset(
        cfg_path(config, "data", "source_train"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    _, _, source_split = split_daeac_source_fit_val(source)
    target = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    if not (isinstance(target[0], tuple) and len(target[0]) == 2):
        raise AssertionError("Target adaptation dataset must return (x, index), never y.")
    overlap = audit_daeac_disjoint(
        cfg_path(config, "data", "target_unlabeled"),
        cfg_path(config, "data", "target_test"),
    )
    after5 = inspect_daeac_time_split(
        cfg_path(config, "data", "target_test"),
        float(config["data"]["target_split_seconds"]),
    )
    checkpoint = cfg_path(config, "adaptation", "init_checkpoint")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing source-selected base checkpoint: {checkpoint}")
    bank = build_prototype_bank(config, device="cpu")
    if list(bank.parameters()):
        raise AssertionError("Prototype bank must not expose trainable parameters.")
    if args.strict and not overlap["disjoint"]:
        raise ValueError(f"Strict protocol failed: {overlap['overlap_count']} target samples overlap.")
    threshold = float(config["data"]["target_split_seconds"])
    if after5["time_min_sec"] is not None and after5["time_min_sec"] < threshold:
        raise ValueError(f"Target test includes a sample before {threshold}s: {after5['time_min_sec']}")
    report = {
        "config": args.config,
        "usage": usage,
        "init_checkpoint": str(checkpoint),
        "source_split": source_split,
        "target_overlap": overlap,
        "after5": after5,
        "target_loader_returns_label": False,
        "prototype_bank_parameters": 0,
        "datasets": datasets,
    }
    output = cfg_path(config, "paths", "output_dir") / "diagnostics" / "protocol_validation.json"
    write_json(report, output)
    print(f"Prototype-bank validation passed for {usage}: {output}")


if __name__ == "__main__":
    main()
