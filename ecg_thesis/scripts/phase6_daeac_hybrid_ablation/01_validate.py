from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, inspect_daeac_npz, split_daeac_source_fit_val
from src.data.daeac_protocol import audit_daeac_disjoint, inspect_daeac_time_split
from src.training.train_daeac_hybrid_ablation import validate_ablation_config
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
    ablation = validate_ablation_config(config)
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    required = ("source_train", "source_eval", "target_unlabeled", "target_test", "target_full_transductive")
    datasets = {}
    for key in required:
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

    source = DAEACDataset(cfg_path(config, "data", "source_train"), input_key=input_key, label_key=label_key, class_names=class_names)
    _, _, source_split = split_daeac_source_fit_val(source)
    target_overlap = audit_daeac_disjoint(
        cfg_path(config, "data", "target_unlabeled"),
        cfg_path(config, "data", "target_test"),
    )
    after5 = inspect_daeac_time_split(
        cfg_path(config, "data", "target_test"),
        float(config["data"].get("target_split_seconds", 300.0)),
    )
    checkpoint = cfg_path(config, "adaptation", "init_checkpoint")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing focal-standard init checkpoint: {checkpoint}")
    if args.strict and not target_overlap["disjoint"]:
        raise ValueError(f"Strict protocol failed: {target_overlap['overlap_count']} target samples overlap.")
    if after5["time_min_sec"] is not None and after5["time_min_sec"] < float(config["data"]["target_split_seconds"]):
        raise ValueError(f"After5 target contains a sample before split threshold: {after5['time_min_sec']}")

    report = {
        "config": args.config,
        "ablation": ablation,
        "init_checkpoint": str(checkpoint),
        "source_split": source_split,
        "target_overlap": target_overlap,
        "after5": after5,
        "datasets": datasets,
        "optional_external": {
            name: str(cfg_path(config, "data", "external_targets", name))
            for name in config["data"].get("external_targets", {})
        },
    }
    output = cfg_path(config, "paths", "output_dir") / "diagnostics" / "protocol_validation.json"
    write_json(report, output)
    print(f"Protocol validation passed for {ablation}: {output}")


if __name__ == "__main__":
    main()
