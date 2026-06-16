from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.data.daeac_dataset import inspect_daeac_npz
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_dann.yaml")
    parser.add_argument("--require-external", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    class_names = list(config["data"]["class_names"])
    required = {
        "source_train": cfg_path(config, "data", "source_train"),
        "source_eval": cfg_path(config, "data", "source_eval"),
        "target_unlabeled": cfg_path(config, "data", "target_unlabeled"),
        "target_test": cfg_path(config, "data", "target_test"),
    }
    summary = {"required": {}, "external": {}, "missing_external": []}
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 6 DAEAC file for {name}: {path}")
        require_labels = name != "target_unlabeled"
        summary["required"][name] = inspect_daeac_npz(
            path,
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=require_labels,
        )

    for name, value in dict(config.get("data", {}).get("external_targets", {})).items():
        path = _resolve_data_path(config, value)
        if not path.exists():
            summary["missing_external"].append({"name": name, "path": str(path)})
            continue
        summary["external"][name] = inspect_daeac_npz(
            path,
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=True,
        )

    if args.require_external and summary["missing_external"]:
        raise FileNotFoundError(f"Missing external target files: {summary['missing_external']}")
    write_json(summary, output / "metrics" / "phase6_daeac_adversarial_data_validation.json")
    print("Phase 6 DAEAC adversarial data validation passed.")
    if summary["missing_external"]:
        print("Missing optional external targets:", summary["missing_external"])


def _resolve_data_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path


if __name__ == "__main__":
    main()
