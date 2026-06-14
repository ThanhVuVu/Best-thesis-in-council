from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from common import cfg_path, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, inspect_daeac_npz
from src.utils.io import write_json


TRAIN_SCRIPT_NAMES = (
    "scripts/phase6_daeac_mcc/01_train.py",
    "scripts/phase6_daeac_mcc/02_train_hybrid_mkmmd_mcc.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_mcc.yaml")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))

    report: dict[str, Any] = {"config": args.config, "strict": bool(args.strict), "checks": [], "warnings": []}
    _check_dataset_paths(config, report)
    _check_train_scripts_do_not_load_eval_data(config, report)
    _inspect_all_npz(config, class_names, input_key, label_key, report)
    _check_unlabeled_dataset_contract(config, class_names, input_key, label_key, report)
    if not args.skip_overlap:
        _check_target_overlap(config, input_key, label_key, class_names, bool(args.strict), report)

    output = cfg_path(config, "paths", "output_dir") / "diagnostics" / "protocol_check.json"
    write_json(report, output)
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"protocol_check written to {output}")


def _check_dataset_paths(config: dict[str, Any], report: dict[str, Any]) -> None:
    target_unlabeled = cfg_path(config, "data", "target_unlabeled")
    target_test = cfg_path(config, "data", "target_test")
    if target_unlabeled == target_test:
        raise ValueError("Leakage risk: data.target_unlabeled points to the same path as data.target_test.")
    report["checks"].append("target_unlabeled_path_is_not_target_test_path")
    for key in ("source_train", "source_eval", "target_unlabeled", "target_test"):
        path = cfg_path(config, "data", key)
        if not path.exists():
            raise FileNotFoundError(f"Missing data.{key}: {path}")
    for name, value in dict(config["data"].get("external_targets", {})).items():
        path = _resolve(config, value)
        if not path.exists():
            raise FileNotFoundError(f"Missing external target '{name}': {path}")
    report["checks"].append("all_configured_dataset_paths_exist")


def _check_train_scripts_do_not_load_eval_data(config: dict[str, Any], report: dict[str, Any]) -> None:
    base = Path(config["_base_dir"])
    forbidden = ("target_test", "external_targets")
    for rel in TRAIN_SCRIPT_NAMES:
        path = base / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden if term in text]
        if found:
            raise ValueError(f"{rel} contains eval-only config keys in train script: {found}.")
    report["checks"].append("train_scripts_do_not_reference_target_test_or_external_targets")


def _inspect_all_npz(
    config: dict[str, Any],
    class_names: list[str],
    input_key: str,
    label_key: str,
    report: dict[str, Any],
) -> None:
    summaries = {}
    for key in ("source_train", "source_eval", "target_test"):
        summaries[key] = inspect_daeac_npz(
            cfg_path(config, "data", key),
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=True,
        )
    summaries["target_unlabeled"] = inspect_daeac_npz(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        require_labels=False,
    )
    for name, value in dict(config["data"].get("external_targets", {})).items():
        summaries[f"external_{name}"] = inspect_daeac_npz(
            _resolve(config, value),
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=True,
        )
    report["dataset_summaries"] = summaries
    report["checks"].append("dataset_shapes_labels_and_class_order_valid")


def _check_unlabeled_dataset_contract(
    config: dict[str, Any],
    class_names: list[str],
    input_key: str,
    label_key: str,
    report: dict[str, Any],
) -> None:
    ds = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    item = ds[0]
    if not isinstance(item, tuple) or len(item) != 2:
        raise ValueError("DAEACTargetUnlabeledDataset must return exactly (x, index), never labels.")
    report["target_unlabeled_file_has_label_key"] = bool(ds.y is not None)
    report["checks"].append("target_unlabeled_dataset_does_not_expose_labels")


def _check_target_overlap(
    config: dict[str, Any],
    input_key: str,
    label_key: str,
    class_names: list[str],
    strict: bool,
    report: dict[str, Any],
) -> None:
    source = DAEACDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        require_labels=False,
    )
    target = DAEACDataset(
        cfg_path(config, "data", "target_test"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        require_labels=True,
    )
    source_keys = _sample_keys(source)
    target_keys = _sample_keys(target)
    overlap = sorted(source_keys & target_keys)
    report["target_unlabeled_target_test_overlap_count"] = int(len(overlap))
    report["target_unlabeled_target_test_overlap_examples"] = overlap[:10]
    if overlap:
        message = (
            f"Found {len(overlap)} overlapping samples between target_unlabeled and target_test. "
            "This is transductive input overlap and is disallowed in strict mode."
        )
        if strict:
            raise ValueError(message)
        report["warnings"].append(message)
    report["checks"].append("target_unlabeled_target_test_overlap_checked")


def _sample_keys(dataset: DAEACDataset) -> set[str]:
    records = dataset.records
    samples = None
    for key in ("sample", "r_peak_sample", "r_peak_time_sec"):
        if key in dataset.data:
            samples = dataset.data[key]
            break
    if records is not None and samples is not None:
        return {f"{str(rec)}::{str(sample)}" for rec, sample in zip(records, samples)}
    return {_hash_sample(dataset.x[idx]) for idx in range(len(dataset))}


def _hash_sample(x: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(x)
    return hashlib.sha1(contiguous.view(np.uint8).tobytes()).hexdigest()


def _resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path


if __name__ == "__main__":
    main()
