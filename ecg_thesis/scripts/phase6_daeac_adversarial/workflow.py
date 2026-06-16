from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Subset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, subset_first
from src.data.splits import mitbih_fit_val_records
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


def train_parser(default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    add_wandb_args(parser)
    return parser


def prepare_train_run(args, method_section: str) -> tuple[dict[str, Any], Any, Any, Any, Any, Any]:
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.init_checkpoint is not None:
        config["training"]["init_checkpoint"] = str(args.init_checkpoint)
    if args.checkpoint_prefix is not None:
        config["training"]["checkpoint_prefix"] = str(args.checkpoint_prefix)
    if method_section == "cdan" and "method" in config.get("cdan", {}):
        config["cdan"]["method"] = str(config["cdan"]["method"])
    set_seed(int(config["seed"]))
    device = device_from_torch()
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    class_names = list(config["data"]["class_names"])
    source_path = cfg_path(config, "data", "source_train")
    val_path = cfg_path(config, "data", "source_eval")
    source_full = DAEACDataset(source_path, input_key=input_key, label_key=label_key, class_names=class_names)
    if _same_path(source_path, val_path) and bool(config.get("source_split", {}).get("enabled", True)):
        source_ds, val_ds = _split_source_fit_val(source_full, config)
    else:
        source_ds = source_full
        val_ds = DAEACDataset(val_path, input_key=input_key, label_key=label_key, class_names=class_names)
    target_ds = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    source_ds = subset_first(source_ds, args.max_source_samples)
    val_ds = subset_first(val_ds, args.max_val_samples)
    target_ds = subset_first(target_ds, args.max_target_samples)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    return config, source_ds, val_ds, target_ds, output, device


def _same_path(left: Path, right: Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _split_source_fit_val(dataset: DAEACDataset, config: dict[str, Any]) -> tuple[Subset, Subset]:
    split_cfg = dict(config.get("source_split", {}))
    mode = str(split_cfg.get("mode", "mitbih_fit_val_records")).lower()
    records = dataset.records
    if records is None:
        raise ValueError("source_train/source_eval share one file, but DAEAC source data has no record metadata for record-wise validation split.")
    record_strings = np.asarray([str(value) for value in records])
    if mode != "mitbih_fit_val_records":
        raise ValueError(f"Unsupported Phase 6 adversarial source_split.mode: {mode!r}")
    fit_records, val_records = mitbih_fit_val_records()
    fit_set = set(fit_records)
    val_set = set(val_records)
    fit_idx = [idx for idx, rec in enumerate(record_strings) if rec in fit_set]
    val_idx = [idx for idx, rec in enumerate(record_strings) if rec in val_set]
    if not fit_idx or not val_idx:
        present = sorted(set(record_strings))
        raise ValueError(
            "Phase 6 adversarial source fit/validation split is empty. "
            f"Expected fit records={fit_records}, val records={val_records}, present records={present}."
        )
    overlap = sorted(set(record_strings[fit_idx]) & set(record_strings[val_idx]))
    if overlap:
        raise ValueError(f"Record overlap between source fit and validation splits: {overlap}")
    print(
        "Using record-wise source fit/validation split:",
        {
            "mode": mode,
            "fit_records": fit_records,
            "val_records": val_records,
            "fit_samples": len(fit_idx),
            "val_samples": len(val_idx),
        },
    )
    return Subset(dataset, fit_idx), Subset(dataset, val_idx)
