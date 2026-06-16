from __future__ import annotations

import argparse
from typing import Any

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, subset_first
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
    source_ds = DAEACDataset(cfg_path(config, "data", "source_train"), input_key=input_key, label_key=label_key, class_names=class_names)
    val_ds = DAEACDataset(cfg_path(config, "data", "source_eval"), input_key=input_key, label_key=label_key, class_names=class_names)
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
