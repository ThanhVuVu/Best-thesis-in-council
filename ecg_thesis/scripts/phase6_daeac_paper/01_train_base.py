from __future__ import annotations

import argparse

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, subset_first
from src.training.train_daeac_paper import train_daeac_base
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_paper.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.checkpoint_prefix is not None:
        config["training"]["checkpoint_prefix"] = str(args.checkpoint_prefix)
    set_seed(int(config["seed"]))
    device = device_from_torch()
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    class_names = list(config["data"]["class_names"])
    train_ds = DAEACDataset(cfg_path(config, "data", "source_train"), input_key=input_key, label_key=label_key, class_names=class_names)
    val_ds = DAEACDataset(cfg_path(config, "data", "source_eval"), input_key=input_key, label_key=label_key, class_names=class_names)
    train_ds = subset_first(train_ds, args.max_source_samples)
    val_ds = subset_first(val_ds, args.max_val_samples)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_daeac_base(train_ds, val_ds, config, output, device)
    write_json(summary, output / "metrics" / "daeac_base_train_summary.json")


if __name__ == "__main__":
    main()
