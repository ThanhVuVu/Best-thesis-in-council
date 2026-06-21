from __future__ import annotations

import argparse
from pathlib import Path

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACTargetUnlabeledDataset, load_daeac_source_fit_val, subset_first
from src.training.train_daeac_hybrid_ablation import train_daeac_hybrid_ablation
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    if args.epochs is not None:
        config["adaptation"]["epochs"] = args.epochs
    if args.init_checkpoint:
        config["adaptation"]["init_checkpoint"] = str(Path(args.init_checkpoint).resolve())
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    if args.checkpoint_prefix:
        config["adaptation"]["checkpoint_prefix"] = args.checkpoint_prefix

    set_seed(int(config["seed"]))
    device = device_from_torch()
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    source_ds, val_ds, split_summary = load_daeac_source_fit_val(
        cfg_path(config, "data", "source_train"),
        cfg_path(config, "data", "source_eval"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
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
    write_json(split_summary, output / "diagnostics" / "source_fit_val_split.json")
    summary = train_daeac_hybrid_ablation(source_ds, val_ds, target_ds, config, output, device)
    print(summary)


if __name__ == "__main__":
    main()
