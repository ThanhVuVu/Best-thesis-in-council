from __future__ import annotations

import argparse

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACTargetUnlabeledDataset, load_daeac_source_fit_val, subset_first
from src.training.train_daeac_hybrid_mkmmd_mcc import train_daeac_hybrid_mkmmd_mcc
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_hybrid_mkmmd_mcc.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    if args.epochs is not None:
        config["adaptation"]["epochs"] = int(args.epochs)
    if args.init_checkpoint is not None:
        config["adaptation"]["init_checkpoint"] = str(args.init_checkpoint)
    if args.checkpoint_prefix is not None:
        config["adaptation"]["checkpoint_prefix"] = str(args.checkpoint_prefix)

    set_seed(int(config["seed"]))
    device = device_from_torch()
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    class_names = list(config["data"]["class_names"])
    rr_mode = str(config["data"].get("rr_mode", "real"))
    rr_features_key = str(config["data"].get("rr_features_key", "rr_features"))
    return_rr_features = bool(config["data"].get("return_rr_features", False))
    morphology_only = bool(config["data"].get("morphology_only", False))
    source_ds, val_ds, split_summary = load_daeac_source_fit_val(
        cfg_path(config, "data", "source_train"),
        cfg_path(config, "data", "source_eval"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        rr_mode=rr_mode,
        rr_features_key=rr_features_key,
        return_rr_features=return_rr_features,
        morphology_only=morphology_only,
    )
    print(f"DAEAC source fit/validation split: {split_summary}")
    target_ds = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        rr_mode=rr_mode,
        rr_features_key=rr_features_key,
        return_rr_features=return_rr_features,
        morphology_only=morphology_only,
    )
    source_ds = subset_first(source_ds, args.max_source_samples)
    val_ds = subset_first(val_ds, args.max_val_samples)
    target_ds = subset_first(target_ds, args.max_target_samples)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_daeac_hybrid_mkmmd_mcc(source_ds, val_ds, target_ds, config, output, device)
    write_json(summary, output / "metrics" / f"{config['adaptation']['checkpoint_prefix']}_train_summary.json")


if __name__ == "__main__":
    main()
