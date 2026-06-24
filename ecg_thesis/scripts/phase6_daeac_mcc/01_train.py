from __future__ import annotations

import argparse

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACTargetUnlabeledDataset, load_daeac_source_fit_val, subset_first
from src.training.train_daeac_mcc import train_daeac_mcc
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_mcc.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument(
        "--domain-pair",
        choices=["ds1_ds2", "ds1_incart", "ds1_svdb", "mitbih_incart", "mitbih_svdb"],
        default=None,
    )
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    _apply_domain_pair(config, args.domain_pair)
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
    source_ds, val_ds, split_summary = load_daeac_source_fit_val(
        cfg_path(config, "data", "source_train"),
        cfg_path(config, "data", "source_eval"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        full_source_fit=False,
    )
    print(f"DAEAC source fit/validation split: {split_summary}")
    target_ds = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    dev_target_ds = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_val"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    source_ds = subset_first(source_ds, args.max_source_samples)
    val_ds = subset_first(val_ds, args.max_val_samples)
    target_ds = subset_first(target_ds, args.max_target_samples)
    dev_target_ds = subset_first(dev_target_ds, args.max_target_samples)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_daeac_mcc(source_ds, val_ds, target_ds, dev_target_ds, config, output, device)
    write_json(summary, output / "metrics" / f"{config['adaptation']['checkpoint_prefix']}_train_summary.json")


def _apply_domain_pair(config, domain_pair: str | None) -> None:
    if domain_pair is None:
        return
    specs = {
        "ds1_ds2": ("ds1", "ds2", "ds1"),
        "ds1_incart": ("ds1", "incart", "ds1"),
        "ds1_svdb": ("ds1", "svdb", "ds1"),
        "mitbih_incart": ("mitbih", "incart", "mitbih"),
        "mitbih_svdb": ("mitbih", "svdb", "mitbih"),
    }
    source, target, source_kind = specs[domain_pair]
    root = "data/processed/phase6_daeac_record_splits"
    config["domain_pair"] = domain_pair
    config["data"].update(
        source_train=f"{root}/{source}_train.npz",
        source_eval=f"{root}/{source}_val.npz",
        target_unlabeled=f"{root}/{target}_train.npz",
        target_val=f"{root}/{target}_val.npz",
        target_test=f"{root}/{target}_test.npz",
        target_protocol="record_60_20_20",
    )
    config["paths"]["output_dir"] = f"outputs/phase6_daeac_mcc_{domain_pair}"
    config["adaptation"]["checkpoint_prefix"] = f"daeac_mcc_{domain_pair}"
    config["adaptation"]["source_kind"] = source_kind


if __name__ == "__main__":
    main()
