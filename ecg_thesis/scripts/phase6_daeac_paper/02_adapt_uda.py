from __future__ import annotations

import argparse

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, subset_first
from src.training.train_daeac_paper import adapt_daeac
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_paper.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--domain-pair", choices=["ds1_ds2", "ds1_incart", "ds1_svdb", "mitbih_incart", "mitbih_svdb"])
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
    source_ds = DAEACDataset(
        cfg_path(config, "data", "source_train"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    source_val_ds = DAEACDataset(cfg_path(config, "data", "source_eval"), input_key=input_key, label_key=label_key, class_names=class_names)
    target_ds = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    target_val_ds = DAEACTargetUnlabeledDataset(cfg_path(config, "data", "target_val"), input_key=input_key, label_key=label_key, class_names=class_names)
    source_ds = subset_first(source_ds, args.max_source_samples)
    target_ds = subset_first(target_ds, args.max_target_samples)
    source_val_ds = subset_first(source_val_ds, args.max_val_samples)
    target_val_ds = subset_first(target_val_ds, args.max_val_samples)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = adapt_daeac(source_ds, target_ds, config, output, device, source_val_dataset=source_val_ds, target_val_dataset=target_val_ds)
    write_json(summary, output / "metrics" / "daeac_uda_train_summary.json")


def _apply_domain_pair(config: dict, pair: str | None) -> None:
    if pair is None:
        return
    source, target, checkpoint = {
        "ds1_ds2": ("ds1", "ds2", "daeac_base_best.pt"),
        "ds1_incart": ("ds1", "incart", "daeac_base_best.pt"),
        "ds1_svdb": ("ds1", "svdb", "daeac_base_best.pt"),
        "mitbih_incart": ("mitbih", "incart", "daeac_base_mitbih_best.pt"),
        "mitbih_svdb": ("mitbih", "svdb", "daeac_base_mitbih_best.pt"),
    }[pair]
    root = "data/processed/phase6_daeac_record_splits"
    config["domain_pair"] = pair
    config["data"].update(source_train=f"{root}/{source}_train.npz", source_eval=f"{root}/{source}_val.npz",
        target_unlabeled=f"{root}/{target}_train.npz", target_val=f"{root}/{target}_val.npz", target_test=f"{root}/{target}_test.npz")
    config["paths"]["output_dir"] = f"outputs/phase6_daeac_paper_{pair}"
    config["adaptation"]["checkpoint_prefix"] = f"daeac_paper_{pair}"
    if not config["adaptation"].get("init_checkpoint"):
        config["adaptation"]["init_checkpoint"] = checkpoint


if __name__ == "__main__":
    main()
