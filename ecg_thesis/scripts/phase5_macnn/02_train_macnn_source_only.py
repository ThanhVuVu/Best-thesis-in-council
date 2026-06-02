from __future__ import annotations

import argparse
import copy

from torch.utils.data import Subset

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train_macnn import train_macnn_source_only
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--source-loss", choices=["weighted_ce", "focal", "dynamic_focal"], default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--max-fit-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    if args.epochs is not None:
        config["source_only"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        config["source_only"]["lr"] = float(args.lr)
    if args.source_loss is not None:
        config["source_only"]["source_loss"] = str(args.source_loss)
    if args.focal_gamma is not None:
        config["source_only"]["focal_gamma"] = float(args.focal_gamma)
    if args.checkpoint_prefix is not None:
        config["source_only"]["checkpoint_prefix"] = str(args.checkpoint_prefix)
    set_seed(int(config["seed"]))
    device = device_from_torch()

    full_train = ECGMACNNDataset(cfg_path(config, "data", "source_train"))
    fit_records, val_records = mitbih_fit_val_records()
    fit_ds = subset_by_records(full_train, fit_records)
    val_ds = subset_by_records(full_train, val_records)
    if args.max_fit_samples is not None:
        fit_ds = Subset(fit_ds, list(range(min(int(args.max_fit_samples), len(fit_ds)))))
    if args.max_val_samples is not None:
        val_ds = Subset(val_ds, list(range(min(int(args.max_val_samples), len(val_ds)))))

    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_macnn_source_only(fit_ds, val_ds, copy.deepcopy(config), output, device)
    write_json(summary, output / "metrics" / "macnn_se_source_only_train_summary.json")


if __name__ == "__main__":
    main()
