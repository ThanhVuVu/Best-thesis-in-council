from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train_dann import train_dann
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--source-loss", choices=["weighted_ce", "focal", "dynamic_focal"], default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.source_loss is not None:
        config["training"]["source_loss"] = str(args.source_loss)
    if args.focal_gamma is not None:
        config["training"]["focal_gamma"] = float(args.focal_gamma)
    if args.checkpoint_prefix is not None:
        config["training"]["checkpoint_prefix"] = str(args.checkpoint_prefix)
    set_seed(int(config["seed"]))
    device = device_from_torch()

    full_train = ECGMACNNDataset(cfg_path(config, "data", "source_train"))
    fit_records, val_records = mitbih_fit_val_records()
    source_fit = subset_by_records(full_train, fit_records)
    source_val = subset_by_records(full_train, val_records)
    target = ECGMACNNDataset(cfg_path(config, "data", "target_unlabeled"))
    if args.max_source_samples is not None:
        source_fit = Subset(source_fit, list(range(min(int(args.max_source_samples), len(source_fit)))))
    if args.max_target_samples is not None:
        target = Subset(target, list(range(min(int(args.max_target_samples), len(target)))))
    if args.max_val_samples is not None:
        source_val = Subset(source_val, list(range(min(int(args.max_val_samples), len(source_val)))))

    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_dann(source_fit, source_val, target, config, output, device)
    write_json(summary, output / "metrics" / "macnn_se_dann_train_summary.json")


if __name__ == "__main__":
    main()
