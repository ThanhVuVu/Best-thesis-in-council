from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatRRDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train_dann import train_dann
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_rr_dann.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        config["training"]["lr"] = float(args.lr)
    if args.alpha is not None:
        config["dann"]["alpha"] = float(args.alpha)
    if args.warmup_epochs is not None:
        config["dann"]["warmup_epochs"] = int(args.warmup_epochs)
    set_seed(int(config["seed"]))
    device = device_from_torch()
    print(f"Using device: {device}")

    source_full = ECGBeatRRDataset(cfg_path(config, "data", "source_train"))
    fit_records, val_records = mitbih_fit_val_records()
    source_fit = subset_by_records(source_full, fit_records)
    source_val = subset_by_records(source_full, val_records)
    target = ECGBeatRRDataset(cfg_path(config, "data", "target_unlabeled"))

    if args.max_source_samples is not None:
        source_fit = Subset(source_fit, list(range(min(args.max_source_samples, len(source_fit)))))
    if args.max_target_samples is not None:
        target = Subset(target, list(range(min(args.max_target_samples, len(target)))))
    if args.max_val_samples is not None:
        source_val = Subset(source_val, list(range(min(args.max_val_samples, len(source_val)))))

    print(f"DANN-RR source fit beats: {len(source_fit)}, source val beats: {len(source_val)}, target unlabeled beats: {len(target)}")
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_dann(source_fit, source_val, target, config, output, device)
    write_json(summary, output / "metrics" / "dann_rr_train_summary.json")


if __name__ == "__main__":
    main()
