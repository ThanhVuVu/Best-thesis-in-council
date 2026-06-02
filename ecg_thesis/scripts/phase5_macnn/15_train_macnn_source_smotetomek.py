from __future__ import annotations

import argparse
import copy
from pathlib import Path

from torch.utils.data import Subset

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train_macnn import train_macnn_source_only
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_smotetomek.yaml")
    parser.add_argument("--resampled-train", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--source-loss", choices=["weighted_ce", "focal", "dynamic_focal"], default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--use-class-weights", action="store_true")
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
    config["source_only"]["use_class_weights"] = bool(args.use_class_weights)
    set_seed(int(config["seed"]))
    device = device_from_torch()

    resampled_path = args.resampled_train or config.get("smotetomek", {}).get(
        "source_train",
        "data/processed/phase5_macnn/mitbih_fit_macnn_smotetomek_cap_n12000_s4000_v6000.npz",
    )
    resampled_path = _resolve(resampled_path, config)
    fit_ds = ECGMACNNDataset(resampled_path)

    full_train = ECGMACNNDataset(cfg_path(config, "data", "source_train"))
    _fit_records, val_records = mitbih_fit_val_records()
    val_ds = subset_by_records(full_train, val_records)
    if args.max_fit_samples is not None:
        fit_ds = Subset(fit_ds, list(range(min(int(args.max_fit_samples), len(fit_ds)))))
    if args.max_val_samples is not None:
        val_ds = Subset(val_ds, list(range(min(int(args.max_val_samples), len(val_ds)))))

    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_macnn_source_only(fit_ds, val_ds, copy.deepcopy(config), output, device)
    prefix = config["source_only"]["checkpoint_prefix"]
    summary["resampled_train"] = str(resampled_path)
    summary["validation_source"] = str(cfg_path(config, "data", "source_train"))
    summary["validation_records"] = val_records
    summary["no_target_labels_used"] = True
    write_json(summary, output / "metrics" / f"{prefix}_train_summary.json")


def _resolve(value: str | Path, config: dict) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path


if __name__ == "__main__":
    main()
