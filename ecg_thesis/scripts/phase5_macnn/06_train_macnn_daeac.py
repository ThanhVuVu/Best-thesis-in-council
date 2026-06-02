from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train_macnn import train_macnn_daeac
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--source-loss", choices=["weighted_ce", "focal", "dynamic_focal"], default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--threshold-n", type=float, default=None)
    parser.add_argument("--threshold-s", type=float, default=None)
    parser.add_argument("--threshold-v", type=float, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--align-compact", action="store_true")
    parser.add_argument("--no-separation", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    if args.epochs is not None:
        config["daeac"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        config["daeac"]["lr"] = float(args.lr)
    if args.source_loss is not None:
        config["daeac"]["source_loss"] = str(args.source_loss)
    if args.focal_gamma is not None:
        config["daeac"]["focal_gamma"] = float(args.focal_gamma)
    if args.beta1 is not None:
        config["daeac"]["beta1"] = float(args.beta1)
    if args.beta2 is not None:
        config["daeac"]["beta2"] = float(args.beta2)
    if args.threshold_n is not None:
        config["daeac"]["pseudo_thresholds"]["N"] = float(args.threshold_n)
    if args.threshold_s is not None:
        config["daeac"]["pseudo_thresholds"]["S"] = float(args.threshold_s)
    if args.threshold_v is not None:
        config["daeac"]["pseudo_thresholds"]["V"] = float(args.threshold_v)
    if args.checkpoint_prefix is not None:
        config["daeac"]["checkpoint_prefix"] = str(args.checkpoint_prefix)
    if args.init_checkpoint is not None:
        config["daeac"]["init_checkpoint"] = str(args.init_checkpoint)
    if args.align_only:
        config["daeac"]["use_align"] = True
        config["daeac"]["use_compact"] = False
        config["daeac"]["use_separation"] = False
        config["daeac"]["checkpoint_prefix"] = "macnn_se_daeac_align_only"
    if args.align_compact:
        config["daeac"]["use_align"] = True
        config["daeac"]["use_compact"] = True
        config["daeac"]["use_separation"] = False
        config["daeac"]["checkpoint_prefix"] = "macnn_se_daeac_align_compact"
    if args.no_separation:
        config["daeac"]["use_separation"] = False
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
    summary = train_macnn_daeac(source_fit, source_val, target, config, output, device)
    name = config["daeac"]["checkpoint_prefix"]
    write_json(summary, output / "metrics" / f"{name}_train_summary.json")


if __name__ == "__main__":
    main()
