from __future__ import annotations

import argparse
import copy

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train_dann import train_dann
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--phase2-config", default="configs/phase2_dann.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    phase5 = load_phase1_config(args.config)
    config = load_phase1_config(args.phase2_config)
    config["data"]["source_train"] = phase5["catnet_first5"]["source_train"]
    config["data"]["source_test"] = phase5["catnet_first5"]["source_test"]
    config["data"]["target_unlabeled"] = phase5["catnet_first5"]["target_unlabeled"]
    config["data"]["target_test"] = phase5["catnet_first5"]["target_test"]
    config["training"]["checkpoint_prefix"] = phase5["catnet_first5"].get("checkpoint_prefix", "catnet_first5_dann")
    config["dann"]["source_init_checkpoint"] = None
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    set_seed(int(config["seed"]))
    device = device_from_torch()

    full_train = ECGBeatDataset(cfg_path(config, "data", "source_train"))
    fit_records, val_records = mitbih_fit_val_records()
    source_fit = subset_by_records(full_train, fit_records)
    source_val = subset_by_records(full_train, val_records)
    target = ECGBeatDataset(cfg_path(config, "data", "target_unlabeled"))
    if args.max_source_samples is not None:
        source_fit = Subset(source_fit, list(range(min(int(args.max_source_samples), len(source_fit)))))
    if args.max_target_samples is not None:
        target = Subset(target, list(range(min(int(args.max_target_samples), len(target)))))
    if args.max_val_samples is not None:
        source_val = Subset(source_val, list(range(min(int(args.max_val_samples), len(source_val)))))
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_dann(source_fit, source_val, target, copy.deepcopy(config), output, device)
    write_json(summary, output / "metrics" / "catnet_first5_dann_train_summary.json")


if __name__ == "__main__":
    main()
