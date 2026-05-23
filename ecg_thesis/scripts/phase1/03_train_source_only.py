from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train import train_source_only
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    device = device_from_torch()
    print(f"Using device: {device}")

    processed = cfg_path(config, "paths", "processed_dir")
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    full_train = ECGBeatDataset(processed / "mitbih_train.npz")
    fit_records, val_records = mitbih_fit_val_records()
    fit_ds = subset_by_records(full_train, fit_records)
    val_ds = subset_by_records(full_train, val_records)
    print(f"Fit beats: {len(fit_ds)}, validation beats: {len(val_ds)}")

    summary = train_source_only(fit_ds, val_ds, config, output, device)
    write_json(summary, output / "metrics" / "train_summary.json")


if __name__ == "__main__":
    main()
