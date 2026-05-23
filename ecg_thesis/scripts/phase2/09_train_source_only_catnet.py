from __future__ import annotations

import argparse
import copy

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.training.train import train_source_only
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_dann.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-fit-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    run_config = copy.deepcopy(config)
    run_config["training"] = copy.deepcopy(config["source_only"])
    run_config["training"]["model_kwargs"] = _model_kwargs(config["model"])
    if args.epochs is not None:
        run_config["training"]["epochs"] = int(args.epochs)

    device = device_from_torch()
    print(f"Using device: {device}")

    source_train = cfg_path(config, "data", "source_train")
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    full_train = ECGBeatDataset(source_train)
    fit_records, val_records = mitbih_fit_val_records()
    fit_ds = subset_by_records(full_train, fit_records)
    val_ds = subset_by_records(full_train, val_records)
    if args.max_fit_samples is not None:
        fit_ds = Subset(fit_ds, list(range(min(int(args.max_fit_samples), len(fit_ds)))))
    if args.max_val_samples is not None:
        val_ds = Subset(val_ds, list(range(min(int(args.max_val_samples), len(val_ds)))))
    print(f"Source-only CATNet fit beats: {len(fit_ds)}, validation beats: {len(val_ds)}")

    summary = train_source_only(fit_ds, val_ds, run_config, output, device)
    write_json(summary, output / "metrics" / "source_only_catnet_train_summary.json")


def _model_kwargs(model_cfg: dict) -> dict:
    allowed = {
        "d_model",
        "num_heads",
        "dff",
        "num_transformer_layers",
        "attention_reduction",
        "dropout",
    }
    return {key: model_cfg[key] for key in allowed if key in model_cfg}


if __name__ == "__main__":
    main()
