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
    parser.add_argument("--config", default="configs/phase2_clef_source.yaml")
    parser.add_argument("--mode", choices=["clef_frozen", "clef_finetune"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-fit-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--clef-checkpoint", default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    _apply_mode(config, args.mode)
    if args.clef_checkpoint is not None:
        config["model"]["clef_checkpoint_path"] = args.clef_checkpoint
    set_seed(int(config["seed"]))

    run_config = copy.deepcopy(config)
    run_config["training"] = copy.deepcopy(config["source_only"])
    run_config["training"]["model_kwargs"] = _model_kwargs(config["model"])
    if args.epochs is not None:
        run_config["training"]["epochs"] = int(args.epochs)

    device = device_from_torch()
    print(f"Using device: {device}")
    full_train = ECGBeatDataset(cfg_path(config, "data", "source_train"))
    fit_records, val_records = mitbih_fit_val_records()
    fit_ds = subset_by_records(full_train, fit_records)
    val_ds = subset_by_records(full_train, val_records)
    if args.max_fit_samples is not None:
        fit_ds = Subset(fit_ds, list(range(min(int(args.max_fit_samples), len(fit_ds)))))
    if args.max_val_samples is not None:
        val_ds = Subset(val_ds, list(range(min(int(args.max_val_samples), len(val_ds)))))

    print(f"Phase 2 CLEF {args.mode} fit beats: {len(fit_ds)}, validation beats: {len(val_ds)}")
    print(
        "CLEF run config:",
        {
            "mode": args.mode,
            "model_size": config["model"]["model_size"],
            "freeze_encoder": config["model"]["freeze_encoder"],
            "checkpoint": config["model"]["clef_checkpoint_path"],
            "head_lr": run_config["training"]["lr"],
            "encoder_lr": config["model"].get("encoder_lr"),
        },
    )
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_source_only(fit_ds, val_ds, run_config, output, device)
    write_json(summary, output / "metrics" / f"{args.mode}_train_summary.json")


def _apply_mode(config: dict, mode: str) -> None:
    mode_cfg = dict(config["modes"][mode])
    config["model"]["freeze_encoder"] = bool(mode_cfg["freeze_encoder"])
    config["model"]["encoder_lr"] = float(mode_cfg["encoder_lr"])
    config["source_only"]["checkpoint_prefix"] = str(mode_cfg["checkpoint_prefix"])
    config["source_only"]["lr"] = float(mode_cfg["lr"])


def _model_kwargs(model_cfg: dict) -> dict:
    allowed = {
        "model_size",
        "clef_checkpoint_path",
        "in_channels",
        "freeze_encoder",
        "head_hidden_dim",
        "dropout",
        "encoder_lr",
    }
    return {key: model_cfg[key] for key in allowed if key in model_cfg}


if __name__ == "__main__":
    main()
