from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset
from src.training.evaluate import predict_model
from src.training.train_dann import load_dann_from_checkpoint
from src.utils.io import ensure_dir, write_json
from src.utils.wandb_logging import init_wandb, log_eval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/macnn_se_dann_best.pt")
    parser.add_argument("--method-name", default="macnn_se_dann")
    parser.add_argument("--dataset", choices=["mitbih", "incart", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    device = device_from_torch()
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = cfg_path(config, "paths", "output_dir").parent / checkpoint if str(checkpoint).startswith("outputs") else Path(config["_base_dir"]) / checkpoint
    model, _ = load_dann_from_checkpoint(checkpoint, device)
    wandb_run = init_wandb(
        config,
        job_type="eval_macnn_dann",
        default_name=f"{args.method_name}_eval",
        extra_config={"checkpoint": str(checkpoint), "dataset": args.dataset},
    )

    datasets = []
    if args.dataset in {"mitbih", "both"}:
        datasets.append(("mitbih_test", cfg_path(config, "data", "source_test")))
    if args.dataset in {"incart", "both"}:
        datasets.append(("incart_after5_heldout", cfg_path(config, "data", "target_test")))
    for name, path in datasets:
        ds = ECGMACNNDataset(path)
        if args.max_samples is not None:
            ds = Subset(ds, list(range(min(int(args.max_samples), len(ds)))))
        loader = DataLoader(ds, batch_size=int(config["training"]["source_batch_size"]), shuffle=False, num_workers=0)
        result = predict_model(model, loader, device, desc=f"{args.method_name} {name}")
        stem = f"{args.method_name}_{name}"
        if args.max_samples is not None:
            stem += f"_max_samples_{args.max_samples}"
        metrics = dict(result["metrics"])
        metrics["dataset"] = name
        metrics["setting"] = args.method_name
        write_json(metrics, output / "metrics" / f"{stem}_metrics.json")
        _write_predictions(output / "predictions" / f"{stem}_predictions.csv", result)
        log_eval_metrics(wandb_run, metrics, prefix=f"eval/{args.method_name}/{name}")
        print(stem, metrics)
    wandb_run.finish()


def _write_predictions(path: Path, result: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "prob_N", "prob_S", "prob_V"])
        for y_true, y_pred, prob in zip(result["y_true"], result["y_pred"], result["probabilities"]):
            writer.writerow([int(y_true), int(y_pred), *[float(x) for x in np.asarray(prob)]])


if __name__ == "__main__":
    main()
