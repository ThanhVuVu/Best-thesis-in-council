from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, subset_first
from src.training.train_daeac_paper import evaluate_daeac_model, load_daeac_checkpoint
from src.utils.io import ensure_dir, write_json
from src.utils.wandb_logging import init_wandb, log_eval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_paper.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-name", default="daeac")
    parser.add_argument("--dataset", choices=["source", "target", "both"], default="target")
    parser.add_argument("--max-samples", type=int, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    device = device_from_torch()
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    model = load_daeac_checkpoint(args.checkpoint, config, device)
    wandb_run = init_wandb(
        config,
        job_type="eval_daeac_paper",
        default_name=f"{args.method_name}_eval",
        extra_config={"checkpoint": str(args.checkpoint), "dataset": args.dataset},
    )
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    class_names = list(config["data"]["class_names"])
    datasets = []
    if args.dataset in {"source", "both"}:
        datasets.append(("source_eval", cfg_path(config, "data", "source_eval")))
    if args.dataset in {"target", "both"}:
        datasets.append(("target_test", cfg_path(config, "data", "target_test")))
    for name, path in datasets:
        ds = DAEACDataset(path, input_key=input_key, label_key=label_key, class_names=class_names)
        ds = subset_first(ds, args.max_samples)
        loader = DataLoader(ds, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False, num_workers=0)
        result = evaluate_daeac_model(model, loader, device, class_names)
        stem = f"{args.method_name}_{name}"
        if args.max_samples is not None:
            stem += f"_max_samples_{args.max_samples}"
        metrics = dict(result["metrics"])
        metrics["dataset"] = name
        metrics["setting"] = args.method_name
        metrics["checkpoint"] = str(args.checkpoint)
        write_json(metrics, output / "metrics" / f"{stem}_metrics.json")
        _write_predictions(output / "predictions" / f"{stem}_predictions.csv", result, class_names)
        _write_confusion(output / "metrics" / f"{stem}_confusion_matrix.csv", metrics["confusion_matrix"], class_names)
        log_eval_metrics(wandb_run, metrics, prefix=f"eval/{args.method_name}/{name}")
        print(stem, metrics["paper_metrics"])
    wandb_run.finish()


def _write_predictions(path: Path, result: dict, class_names: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "true_class", "pred_class", *[f"prob_{name}" for name in class_names]])
        for y_true, y_pred, prob in zip(result["y_true"], result["y_pred"], result["probabilities"]):
            writer.writerow([int(y_true), int(y_pred), class_names[int(y_true)], class_names[int(y_pred)], *[float(v) for v in np.asarray(prob)]])


def _write_confusion(path: Path, matrix: list[list[int]], class_names: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])


if __name__ == "__main__":
    main()
