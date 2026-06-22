from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, split_daeac_source_fit_val, subset_first
from src.training.train_daeac_paper import evaluate_daeac_model, load_daeac_checkpoint
from src.utils.io import ensure_dir, write_json
from src.utils.wandb_logging import init_wandb, log_eval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--dataset", default="all", choices=["source", "target", "both", "external", "all", "incart", "svdb"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    add_wandb_args(parser)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    method = args.method_name or str(config["adaptation"]["checkpoint_prefix"])
    checkpoint_kind = "best" if Path(args.checkpoint).stem.endswith("_best") else "latest"
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    device = device_from_torch()
    model = load_daeac_checkpoint(args.checkpoint, config, device)
    class_names = list(config["data"]["class_names"])
    run = init_wandb(config, job_type="eval_daeac_prototype_bank", default_name=f"{method}_{checkpoint_kind}_eval")
    for dataset_name, path in _datasets(config, args.dataset):
        dataset = DAEACDataset(
            path,
            input_key=str(config["data"].get("input_key", "auto")),
            label_key=str(config["data"].get("label_key", "y")),
            class_names=class_names,
        )
        if dataset_name == "source_val":
            _, dataset, _ = split_daeac_source_fit_val(dataset)
        dataset = subset_first(dataset, args.max_samples)
        result = evaluate_daeac_model(
            model,
            DataLoader(dataset, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False),
            device,
            class_names,
        )
        stem = f"{method}_{checkpoint_kind}_{dataset_name}"
        metrics = dict(result["metrics"])
        metrics.update(
            {
                "dataset": dataset_name,
                "setting": method,
                "checkpoint": str(args.checkpoint),
                "checkpoint_kind": checkpoint_kind,
                "used_for_checkpoint_selection": dataset_name == "source_val",
            }
        )
        write_json(metrics, output / "metrics" / f"{stem}_metrics.json")
        _write_predictions(output / "predictions" / f"{stem}_predictions.csv", result, class_names)
        _write_confusion(output / "metrics" / f"{stem}_confusion_matrix.csv", metrics["confusion_matrix"], class_names)
        log_eval_metrics(run, metrics, prefix=f"eval/{method}/{checkpoint_kind}/{dataset_name}")
        print(stem, metrics["paper_metrics"])
    run.finish()


def _datasets(config, selection):
    selected = []
    if selection in {"source", "both", "all"}:
        selected.append(("source_val", cfg_path(config, "data", "source_eval")))
    if selection in {"target", "both", "all"}:
        selected.append(("target_full", cfg_path(config, "data", "target_test")))
    external = dict(config["data"].get("external_targets", {}))
    names = external.keys() if selection in {"external", "all"} else [selection] if selection in external else []
    for name in names:
        path = cfg_path(config, "data", "external_targets", name)
        if path.exists():
            selected.append((name, path))
        else:
            print(f"Skipping missing optional external dataset {name}: {path}")
    return selected


def _write_predictions(path, result, class_names):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y_true", "y_pred", "true_class", "pred_class", *[f"prob_{name}" for name in class_names]])
        for true, pred, probability in zip(result["y_true"], result["y_pred"], result["probabilities"]):
            writer.writerow(
                [int(true), int(pred), class_names[int(true)], class_names[int(pred)], *[float(value) for value in np.asarray(probability)]]
            )


def _write_confusion(path, matrix, class_names):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])


if __name__ == "__main__":
    main()
