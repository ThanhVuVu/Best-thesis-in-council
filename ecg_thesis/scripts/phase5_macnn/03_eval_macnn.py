from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset
from src.training.train_macnn import evaluate_macnn_model, load_macnn_checkpoint
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-name", default="macnn_se")
    parser.add_argument("--dataset", choices=["mitbih", "incart", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    device = device_from_torch()
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    model = load_macnn_checkpoint(args.checkpoint, config, device)

    datasets = []
    if args.dataset in {"mitbih", "both"}:
        datasets.append(("mitbih_test", cfg_path(config, "data", "source_test")))
    if args.dataset in {"incart", "both"}:
        datasets.append(("incart_after5_heldout", cfg_path(config, "data", "target_test")))

    for name, path in datasets:
        ds = ECGMACNNDataset(path)
        if args.max_samples is not None:
            ds = Subset(ds, list(range(min(int(args.max_samples), len(ds)))))
        loader = DataLoader(ds, batch_size=int(config["source_only"]["batch_size"]), shuffle=False, num_workers=0)
        result = evaluate_macnn_model(model, loader, device, config["data"]["class_names"], desc=f"{args.method_name} {name}")
        stem = f"{args.method_name}_{name}"
        if args.max_samples is not None:
            stem += f"_max_samples_{args.max_samples}"
        metrics = dict(result["metrics"])
        metrics["dataset"] = name
        metrics["setting"] = args.method_name
        write_json(metrics, output / "metrics" / f"{stem}_metrics.json")
        _write_predictions(output / "predictions" / f"{stem}_predictions.csv", result)
        print(stem, metrics)


def _write_predictions(path: Path, result: dict) -> None:
    ensure_dir(path.parent)
    probs = result["probabilities"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "prob_N", "prob_S", "prob_V"])
        for y_true, y_pred, prob in zip(result["y_true"], result["y_pred"], probs):
            writer.writerow([int(y_true), int(y_pred), *[float(x) for x in np.asarray(prob)]])


if __name__ == "__main__":
    main()
