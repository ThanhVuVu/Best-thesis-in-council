from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.label_mapping import CLASS_NAMES, ID_TO_CLASS
from src.training.evaluate import predict_model
from src.utils.io import ensure_dir, write_json
from src.visualization.plot_confusion import plot_confusion_matrix


def evaluate_and_save(model, dataset, device, output_dir: str | Path, dataset_name: str, setting: str) -> dict:
    output = Path(output_dir)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    result = predict_model(model, loader, device, desc=f"eval {dataset_name}")
    metrics = result["metrics"]
    metrics.update({"dataset": dataset_name, "setting": setting})

    metrics_dir = ensure_dir(output / "metrics")
    figures_dir = ensure_dir(output / "figures")
    predictions_dir = ensure_dir(output / "predictions")
    write_json(metrics, metrics_dir / f"{dataset_name}_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        figures_dir / f"confusion_{dataset_name}.png",
        f"{dataset_name} confusion matrix",
    )
    save_predictions(result, predictions_dir / f"{dataset_name}_predictions.csv")
    print(metrics)
    return metrics


def save_predictions(result: dict, path: str | Path) -> None:
    rows = []
    probs = result["probabilities"]
    for i, meta in enumerate(result["metadata"]):
        row = dict(meta)
        row["y_true"] = int(result["y_true"][i])
        row["y_pred"] = int(result["y_pred"][i])
        row["true_class"] = ID_TO_CLASS[int(result["y_true"][i])]
        row["pred_class"] = ID_TO_CLASS[int(result["y_pred"][i])]
        for class_id, class_name in enumerate(CLASS_NAMES):
            row[f"prob_{class_name}"] = float(probs[i, class_id])
        rows.append(row)
    ensure_dir(Path(path).parent)
    pd.DataFrame(rows).to_csv(path, index=False)
