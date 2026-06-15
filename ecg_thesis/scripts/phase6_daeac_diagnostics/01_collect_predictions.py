from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import device_from_torch
from diagnostics_common import (
    checkpoint_path,
    class_names_from,
    embedding_path,
    load_diagnostics_config,
    method_name,
    output_dir,
    prediction_path,
    selected_datasets,
)
from src.data.daeac_dataset import DAEACDataset, subset_first
from src.training.diagnostics import confidence_from_probs, entropy_from_probs
from src.training.train_daeac_paper import daeac_metrics, load_daeac_checkpoint
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--dataset", default="configured")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()

    diag_config, base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    class_names = class_names_from(base_config)
    output = output_dir(diag_config)
    device = device_from_torch()
    model = load_daeac_checkpoint(checkpoint_path(diag_config, args.checkpoint), base_config, device)
    model.eval()

    dataset_names = list(diag_config["analysis"].get("datasets", []))
    for dataset_name, path in selected_datasets(base_config, args.dataset, configured=dataset_names):
        ds = DAEACDataset(
            path,
            input_key=str(base_config["data"].get("input_key", "auto")),
            label_key=str(base_config["data"].get("label_key", "y")),
            class_names=class_names,
        )
        n = len(ds) if args.max_samples is None else min(int(args.max_samples), len(ds))
        eval_ds = subset_first(ds, args.max_samples)
        loader = DataLoader(eval_ds, batch_size=int(base_config["evaluation"]["batch_size"]), shuffle=False, num_workers=0)
        result = _predict(model, loader, device)
        metrics = daeac_metrics(result["y_true"], result["y_pred"], class_names)
        metrics.update({"dataset": dataset_name, "setting": method, "checkpoint": str(checkpoint_path(diag_config, args.checkpoint))})
        write_json(metrics, output / "metrics" / f"{method}_{dataset_name}_metrics.json")
        _write_confusion(output / "metrics" / f"{method}_{dataset_name}_confusion_matrix.csv", metrics["confusion_matrix"], class_names)
        _write_predictions(prediction_path(diag_config, method, dataset_name), result, ds, n, class_names)
        if args.save_embeddings:
            _write_embeddings(embedding_path(diag_config, method, dataset_name), result, ds, n)
        print(f"{method}_{dataset_name}", metrics["paper_metrics"])


def _predict(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    features, probs, y_true = [], [], []
    with torch.no_grad():
        for x, y in loader:
            z, _logits, p = model(x.to(device), return_logits=True)
            features.append(z.detach().cpu().numpy())
            probs.append(p.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
    probabilities = np.concatenate(probs, axis=0)
    labels = np.concatenate(y_true, axis=0).astype(np.int64)
    return {
        "features": np.concatenate(features, axis=0),
        "probabilities": probabilities,
        "y_true": labels,
        "y_pred": probabilities.argmax(axis=1).astype(np.int64),
    }


def _write_predictions(path, result: dict[str, np.ndarray], dataset: DAEACDataset, n: int, class_names: list[str]) -> None:
    ensure_dir(path.parent)
    confidence = confidence_from_probs(result["probabilities"])
    entropy = entropy_from_probs(result["probabilities"])
    metadata_rows = [dataset.metadata(idx) for idx in range(n)]
    metadata_fields = ["record", "record_id", "symbol", "sample", "r_peak_sample", "r_peak_time_sec", "fs", "domain", "lead_index", "lead_name"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "y_true",
                "y_pred",
                "true_class",
                "pred_class",
                "confidence",
                "entropy",
                *[f"prob_{name}" for name in class_names],
                *metadata_fields,
            ]
        )
        for idx in range(n):
            y_true = int(result["y_true"][idx])
            y_pred = int(result["y_pred"][idx])
            meta = metadata_rows[idx]
            writer.writerow(
                [
                    idx,
                    y_true,
                    y_pred,
                    class_names[y_true],
                    class_names[y_pred],
                    float(confidence[idx]),
                    float(entropy[idx]),
                    *[float(v) for v in result["probabilities"][idx]],
                    *[meta.get(field, "") for field in metadata_fields],
                ]
            )


def _write_embeddings(path, result: dict[str, np.ndarray], dataset: DAEACDataset, n: int) -> None:
    ensure_dir(path.parent)
    metadata_rows = [dataset.metadata(idx) for idx in range(n)]
    np.savez_compressed(
        path,
        features=result["features"].astype(np.float32),
        probabilities=result["probabilities"].astype(np.float32),
        y_true=result["y_true"].astype(np.int64),
        y_pred=result["y_pred"].astype(np.int64),
        record=np.asarray([row.get("record", row.get("record_id", "")) for row in metadata_rows], dtype=object),
        symbol=np.asarray([row.get("symbol", "") for row in metadata_rows], dtype=object),
        sample=np.asarray([row.get("sample", row.get("r_peak_sample", -1)) for row in metadata_rows], dtype=object),
    )


def _write_confusion(path, matrix: list[list[int]], class_names: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])


if __name__ == "__main__":
    main()
