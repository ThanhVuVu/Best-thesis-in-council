from __future__ import annotations

import argparse

import numpy as np

from diagnostics_common import (
    class_names_from,
    load_diagnostics_config,
    method_name,
    output_dir,
    prediction_path,
    read_predictions,
    selected_datasets,
    write_csv,
)
from src.training.diagnostics import error_pair_rows, per_record_rows
from src.training.metrics import classification_metrics
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--dataset", default="configured")
    args = parser.parse_args()

    diag_config, base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    class_names = class_names_from(base_config)
    tables_dir = ensure_dir(output_dir(diag_config) / "diagnostics" / "tables")
    summary = {}

    dataset_names = list(diag_config["analysis"].get("datasets", []))
    for dataset_name, _path in selected_datasets(base_config, args.dataset, configured=dataset_names):
        pred_path = prediction_path(diag_config, method, dataset_name)
        if not pred_path.exists():
            print(f"skipped {dataset_name}: missing predictions {pred_path}")
            continue
        pred = read_predictions(pred_path, class_names)
        metrics = classification_metrics(pred["y_true"], pred["y_pred"], class_names)
        cm = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        prefix = f"{method}_{dataset_name}"
        write_csv(tables_dir / f"{prefix}_error_pairs.csv", error_pair_rows(pred["y_true"], pred["y_pred"], pred["probs"], class_names))
        write_csv(tables_dir / f"{prefix}_per_record.csv", per_record_rows(pred["records"], pred["y_true"], pred["y_pred"], pred["probs"], class_names))
        write_csv(tables_dir / f"{prefix}_confusion_normalized.csv", _matrix_rows(normalized, class_names))
        write_csv(tables_dir / f"{prefix}_per_class.csv", _per_class_rows(metrics, class_names))
        summary[dataset_name] = {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "top_error_pairs": error_pair_rows(pred["y_true"], pred["y_pred"], pred["probs"], class_names)[:8],
        }
    write_json(summary, tables_dir / f"{method}_failure_summary.json")
    print(f"failure tables written under {tables_dir}")


def _matrix_rows(matrix: np.ndarray, class_names: list[str]) -> list[dict[str, float | str]]:
    rows = []
    for idx, name in enumerate(class_names):
        row: dict[str, float | str] = {"true_class": name}
        for pred_idx, pred_name in enumerate(class_names):
            row[f"pred_{pred_name}"] = float(matrix[idx, pred_idx])
        rows.append(row)
    return rows


def _per_class_rows(metrics: dict, class_names: list[str]) -> list[dict[str, float | str | int]]:
    rows = []
    for name in class_names:
        values = metrics["per_class"][name]
        rows.append(
            {
                "class": name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1"]),
                "support": int(values["support"]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
