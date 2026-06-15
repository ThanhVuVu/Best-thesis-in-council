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
from src.training.diagnostics import (
    classwise_predicted_ece,
    confidence_from_probs,
    entropy_from_probs,
    expected_calibration_error,
)
from src.utils.io import ensure_dir, write_json
from src.visualization.diagnostics import plot_confidence_histogram, plot_entropy_by_class, plot_reliability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--dataset", default="configured")
    parser.add_argument("--n-bins", type=int, default=None)
    args = parser.parse_args()

    diag_config, base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    class_names = class_names_from(base_config)
    n_bins = int(args.n_bins or diag_config["analysis"].get("n_bins", 15))
    out = output_dir(diag_config)
    tables_dir = ensure_dir(out / "diagnostics" / "calibration")
    figures_dir = ensure_dir(out / "figures" / "calibration")
    summary = {}

    dataset_names = list(diag_config["analysis"].get("datasets", []))
    for dataset_name, _path in selected_datasets(base_config, args.dataset, configured=dataset_names):
        pred = read_predictions(prediction_path(diag_config, method, dataset_name), class_names)
        ece, bins = expected_calibration_error(pred["probs"], pred["y_true"], n_bins=n_bins)
        classwise = classwise_predicted_ece(pred["probs"], pred["y_true"], class_names, n_bins=n_bins)
        confidence = confidence_from_probs(pred["probs"])
        entropy = entropy_from_probs(pred["probs"])
        correct = pred["y_true"] == pred["y_pred"]
        prefix = f"{method}_{dataset_name}"
        write_csv(tables_dir / f"{prefix}_reliability_bins.csv", bins)
        write_json({"ece": ece, "classwise_predicted_ece": classwise}, tables_dir / f"{prefix}_calibration_summary.json")
        plot_reliability(bins, figures_dir / f"{prefix}_reliability.png", f"{prefix} reliability")
        plot_confidence_histogram(confidence[correct], confidence[~correct], figures_dir / f"{prefix}_confidence_histogram.png", f"{prefix} confidence")
        plot_entropy_by_class(entropy, pred["y_true"], class_names, figures_dir / f"{prefix}_entropy_by_true_class.png", f"{prefix} entropy by class")
        summary[dataset_name] = {"ece": ece, "classwise_predicted_ece": {k: v.get("ece") for k, v in classwise.items()}}
    write_json(summary, tables_dir / f"{method}_calibration_summary.json")
    print(f"calibration outputs written under {tables_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
