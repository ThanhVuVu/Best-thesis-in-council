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
from src.training.diagnostics import pr_roc_summary, threshold_sweep_rows
from src.utils.io import ensure_dir, write_json
from src.visualization.diagnostics import plot_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--dataset", default="configured")
    args = parser.parse_args()

    diag_config, base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    class_names = class_names_from(base_config)
    focus = list(diag_config["analysis"].get("focus_classes", ["F", "S"]))
    out = output_dir(diag_config)
    tables_dir = ensure_dir(out / "diagnostics" / "pr_roc")
    figures_dir = ensure_dir(out / "figures" / "pr_roc")
    summary = {}

    dataset_names = list(diag_config["analysis"].get("datasets", []))
    for dataset_name, _path in selected_datasets(base_config, args.dataset, configured=dataset_names):
        pred_path = prediction_path(diag_config, method, dataset_name)
        if not pred_path.exists():
            print(f"skipped {dataset_name}: missing predictions {pred_path}")
            continue
        pred = read_predictions(pred_path, class_names)
        dataset_summary = {}
        for class_name in focus:
            cls = class_names.index(class_name)
            scores = pred["probs"][:, cls]
            curves = pr_roc_summary(pred["y_true"], scores, cls)
            sweep = threshold_sweep_rows(pred["y_true"], scores, cls)
            prefix = f"{method}_{dataset_name}_{class_name}"
            write_csv(tables_dir / f"{prefix}_threshold_sweep.csv", sweep)
            write_csv(tables_dir / f"{prefix}_pr_curve.csv", curves["pr_curve"])
            write_csv(tables_dir / f"{prefix}_roc_curve.csv", curves["roc_curve"])
            plot_curve(curves["pr_curve"], "recall", "precision", figures_dir / f"{prefix}_pr_curve.png", f"{prefix} PR", "Recall", "Precision")
            plot_curve(curves["roc_curve"], "fpr", "tpr", figures_dir / f"{prefix}_roc_curve.png", f"{prefix} ROC", "FPR", "TPR")
            best = max(sweep, key=lambda row: row["f1"]) if sweep else None
            dataset_summary[class_name] = {"auprc": curves["auprc"], "auroc": curves["auroc"], "best_threshold_by_f1": best}
        summary[dataset_name] = dataset_summary
    write_json(summary, tables_dir / f"{method}_pr_roc_summary.json")
    print(f"PR/ROC outputs written under {tables_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
