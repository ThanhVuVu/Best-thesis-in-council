from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diagnostics_common import (
    class_names_from,
    load_diagnostics_config,
    method_name,
    output_dir,
    prediction_path,
    read_predictions,
    selected_datasets,
)
from src.data.daeac_dataset import DAEACDataset
from src.visualization.diagnostics import plot_morphology_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--dataset", default="configured")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    diag_config, base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    class_names = class_names_from(base_config)
    top_k = int(args.top_k or diag_config["analysis"].get("morphology_top_k", 24))
    pairs = [tuple(pair) for pair in diag_config["analysis"].get("error_pairs", [])]
    figures_dir = output_dir(diag_config) / "figures" / "morphology"

    dataset_names = list(diag_config["analysis"].get("datasets", []))
    for dataset_name, data_path in selected_datasets(base_config, args.dataset, configured=dataset_names):
        dataset = DAEACDataset(
            data_path,
            input_key=str(base_config["data"].get("input_key", "auto")),
            label_key=str(base_config["data"].get("label_key", "y")),
            class_names=class_names,
        )
        pred = read_predictions(prediction_path(diag_config, method, dataset_name), class_names)
        rows = pred["rows"]
        for true_name, pred_name in pairs:
            selected = [
                row
                for row in rows
                if row.get("true_class") == true_name and row.get("pred_class") == pred_name
            ]
            selected = sorted(selected, key=lambda row: float(row.get("confidence", 0.0)), reverse=True)[:top_k]
            if not selected:
                continue
            indices = [int(row["index"]) for row in selected]
            beats = dataset.x[indices]
            for row in selected:
                meta = dataset.metadata(int(row["index"]))
                row.update({key: str(value) for key, value in meta.items()})
            path = figures_dir / f"{method}_{dataset_name}_{true_name}_to_{pred_name}_top{top_k}.png"
            plot_morphology_panel(beats, selected, path, f"{method} {dataset_name} {true_name}->{pred_name}")
    print(f"morphology panels written under {figures_dir}")


if __name__ == "__main__":
    main()
