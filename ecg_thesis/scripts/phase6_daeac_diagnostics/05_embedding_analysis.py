from __future__ import annotations

import argparse

import numpy as np

from diagnostics_common import (
    class_names_from,
    embedding_path,
    load_diagnostics_config,
    method_name,
    output_dir,
    selected_datasets,
    write_csv,
)
from src.training.diagnostics import centroid_distance_matrix, safe_silhouette_score
from src.utils.io import ensure_dir, write_json
from src.visualization.diagnostics import plot_embedding, plot_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--dataset", default="configured")
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diag_config, base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    class_names = class_names_from(base_config)
    max_samples = int(args.max_samples or diag_config["analysis"].get("embedding_max_samples", 10000))
    out = output_dir(diag_config)
    tables_dir = ensure_dir(out / "diagnostics" / "embeddings")
    figures_dir = ensure_dir(out / "figures" / "embeddings")
    summary = {}

    dataset_names = list(diag_config["analysis"].get("datasets", []))
    for dataset_name, _path in selected_datasets(base_config, args.dataset, configured=dataset_names):
        path = embedding_path(diag_config, method, dataset_name)
        if not path.exists():
            print(f"skipped {dataset_name}: missing embeddings {path}")
            continue
        data = np.load(path, allow_pickle=True)
        features = data["features"]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        indices = _sample_indices(y_true, max_samples=max_samples, seed=args.seed)
        prefix = f"{method}_{dataset_name}"
        plot_embedding(
            features[indices],
            y_true[indices],
            y_pred[indices],
            class_names,
            figures_dir / f"{prefix}_{args.method}.png",
            method=args.method,
            seed=args.seed,
            title=f"{prefix} gap_embed",
        )
        distances, labels = centroid_distance_matrix(features, y_true, class_names)
        write_csv(tables_dir / f"{prefix}_centroid_distances.csv", _matrix_rows(distances, labels))
        plot_heatmap(distances, labels, figures_dir / f"{prefix}_centroid_distances.png", f"{prefix} centroid distances")
        silhouette = safe_silhouette_score(features[indices], y_true[indices])
        summary[dataset_name] = {"silhouette_true_class": silhouette, "num_embedding_samples": int(len(indices))}
    write_json(summary, tables_dir / f"{method}_embedding_summary.json")
    print(f"embedding outputs written under {tables_dir} and {figures_dir}")


def _sample_indices(y_true: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    n = len(y_true)
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(int(seed))
    per_class = max(1, int(max_samples) // max(len(np.unique(y_true)), 1))
    chosen = []
    for cls in np.unique(y_true):
        idx = np.flatnonzero(y_true == cls)
        take = min(len(idx), per_class)
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    remaining = max_samples - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n), np.asarray(chosen, dtype=np.int64), assume_unique=False)
        chosen.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _matrix_rows(matrix: np.ndarray, labels: list[str]) -> list[dict[str, float | str]]:
    rows = []
    for idx, label in enumerate(labels):
        row: dict[str, float | str] = {"class": label}
        for jdx, pred in enumerate(labels):
            row[pred] = float(matrix[idx, jdx])
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
