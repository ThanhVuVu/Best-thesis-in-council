from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.utils.io import ensure_dir


def plot_temporal_contrast(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    if not rows:
        return
    output = ensure_dir(output_dir)
    classes = sorted({str(row["minority_class"]) for row in rows})
    for cls in classes:
        subset = [row for row in rows if row["minority_class"] == cls]
        x = np.asarray([row["time_index"] for row in subset], dtype=np.int64)
        diff = np.asarray([row["abs_mean_diff"] for row in subset], dtype=np.float64)
        d = np.asarray([row["cohen_d"] for row in subset], dtype=np.float64)
        order = np.argsort(x)
        plt.figure(figsize=(8, 4))
        plt.plot(x[order], diff[order], label="abs mean diff")
        plt.plot(x[order], np.abs(d[order]), label="abs Cohen d", alpha=0.75)
        plt.xlabel("Relative morphology/raw window index")
        plt.ylabel("Contrast")
        plt.title(f"{cls} vs N temporal contrast")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"temporal_contrast_{cls}_vs_N.png", dpi=180)
        plt.close()


def plot_layer_collapse(pairwise_rows: list[dict[str, Any]], output_path: str | Path) -> None:
    if not pairwise_rows:
        return
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    layers = list(dict.fromkeys(str(row["layer"]) for row in pairwise_rows))
    classes = sorted({str(row["minority_class"]) for row in pairwise_rows})
    plt.figure(figsize=(10, 5))
    for cls in classes:
        values = []
        for layer in layers:
            match = [row for row in pairwise_rows if row["layer"] == layer and row["minority_class"] == cls]
            values.append(float(match[0]["fisher_ratio_proxy"]) if match else np.nan)
        plt.plot(range(len(layers)), values, marker="o", label=f"{cls} vs N")
    plt.xticks(range(len(layers)), layers, rotation=45, ha="right")
    plt.ylabel("Fisher ratio proxy")
    plt.title("Layer-wise minority-to-N separability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pca_embeddings(features_by_layer: dict[str, np.ndarray], y: np.ndarray, class_names: list[str], output_dir: str | Path, max_points: int) -> None:
    output = ensure_dir(output_dir)
    rng = np.random.default_rng(42)
    for layer, features in features_by_layer.items():
        x = np.asarray(features)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        if len(x) < 2:
            continue
        indices = np.arange(len(x))
        if len(indices) > int(max_points):
            indices = rng.choice(indices, size=int(max_points), replace=False)
        coords = PCA(n_components=2, random_state=42).fit_transform(np.nan_to_num(x[indices]))
        plt.figure(figsize=(7, 5))
        for cls, name in enumerate(class_names):
            mask = y[indices] == cls
            if mask.any():
                plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.65, label=name)
        plt.title(f"{layer} PCA")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"embedding_pca_{layer}.png", dpi=180)
        plt.close()


def plot_feature_effect_heatmap(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    if not rows:
        return
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    features = list(dict.fromkeys(str(row["feature"]) for row in rows))
    pairs = list(dict.fromkeys(str(row["class_pair"]) for row in rows))
    matrix = np.zeros((len(features), len(pairs)), dtype=np.float64)
    for row in rows:
        matrix[features.index(str(row["feature"])), pairs.index(str(row["class_pair"]))] = abs(float(row.get("cohen_d", 0.0)))
    plt.figure(figsize=(max(6, len(pairs) * 1.6), max(4, len(features) * 0.28)))
    image = plt.imshow(matrix, aspect="auto", cmap="magma")
    plt.colorbar(image, label="abs Cohen d")
    plt.xticks(range(len(pairs)), pairs, rotation=30, ha="right")
    plt.yticks(range(len(features)), features)
    plt.title("Minority vs N feature effect sizes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
