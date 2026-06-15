from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.io import ensure_dir

try:
    from umap import UMAP
except ModuleNotFoundError:
    UMAP = None


def plot_reliability(bins: list[dict[str, float]], path: str | Path, title: str) -> None:
    ensure_dir(Path(path).parent)
    centers = [(row["bin_start"] + row["bin_end"]) / 2 for row in bins]
    acc = [row["accuracy"] for row in bins]
    conf = [row["confidence"] for row in bins]
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.plot(conf, acc, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confidence_histogram(conf_correct: np.ndarray, conf_wrong: np.ndarray, path: str | Path, title: str) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(6, 4))
    plt.hist(conf_correct, bins=30, alpha=0.65, label="correct")
    plt.hist(conf_wrong, bins=30, alpha=0.65, label="wrong")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_entropy_by_class(entropy: np.ndarray, y_true: np.ndarray, class_names: list[str], path: str | Path, title: str) -> None:
    ensure_dir(Path(path).parent)
    values = [np.asarray(entropy)[np.asarray(y_true) == idx] for idx in range(len(class_names))]
    plt.figure(figsize=(7, 4))
    plt.boxplot(values, labels=class_names, showfliers=False)
    plt.ylabel("Entropy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_curve(curve: list[dict[str, Any]], x_key: str, y_key: str, path: str | Path, title: str, x_label: str, y_label: str) -> None:
    ensure_dir(Path(path).parent)
    if not curve:
        return
    plt.figure(figsize=(5, 4))
    plt.plot([float(row[x_key]) for row in curve], [float(row[y_key]) for row in curve])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_embedding(
    features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: str | Path,
    method: str,
    seed: int,
    title: str,
) -> None:
    if len(features) < 2:
        return
    coords, x_label, y_label = _reduce(features, method=method, seed=seed)
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(8, 6))
    correct = np.asarray(y_true) == np.asarray(y_pred)
    for cls, name in enumerate(class_names):
        for is_correct, marker, alpha, suffix in ((True, "o", 0.6, "correct"), (False, "x", 0.95, "error")):
            mask = (np.asarray(y_true) == cls) & (correct == is_correct)
            if mask.any():
                plt.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=alpha, marker=marker, label=f"{name}-{suffix}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_heatmap(matrix: np.ndarray, labels: list[str], path: str | Path, title: str) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(max(5, len(labels)), max(4, len(labels) * 0.8)))
    image = plt.imshow(matrix, cmap="viridis")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_morphology_panel(
    beats: np.ndarray,
    rows: list[dict[str, Any]],
    path: str | Path,
    title: str,
) -> None:
    if len(rows) == 0:
        return
    ensure_dir(Path(path).parent)
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.4, 2.0 * n)), squeeze=False)
    x = np.asarray(beats)
    for ax, beat, row in zip(axes[:, 0], x, rows):
        values = beat.squeeze()
        if values.ndim == 2:
            for ch in range(values.shape[0]):
                ax.plot(values[ch], linewidth=1, alpha=0.8, label=f"ch{ch}")
        else:
            ax.plot(values, linewidth=1)
        ax.set_title(
            f"idx={row.get('index')} {row.get('true_class')}->{row.get('pred_class')} "
            f"conf={float(row.get('confidence', 0.0)):.3f} rec={row.get('record', '')} sym={row.get('symbol', '')}",
            fontsize=9,
        )
        ax.set_xticks([])
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _reduce(features: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, str, str]:
    method_norm = str(method).lower()
    values = np.asarray(features)
    if method_norm == "tsne":
        perplexity = max(2, min(30, (len(values) - 1) // 3))
        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(values), "t-SNE-1", "t-SNE-2"
    if method_norm == "umap" and UMAP is not None:
        return UMAP(n_components=2, random_state=seed).fit_transform(values), "UMAP-1", "UMAP-2"
    return PCA(n_components=2, random_state=seed).fit_transform(values), "PCA-1", "PCA-2"
