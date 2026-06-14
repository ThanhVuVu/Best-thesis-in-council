from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, subset_first
from src.training.train_daeac_paper import evaluate_daeac_model, load_daeac_checkpoint
from src.utils.io import ensure_dir, write_json

try:
    from umap import UMAP
except ModuleNotFoundError:
    UMAP = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_mcc.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-name", default="daeac_mcc")
    parser.add_argument(
        "--dataset",
        default="target",
        help="One of source, target, both, external, all, or a key from data.external_targets such as incart/svdb.",
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--embedding-method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    device = device_from_torch()
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    figures_dir = ensure_dir(output / "figures" / args.method_name)
    diagnostics_dir = ensure_dir(output / "diagnostics" / args.method_name)
    model = load_daeac_checkpoint(args.checkpoint, config, device)
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))

    summaries = {}
    for dataset_name, path in _eval_datasets(config, args.dataset):
        ds = DAEACDataset(path, input_key=input_key, label_key=label_key, class_names=class_names)
        ds = subset_first(ds, args.max_samples)
        loader = DataLoader(ds, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False, num_workers=0)
        result = evaluate_daeac_model(model, loader, device, class_names)
        stem = f"{args.method_name}_{dataset_name}"
        summaries[dataset_name] = _summarize_result(result, class_names)
        _plot_confusion_normalized(result["metrics"]["confusion_matrix"], class_names, figures_dir / f"{stem}_confusion_normalized.png")
        _plot_prediction_distribution(result["y_true"], result["y_pred"], class_names, figures_dir / f"{stem}_prediction_distribution.png")
        _plot_entropy_by_class(result["probabilities"], result["y_true"], class_names, figures_dir / f"{stem}_entropy_by_true_class.png")
        _plot_embeddings(
            result["features"],
            result["y_true"],
            result["y_pred"],
            class_names,
            figures_dir / f"{stem}_{args.embedding_method}.png",
            method=args.embedding_method,
            seed=args.seed,
            title=f"{args.method_name} {dataset_name} gap_embed",
        )
        _write_error_focus_csv(diagnostics_dir / f"{stem}_error_focus.csv", result, class_names)
    write_json(
        {
            "checkpoint": str(args.checkpoint),
            "method_name": args.method_name,
            "dataset": args.dataset,
            "max_samples": args.max_samples,
            "summaries": summaries,
        },
        diagnostics_dir / f"{args.method_name}_visual_summary.json",
    )
    print(f"visualizations written under {figures_dir}")


def _eval_datasets(config: dict, dataset: str) -> list[tuple[str, Path]]:
    external = dict(config.get("data", {}).get("external_targets", {}))
    selected: list[tuple[str, Path]] = []
    if dataset in {"source", "both", "all"}:
        selected.append(("source_eval", cfg_path(config, "data", "source_eval")))
    if dataset in {"target", "both", "all"}:
        selected.append(("target_test", cfg_path(config, "data", "target_test")))
    if dataset in {"external", "all"}:
        selected.extend((name, _resolve_data_path(config, value)) for name, value in external.items())
    elif dataset in external:
        selected.append((dataset, _resolve_data_path(config, external[dataset])))
    if not selected:
        valid = ["source", "target", "both", "external", "all", *external.keys()]
        raise ValueError(f"Unknown dataset '{dataset}'. Valid values: {valid}")
    return selected


def _plot_embeddings(
    features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: Path,
    method: str,
    seed: int,
    title: str,
) -> None:
    if len(features) < 2:
        return
    if method == "tsne":
        perplexity = max(2, min(30, (len(features) - 1) // 3))
        coords = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(features)
        x_label = "t-SNE-1"
        y_label = "t-SNE-2"
    elif method == "umap" and UMAP is not None:
        coords = UMAP(n_components=2, random_state=seed).fit_transform(features)
        x_label = "UMAP-1"
        y_label = "UMAP-2"
    else:
        coords = PCA(n_components=2, random_state=seed).fit_transform(features)
        x_label = "PCA-1"
        y_label = "PCA-2"
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 6))
    correct = y_true == y_pred
    for cls, class_name in enumerate(class_names):
        for is_correct, marker, alpha, suffix in ((True, "o", 0.65, "correct"), (False, "x", 0.95, "error")):
            mask = (y_true == cls) & (correct == is_correct)
            if not mask.any():
                continue
            plt.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=alpha, marker=marker, label=f"{class_name}-{suffix}")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize=7, markerscale=1.5, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_confusion_normalized(cm: list[list[int]], class_names: list[str], path: Path) -> None:
    matrix = np.asarray(cm, dtype=np.float64)
    denom = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    normalized = matrix / denom
    ensure_dir(path.parent)
    plt.figure(figsize=(5, 4))
    image = plt.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", color="white" if normalized[i, j] > 0.5 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: Path) -> None:
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    x = np.arange(len(class_names))
    width = 0.38
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, true_counts, width, label="true")
    plt.bar(x + width / 2, pred_counts, width, label="pred")
    plt.xticks(x, class_names)
    plt.ylabel("Count")
    plt.title("True vs Predicted Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_entropy_by_class(probs: np.ndarray, y_true: np.ndarray, class_names: list[str], path: Path) -> None:
    entropy = -(probs * np.log(np.clip(probs, 1.0e-8, 1.0))).sum(axis=1)
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 4))
    values = [entropy[y_true == idx] for idx in range(len(class_names))]
    plt.boxplot(values, labels=class_names, showfliers=False)
    plt.ylabel("Entropy")
    plt.title("Prediction Entropy by True Class")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_error_focus_csv(path: Path, result: dict, class_names: list[str]) -> None:
    ensure_dir(path.parent)
    pairs = {("N", "F"), ("F", "N"), ("S", "N"), ("N", "V")}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_class", "pred_class", "confidence", "entropy", *[f"prob_{name}" for name in class_names]])
        for idx, (true, pred, prob) in enumerate(zip(result["y_true"], result["y_pred"], result["probabilities"])):
            true_name = class_names[int(true)]
            pred_name = class_names[int(pred)]
            if (true_name, pred_name) not in pairs:
                continue
            entropy = float(-(prob * np.log(np.clip(prob, 1.0e-8, 1.0))).sum())
            writer.writerow([idx, true_name, pred_name, float(np.max(prob)), entropy, *[float(v) for v in prob]])


def _summarize_result(result: dict, class_names: list[str]) -> dict:
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    return {
        "num_samples": int(len(y_true)),
        "accuracy": float(result["metrics"]["accuracy"]),
        "macro_f1": float(result["metrics"]["macro_f1"]),
        "true_counts": {name: int(np.sum(y_true == idx)) for idx, name in enumerate(class_names)},
        "pred_counts": {name: int(np.sum(y_pred == idx)) for idx, name in enumerate(class_names)},
    }


def _resolve_data_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path


if __name__ == "__main__":
    main()
