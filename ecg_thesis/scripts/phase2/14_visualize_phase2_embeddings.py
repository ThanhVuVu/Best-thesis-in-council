from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset
from src.training.evaluate import predict_model
from src.training.train import load_model_from_checkpoint
from src.training.train_dann import load_dann_from_checkpoint
from src.utils.io import ensure_dir, read_json
from src.visualization.plot_embeddings import plot_umap_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_dann.yaml")
    parser.add_argument("--source-checkpoint", default="outputs/checkpoints/source_only_inception_best.pt")
    parser.add_argument("--dann-checkpoint", default="outputs/checkpoints/dann_best.pt")
    parser.add_argument("--max-points-per-domain", type=int, default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    device = device_from_torch()
    print(f"Using device: {device}")
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    figures_dir = ensure_dir(output / "figures" / "phase2")
    max_points = args.max_points_per_domain or int(config["visualization"]["max_points_per_domain"])

    source_model, _ = load_model_from_checkpoint(_resolve_checkpoint(config, args.source_checkpoint), device)
    dann_model, _ = load_dann_from_checkpoint(_resolve_checkpoint(config, args.dann_checkpoint), device)
    subset = _embedding_subset(config, max_points)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    source_result = predict_model(source_model, loader, device, collect_embeddings=True, desc="source-only embeddings")
    domains = [str(m["domain"]) for m in source_result["metadata"]]
    plot_umap_embeddings(
        source_result["embeddings"],
        source_result["y_true"],
        domains,
        figures_dir / "umap_source_only_inception.png",
        seed=int(config["seed"]),
    )

    dann_result = predict_model(dann_model, loader, device, collect_embeddings=True, desc="dann embeddings")
    domains = [str(m["domain"]) for m in dann_result["metadata"]]
    plot_umap_embeddings(
        dann_result["embeddings"],
        dann_result["y_true"],
        domains,
        figures_dir / "umap_dann.png",
        seed=int(config["seed"]),
    )

    _plot_training_curves(output / "logs" / "dann_train_log.csv", figures_dir / "training_curves.png")
    _plot_per_class_f1(output / "metrics", figures_dir / "per_class_f1_comparison.png")
    print(f"Saved Phase 2 figures to {figures_dir}")


def _resolve_checkpoint(config: dict, checkpoint: str) -> Path:
    return cfg_path({"paths": {"checkpoint": checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")


def _embedding_subset(config: dict, max_points_per_domain: int):
    rng = np.random.default_rng(int(config["seed"]))
    source = ECGBeatDataset(cfg_path(config, "data", "source_test"), return_metadata=True)
    target = ECGBeatDataset(cfg_path(config, "data", "target_test"), return_metadata=True)
    subsets = []
    for dataset in (source, target):
        size = min(max_points_per_domain, len(dataset))
        indices = rng.choice(np.arange(len(dataset)), size=size, replace=False).tolist()
        subsets.append(Subset(dataset, indices))
    return ConcatDataset(subsets)


def _plot_training_curves(csv_path: Path, output_path: Path) -> None:
    if not csv_path.exists():
        return
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    keys = ["loss", "loss_cls", "loss_domain", "source_val_macro_f1", "domain_accuracy", "lambda"]
    plt.figure(figsize=(10, 6))
    for key in keys:
        if key in rows[0]:
            plt.plot(epochs, [float(r[key]) for r in rows], label=key)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_per_class_f1(metrics_dir: Path, output_path: Path) -> None:
    metric_files = [
        ("Source-only INCART", metrics_dir / "source_only_inception_incart_heldout_metrics.json"),
        ("DANN INCART", metrics_dir / "dann_incart_heldout_metrics.json"),
    ]
    labels = []
    values = {"N": [], "S": [], "V": []}
    for label, path in metric_files:
        if not path.exists():
            continue
        metrics = read_json(path)
        labels.append(label)
        for cls in values:
            values[cls].append(float(metrics["per_class"][cls]["f1"]))
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for i, cls in enumerate(["N", "S", "V"]):
        plt.bar(x + (i - 1) * width, values[cls], width=width, label=cls)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
