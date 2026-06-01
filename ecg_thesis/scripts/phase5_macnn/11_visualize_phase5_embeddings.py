from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset
from src.training.evaluate import predict_model
from src.training.train_dann import load_dann_from_checkpoint
from src.training.train_macnn import evaluate_macnn_model, load_macnn_checkpoint
from src.visualization.plot_embeddings import plot_umap_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=["macnn", "dann"], default="macnn")
    parser.add_argument("--method-name", default="macnn_se_daeac")
    parser.add_argument("--max-source-samples", type=int, default=3000)
    parser.add_argument("--max-target-samples", type=int, default=3000)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    device = device_from_torch()
    if args.kind == "dann":
        model, _ = load_dann_from_checkpoint(_resolve_checkpoint(args.checkpoint, config), device)
    else:
        model = load_macnn_checkpoint(_resolve_checkpoint(args.checkpoint, config), config, device)

    source = _subset(ECGMACNNDataset(cfg_path(config, "data", "source_test")), args.max_source_samples)
    target = _subset(ECGMACNNDataset(cfg_path(config, "data", "target_test")), args.max_target_samples)
    source_loader = DataLoader(source, batch_size=256, shuffle=False, num_workers=0)
    target_loader = DataLoader(target, batch_size=256, shuffle=False, num_workers=0)

    if args.kind == "dann":
        src = predict_model(model, source_loader, device, collect_embeddings=True, desc=f"{args.method_name} source embeddings")
        tgt = predict_model(model, target_loader, device, collect_embeddings=True, desc=f"{args.method_name} target embeddings")
    else:
        src = evaluate_macnn_model(model, source_loader, device, config["data"]["class_names"], desc=f"{args.method_name} source embeddings", collect_embeddings=True)
        tgt = evaluate_macnn_model(model, target_loader, device, config["data"]["class_names"], desc=f"{args.method_name} target embeddings", collect_embeddings=True)

    embeddings = np.concatenate([src["embeddings"], tgt["embeddings"]], axis=0)
    true_labels = np.concatenate([src["y_true"], tgt["y_true"]], axis=0)
    pseudo_labels = np.concatenate([src["y_pred"], tgt["y_pred"]], axis=0)
    domains = ["mitbih"] * len(src["y_true"]) + ["incart"] * len(tgt["y_true"])
    fig_dir = cfg_path(config, "paths", "output_dir") / "figures" / "phase5"
    plot_umap_embeddings(
        embeddings,
        true_labels,
        domains,
        fig_dir / f"umap_{args.method_name}_true_labels.png",
        seed=int(config["seed"]),
        title=f"{args.method_name} embeddings - true labels",
    )
    plot_umap_embeddings(
        embeddings,
        pseudo_labels,
        domains,
        fig_dir / f"umap_{args.method_name}_pseudo_labels.png",
        seed=int(config["seed"]),
        title=f"{args.method_name} embeddings - predicted/pseudo labels",
    )


def _subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(int(max_samples), len(dataset)))))


def _resolve_checkpoint(value: str, config: dict) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path


if __name__ == "__main__":
    main()
