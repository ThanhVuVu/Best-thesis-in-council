from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset
from src.training.evaluate import predict_model
from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir
from src.visualization.plot_beats import plot_example_beats
from src.visualization.plot_embeddings import plot_umap_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    device = device_from_torch()
    checkpoint_path = cfg_path({"paths": {"checkpoint": args.checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    processed = cfg_path(config, "paths", "processed_dir")
    figures = ensure_dir(cfg_path(config, "paths", "output_dir") / "figures")

    plot_example_beats(
        [processed / "mitbih_test.npz", processed / "incart_test.npz"],
        figures / "example_beats.png",
        seed=int(config["seed"]),
    )

    mit = ECGBeatDataset(processed / "mitbih_test.npz", return_metadata=True)
    inc = ECGBeatDataset(processed / "incart_test.npz", return_metadata=True)
    subset = _balanced_subset([mit, inc], int(config["visualization"]["max_points_per_domain"]), int(config["seed"]))
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
    result = predict_model(model, loader, device, collect_embeddings=True)
    domains = [str(m["domain"]) for m in result["metadata"]]
    plot_umap_embeddings(
        result["embeddings"],
        result["y_true"],
        domains,
        figures / "embedding_umap.png",
        seed=int(config["seed"]),
    )
    print(f"Saved figures to {figures}")


def _balanced_subset(datasets, max_points_per_domain: int, seed: int):
    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    subsets = []
    for ds in datasets:
        n = len(ds)
        size = min(max_points_per_domain, n)
        indices = rng.choice(np.arange(n), size=size, replace=False).tolist()
        subsets.append(Subset(ds, indices))
    return ConcatDataset(subsets)


if __name__ == "__main__":
    main()
