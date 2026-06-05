from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import cfg_path, device_from_torch, fit_val_datasets, loader, load_phase1_config, load_phase2p_checkpoint, maybe_subset
from src.data.datasets import ECGBeatTimeDataset
from src.training.dro import classifier_discrepancy
from src.training.prototypes import compute_prototypes, extract_biclassifier_outputs
from src.training.pseudolabels import select_confident_pseudolabels
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tag", default="phase2p")
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    device = device_from_torch()
    class_names = list(config["data"]["class_names"])
    checkpoint_path = Path(args.checkpoint or config["cluster_source"]["init_checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = cfg_path({"_base_dir": config["_base_dir"], "path": str(checkpoint_path)}, "path")
    model, _ = load_phase2p_checkpoint(checkpoint_path, config, device)

    fit_ds, _ = fit_val_datasets(config, use_duplicated=False)
    fit_ds = maybe_subset(fit_ds, args.max_source_samples)
    source_loader = loader(fit_ds, int(config["evaluation"]["batch_size"]), False, device)
    source_out = extract_biclassifier_outputs(model, source_loader, device, desc="source prototypes")
    source_prototypes, source_stats = compute_prototypes(source_out["embeddings"], source_out["y_true"], num_classes=3)
    source_discrepancy = classifier_discrepancy(
        torch.as_tensor(source_out["probabilities1"], dtype=torch.float32),
        torch.as_tensor(source_out["probabilities2"], dtype=torch.float32),
    )
    source_stats["mean_classifier_discrepancy"] = float(source_discrepancy)
    source_stats["checkpoint"] = str(checkpoint_path)
    source_stats["tag"] = args.tag

    proto_dir = ensure_dir(output / "prototypes")
    metrics_dir = ensure_dir(output / "metrics")
    pred_dir = ensure_dir(output / "predictions")
    torch.save({"prototypes": torch.as_tensor(source_prototypes), "stats": source_stats}, proto_dir / f"{args.tag}_source_prototypes.pt")
    write_json(source_stats, metrics_dir / f"{args.tag}_source_prototype_stats.json")

    target_ds = ECGBeatTimeDataset(cfg_path(config, "data", "target_unlabeled"), return_metadata=True)
    target_ds = maybe_subset(target_ds, args.max_target_samples)
    target_loader = loader(target_ds, int(config["evaluation"]["batch_size"]), False, device)
    target_out = extract_biclassifier_outputs(model, target_loader, device, desc="target pseudo-labels")
    pseudo_cfg = config["pseudolabel"]
    pseudo_df, selected, pseudo_stats = select_confident_pseudolabels(
        embeddings=target_out["embeddings"],
        probabilities=target_out["probabilities"],
        probabilities1=target_out["probabilities1"],
        probabilities2=target_out["probabilities2"],
        source_prototypes=source_prototypes,
        source_stats=source_stats,
        confidence_thresholds=pseudo_cfg["confidence_thresholds"],
        class_names=class_names,
        distance_quantile_key="q95",
        min_target_per_class=int(pseudo_cfg.get("min_target_per_class", 20)),
        metadata=target_out["metadata"],
    )
    pseudo_df.to_csv(pred_dir / f"{args.tag}_target_pseudolabels.csv", index=False)
    pseudo_stats["checkpoint"] = str(checkpoint_path)
    pseudo_stats["tag"] = args.tag
    write_json(pseudo_stats, metrics_dir / f"{args.tag}_pseudolabel_stats.json")

    if bool(selected.any()):
        target_labels = pseudo_df.loc[selected, "pseudo_label"].to_numpy(dtype=np.int64)
        target_prototypes, target_stats = compute_prototypes(target_out["embeddings"][selected], target_labels, num_classes=3)
    else:
        target_prototypes = np.zeros_like(source_prototypes)
        target_stats = {"class_counts": {str(i): 0 for i in range(3)}, "intra_class_distance": {}, "pairwise_distance": {}}
    target_stats["selected_total"] = int(selected.sum())
    target_stats["checkpoint"] = str(checkpoint_path)
    target_stats["tag"] = args.tag
    torch.save({"prototypes": torch.as_tensor(target_prototypes), "stats": target_stats}, proto_dir / f"{args.tag}_target_prototypes.pt")
    write_json(target_stats, metrics_dir / f"{args.tag}_target_prototype_stats.json")
    print(f"Selected {int(selected.sum())}/{len(selected)} target pseudo-labels for tag={args.tag}.")


if __name__ == "__main__":
    main()
