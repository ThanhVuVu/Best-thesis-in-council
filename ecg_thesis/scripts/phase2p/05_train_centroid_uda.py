from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from common import (
    batch_to_device,
    class_weights_for,
    cfg_path,
    device_from_torch,
    evaluate_model,
    fit_val_datasets,
    loader,
    load_phase1_config,
    load_phase2p_checkpoint,
    maybe_subset,
    save_checkpoint,
    write_eval_outputs,
    write_history,
)
from src.data.datasets import ECGBeatTimeDataset
from src.training.dro import ClassGroupDROLoss, classifier_discrepancy
from src.training.prototypes import compactness_loss, compute_prototypes, extract_biclassifier_outputs
from src.training.pseudolabels import select_confident_pseudolabels
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


class PseudoLabelSubset(Dataset):
    def __init__(self, base: ECGBeatTimeDataset, indices: np.ndarray, labels: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.indices) != len(self.labels):
            raise ValueError("Pseudo-label indices and labels length mismatch.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        x, time_features, _ = self.base[int(self.indices[idx])]
        return x, time_features, torch.tensor(int(self.labels[idx]), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    parser.add_argument("--cluster-epochs", type=int, default=None)
    parser.add_argument("--uda-epochs", type=int, default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--skip-cluster", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    device = device_from_torch()
    class_names = list(config["data"]["class_names"])

    fit_ds, val_ds = fit_val_datasets(config, use_duplicated=bool(config["source_pretrain"].get("use_duplicated_source", True)))
    source_proto_ds, _ = fit_val_datasets(config, use_duplicated=False)
    fit_ds = maybe_subset(fit_ds, args.max_source_samples)
    source_proto_ds = maybe_subset(source_proto_ds, args.max_source_samples)
    val_ds = maybe_subset(val_ds, args.max_val_samples)

    cluster_cfg = copy.deepcopy(config["cluster_source"])
    uda_cfg = copy.deepcopy(config["uda"])
    if args.cluster_epochs is not None:
        cluster_cfg["epochs"] = int(args.cluster_epochs)
    if args.uda_epochs is not None:
        uda_cfg["epochs"] = int(args.uda_epochs)

    init_ckpt = _resolve(config, cluster_cfg["init_checkpoint"])
    model, _ = load_phase2p_checkpoint(init_ckpt, config, device)
    if not args.skip_cluster:
        model = train_source_cluster(model, config, cluster_cfg, fit_ds, val_ds, device, output)
    else:
        cluster_ckpt = _resolve(config, uda_cfg["init_checkpoint"])
        model, _ = load_phase2p_checkpoint(cluster_ckpt, config, device)

    source_prototypes, source_stats, target_prototypes, target_stats, pseudo_df = recompute_clustered_artifacts(
        model=model,
        config=config,
        source_ds=source_proto_ds,
        max_target_samples=args.max_target_samples,
        output=output,
        device=device,
        class_names=class_names,
    )
    if int(pseudo_df["selected"].sum()) == 0:
        raise RuntimeError("No target pseudo-labels selected. Loosen thresholds or inspect clustered pseudolabel stats.")
    target_base = ECGBeatTimeDataset(cfg_path(config, "data", "target_unlabeled"))
    if args.max_target_samples is not None:
        target_base = maybe_subset(target_base, args.max_target_samples)
    selected_df = pseudo_df[pseudo_df["selected"]].copy()
    target_ds = PseudoLabelSubset(target_base, selected_df["index"].to_numpy(), selected_df["pseudo_label"].to_numpy())
    train_uda(
        model=model,
        config=config,
        uda_cfg=uda_cfg,
        source_ds=fit_ds,
        val_ds=val_ds,
        target_ds=target_ds,
        source_prototypes=source_prototypes,
        source_stats=source_stats,
        target_prototypes=target_prototypes,
        target_stats=target_stats,
        output=output,
        device=device,
    )


def train_source_cluster(model, config, cluster_cfg, fit_ds, val_ds, device, output: Path):
    prefix = str(cluster_cfg["checkpoint_prefix"])
    ckpt_dir = ensure_dir(output / "checkpoints")
    log_dir = ensure_dir(output / "logs")
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cluster_cfg["lr"]), weight_decay=float(cluster_cfg["weight_decay"]))
    class_weights = class_weights_for(fit_ds, {"use_class_weights": cluster_cfg.get("use_class_weights", True), "num_classes": 3}, device)
    source_ref = _source_prototypes_for_training(model, fit_ds, int(config["evaluation"]["batch_size"]), device)
    source_proto_t = torch.as_tensor(source_ref, dtype=torch.float32, device=device)
    fit_loader = loader(fit_ds, int(cluster_cfg["batch_size"]), True, device)
    best_f1 = -1.0
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, int(cluster_cfg["epochs"]) + 1):
        model.train()
        rows = []
        for batch in tqdm(fit_loader, desc=f"phase2p source cluster epoch {epoch}", dynamic_ncols=True):
            x, time_features, y, _ = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x, time_features, return_all=True)
            ce = F.cross_entropy(out["logits"], y, weight=class_weights)
            compact = compactness_loss(out["embedding"], y, source_proto_t)
            batch_proto, valid = _batch_prototypes(out["embedding"], y, 3)
            sep = _masked_separation_loss(batch_proto, valid, float(cluster_cfg.get("separation_margin", 1.0)))
            loss = ce + float(cluster_cfg.get("compact_weight", 0.05)) * compact + float(cluster_cfg.get("separation_weight", 0.01)) * sep
            loss.backward()
            optimizer.step()
            rows.append({"loss": float(loss.detach().cpu()), "ce": float(ce.detach().cpu()), "compact": float(compact.detach().cpu()), "separation": float(sep.detach().cpu())})
        val_result = evaluate_model(model, val_ds, device, batch_size=int(config["evaluation"]["batch_size"]))
        row = {
            "epoch": epoch,
            "loss": float(np.mean([r["loss"] for r in rows])),
            "ce": float(np.mean([r["ce"] for r in rows])),
            "compact": float(np.mean([r["compact"] for r in rows])),
            "separation": float(np.mean([r["separation"] for r in rows])),
            "val_macro_f1": val_result["metrics"]["macro_f1"],
            "val_accuracy": val_result["metrics"]["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)
        if row["val_macro_f1"] > best_f1:
            best_f1 = row["val_macro_f1"]
            best_epoch = epoch
            stale = 0
            save_checkpoint(model, optimizer, config, epoch, best_f1, history, best_path)
        else:
            stale += 1
        save_checkpoint(model, optimizer, config, epoch, best_f1, history, latest_path)
        if stale >= int(cluster_cfg["early_stopping_patience"]):
            break
    write_history(history, log_dir / f"{prefix}_train_log.csv")
    write_json(
        {"best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path), "best_epoch": best_epoch, "best_val_macro_f1": best_f1},
        output / "metrics" / "phase2p_cluster_source_train_summary.json",
    )
    best_model, _ = load_phase2p_checkpoint(best_path, config, device)
    write_eval_outputs(evaluate_model(best_model, val_ds, device, batch_size=int(config["evaluation"]["batch_size"])), output, "phase2p_cluster_source_validation", config["data"]["class_names"])
    return best_model


def recompute_clustered_artifacts(model, config, source_ds, max_target_samples, output: Path, device, class_names):
    batch_size = int(config["evaluation"]["batch_size"])
    source_out = extract_biclassifier_outputs(model, loader(source_ds, batch_size, False, device), device, desc="clustered source prototypes")
    source_prototypes, source_stats = compute_prototypes(source_out["embeddings"], source_out["y_true"], num_classes=3)
    source_stats["mean_classifier_discrepancy"] = float(np.linalg.norm(source_out["probabilities1"] - source_out["probabilities2"], axis=1).mean())
    source_stats["tag"] = "phase2p_clustered"
    ensure_dir(output / "prototypes")
    ensure_dir(output / "metrics")
    torch.save({"prototypes": torch.as_tensor(source_prototypes), "stats": source_stats}, output / "prototypes" / "phase2p_clustered_source_prototypes.pt")
    write_json(source_stats, output / "metrics" / "phase2p_clustered_source_prototype_stats.json")

    target_ds = ECGBeatTimeDataset(cfg_path(config, "data", "target_unlabeled"), return_metadata=True)
    target_ds = maybe_subset(target_ds, max_target_samples)
    target_out = extract_biclassifier_outputs(model, loader(target_ds, batch_size, False, device), device, desc="clustered target pseudo-labels")
    pseudo_cfg = config["pseudolabel"]
    pseudo_df, selected, pseudo_stats = select_confident_pseudolabels(
        target_out["embeddings"],
        target_out["probabilities"],
        target_out["probabilities1"],
        target_out["probabilities2"],
        source_prototypes,
        source_stats,
        pseudo_cfg["confidence_thresholds"],
        class_names,
        min_target_per_class=int(pseudo_cfg.get("min_target_per_class", 20)),
        metadata=target_out["metadata"],
    )
    ensure_dir(output / "predictions")
    pseudo_df.to_csv(output / "predictions" / "phase2p_clustered_target_pseudolabels.csv", index=False)
    write_json(pseudo_stats, output / "metrics" / "phase2p_clustered_pseudolabel_stats.json")
    if bool(selected.any()):
        target_labels = pseudo_df.loc[selected, "pseudo_label"].to_numpy(dtype=np.int64)
        target_prototypes, target_stats = compute_prototypes(target_out["embeddings"][selected], target_labels, num_classes=3)
    else:
        target_prototypes = np.zeros_like(source_prototypes)
        target_stats = {"class_counts": {str(i): 0 for i in range(3)}, "intra_class_distance": {}, "pairwise_distance": {}}
    target_stats["selected_total"] = int(selected.sum())
    target_stats["tag"] = "phase2p_clustered"
    torch.save({"prototypes": torch.as_tensor(target_prototypes), "stats": target_stats}, output / "prototypes" / "phase2p_clustered_target_prototypes.pt")
    write_json(target_stats, output / "metrics" / "phase2p_clustered_target_prototype_stats.json")
    return source_prototypes, source_stats, target_prototypes, target_stats, pseudo_df


def train_uda(model, config, uda_cfg, source_ds, val_ds, target_ds, source_prototypes, source_stats, target_prototypes, target_stats, output: Path, device):
    prefix = str(uda_cfg["checkpoint_prefix"])
    ckpt_dir = ensure_dir(output / "checkpoints")
    log_dir = ensure_dir(output / "logs")
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(uda_cfg["lr"]), weight_decay=float(uda_cfg["weight_decay"]))
    class_weights = class_weights_for(source_ds, {"use_class_weights": uda_cfg.get("use_class_weights", True), "num_classes": 3}, device)
    dro = ClassGroupDROLoss(num_classes=3, class_weights=class_weights, eta=float(uda_cfg.get("dro_eta", 0.1))).to(device)
    source_proto_t = torch.as_tensor(source_prototypes, dtype=torch.float32, device=device)
    target_proto_t = torch.as_tensor(target_prototypes, dtype=torch.float32, device=device)
    target_counts = target_stats.get("class_counts", {})
    valid_target = torch.as_tensor([int(target_counts.get(str(i), 0)) > 0 for i in range(3)], dtype=torch.bool, device=device)
    global_proto = 0.5 * (source_proto_t + target_proto_t)
    source_loader = loader(source_ds, int(uda_cfg["source_batch_size"]), True, device)
    target_loader = loader(target_ds, int(uda_cfg["target_batch_size"]), True, device)
    best_f1 = -1.0
    best_epoch = -1
    history = []
    target_iter = iter(target_loader)
    for epoch in range(1, int(uda_cfg["epochs"]) + 1):
        model.train()
        rows = []
        for source_batch in tqdm(source_loader, desc=f"phase2p uda epoch {epoch}", dynamic_ncols=True):
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            sx, stime, sy, _ = batch_to_device(source_batch, device)
            tx, ttime, ty, _ = batch_to_device(target_batch, device)
            optimizer.zero_grad(set_to_none=True)
            sout = model(sx, stime, return_all=True)
            tout = model(tx, ttime, return_all=True)
            source_ce, dro_stats = dro(sout["logits"], sy)
            source_compact = compactness_loss(sout["embedding"], sy, source_proto_t)
            target_compact = compactness_loss(tout["embedding"], ty, target_proto_t)
            sp, sv = _batch_prototypes(sout["embedding"], sy, 3)
            tp, tv = _batch_prototypes(tout["embedding"], ty, 3)
            both_valid = sv & tv & valid_target
            inter = _masked_inter_domain_loss(sp, tp, both_valid)
            sep = _masked_separation_loss(sp, sv, float(uda_cfg.get("separation_margin", 1.0)))
            running = _running_combined_loss(sp, sv, tp, tv, global_proto, valid_target)
            disc = classifier_discrepancy(sout["probabilities1"], sout["probabilities2"])
            loss = (
                source_ce
                + float(uda_cfg.get("compact_source_weight", 0.05)) * source_compact
                + float(uda_cfg.get("compact_target_weight", 0.05)) * target_compact
                + float(uda_cfg.get("inter_domain_weight", 0.05)) * inter
                + float(uda_cfg.get("running_combined_weight", 0.05)) * running
                + float(uda_cfg.get("separation_weight", 0.01)) * sep
                + float(uda_cfg.get("discrepancy_weight", 0.0)) * disc
            )
            loss.backward()
            optimizer.step()
            row = {
                "loss": float(loss.detach().cpu()),
                "source_ce": float(source_ce.detach().cpu()),
                "source_compact": float(source_compact.detach().cpu()),
                "target_compact": float(target_compact.detach().cpu()),
                "inter_domain": float(inter.detach().cpu()),
                "running_combined": float(running.detach().cpu()),
                "separation": float(sep.detach().cpu()),
                "source_discrepancy": float(disc.detach().cpu()),
            }
            row.update(dro_stats)
            rows.append(row)
        val_result = evaluate_model(model, val_ds, device, batch_size=int(config["evaluation"]["batch_size"]))
        epoch_row = {key: float(np.mean([r[key] for r in rows])) for key in rows[0].keys()}
        epoch_row.update({"epoch": epoch, "val_macro_f1": val_result["metrics"]["macro_f1"], "val_accuracy": val_result["metrics"]["accuracy"], "lr": optimizer.param_groups[0]["lr"]})
        history.append(epoch_row)
        print(epoch_row)
        if epoch_row["val_macro_f1"] > best_f1:
            best_f1 = epoch_row["val_macro_f1"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, config, epoch, best_f1, history, best_path)
        save_checkpoint(model, optimizer, config, epoch, best_f1, history, latest_path)
    write_history(history, log_dir / f"{prefix}_train_log.csv")
    write_json(
        {
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_path),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "target_selected_samples": len(target_ds),
            "source_prototype_stats": source_stats,
            "target_prototype_stats": target_stats,
        },
        output / "metrics" / "phase2p_centroid_uda_train_summary.json",
    )
    best_model, _ = load_phase2p_checkpoint(best_path, config, device)
    write_eval_outputs(evaluate_model(best_model, val_ds, device, batch_size=int(config["evaluation"]["batch_size"])), output, "phase2p_uda_source_validation", config["data"]["class_names"])


def _source_prototypes_for_training(model, dataset, batch_size: int, device) -> np.ndarray:
    outputs = extract_biclassifier_outputs(model, loader(dataset, batch_size, False, device), device, desc="source prototype reference")
    prototypes, _ = compute_prototypes(outputs["embeddings"], outputs["y_true"], num_classes=3)
    return prototypes


def _batch_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int):
    prototypes = torch.zeros(num_classes, embeddings.shape[1], dtype=embeddings.dtype, device=embeddings.device)
    valid = torch.zeros(num_classes, dtype=torch.bool, device=embeddings.device)
    for cls in range(num_classes):
        mask = labels == cls
        if bool(mask.any()):
            prototypes[cls] = embeddings[mask].mean(dim=0)
            valid[cls] = True
    return prototypes, valid


def _masked_separation_loss(prototypes: torch.Tensor, valid: torch.Tensor, margin: float) -> torch.Tensor:
    losses = []
    for i in range(prototypes.shape[0]):
        for j in range(i + 1, prototypes.shape[0]):
            if bool(valid[i] and valid[j]):
                dist = torch.linalg.vector_norm(prototypes[i] - prototypes[j], ord=2)
                losses.append(torch.relu(torch.as_tensor(margin, dtype=prototypes.dtype, device=prototypes.device) - dist))
    if not losses:
        return torch.zeros((), dtype=prototypes.dtype, device=prototypes.device)
    return torch.stack(losses).mean()


def _masked_inter_domain_loss(source_prototypes: torch.Tensor, target_prototypes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not bool(valid.any()):
        return torch.zeros((), dtype=source_prototypes.dtype, device=source_prototypes.device)
    return torch.linalg.vector_norm(source_prototypes[valid] - target_prototypes[valid], ord=2, dim=1).mean()


def _running_combined_loss(source_proto, source_valid, target_proto, target_valid, global_proto, valid_target):
    losses = []
    for cls in range(source_proto.shape[0]):
        if not bool(valid_target[cls]):
            continue
        chunks = []
        if bool(source_valid[cls]):
            chunks.append(source_proto[cls])
        if bool(target_valid[cls]):
            chunks.append(target_proto[cls])
        if chunks:
            combined = torch.stack(chunks).mean(dim=0)
            losses.append(torch.linalg.vector_norm(combined - global_proto[cls], ord=2))
    if not losses:
        return torch.zeros((), dtype=source_proto.dtype, device=source_proto.device)
    return torch.stack(losses).mean()


def _resolve(config: dict, path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return cfg_path({"_base_dir": config["_base_dir"], "path": str(p)}, "path")


if __name__ == "__main__":
    main()
