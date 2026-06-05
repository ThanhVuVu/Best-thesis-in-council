from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_biclassifier_outputs(model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str) -> dict[str, Any]:
    model.to(device)
    model.eval()
    embeddings, y_true, probs, probs1, probs2, predictions, metadata = [], [], [], [], [], [], []
    for batch in tqdm(loader, desc=desc, dynamic_ncols=True, mininterval=1.0):
        if len(batch) == 4:
            x, time_features, y, meta = batch
            metadata.extend(_batch_metadata_to_rows(meta))
        else:
            x, time_features, y = batch
        x = x.to(device, non_blocking=True)
        time_features = time_features.to(device, non_blocking=True)
        out = model(x, time_features, return_all=True)
        p = torch.softmax(out["logits"], dim=1)
        embeddings.append(out["embedding"].detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())
        probs1.append(out["probabilities1"].detach().cpu().numpy())
        probs2.append(out["probabilities2"].detach().cpu().numpy())
        predictions.append(p.argmax(dim=1).detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())
    return {
        "embeddings": np.concatenate(embeddings),
        "y_true": np.concatenate(y_true),
        "probabilities": np.concatenate(probs),
        "probabilities1": np.concatenate(probs1),
        "probabilities2": np.concatenate(probs2),
        "y_pred": np.concatenate(predictions),
        "metadata": metadata,
    }


def compute_prototypes(embeddings: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    emb = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    prototypes = np.zeros((num_classes, emb.shape[1]), dtype=np.float32)
    class_counts = {}
    intra = {}
    for cls in range(num_classes):
        mask = labels == cls
        class_counts[str(cls)] = int(mask.sum())
        if not mask.any():
            intra[str(cls)] = None
            continue
        prototypes[cls] = emb[mask].mean(axis=0)
        distances = np.linalg.norm(emb[mask] - prototypes[cls], axis=1)
        intra[str(cls)] = {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "q95": float(np.quantile(distances, 0.95)),
        }
    pairwise = {}
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            pairwise[f"{i}_{j}"] = float(np.linalg.norm(prototypes[i] - prototypes[j]))
    return prototypes, {"class_counts": class_counts, "intra_class_distance": intra, "pairwise_distance": pairwise}


def compactness_loss(embeddings: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    proto = prototypes.to(embeddings.device, dtype=embeddings.dtype)[labels]
    return torch.linalg.vector_norm(embeddings - proto, ord=2, dim=1).mean()


def separation_loss(prototypes: torch.Tensor, margin: float) -> torch.Tensor:
    losses = []
    for i in range(prototypes.shape[0]):
        for j in range(i + 1, prototypes.shape[0]):
            dist = torch.linalg.vector_norm(prototypes[i] - prototypes[j], ord=2)
            losses.append(torch.relu(torch.as_tensor(margin, device=prototypes.device, dtype=prototypes.dtype) - dist))
    if not losses:
        return torch.zeros((), device=prototypes.device)
    return torch.stack(losses).mean()


def inter_domain_loss(source_prototypes: torch.Tensor, target_prototypes: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if not bool(valid_mask.any()):
        return torch.zeros((), device=source_prototypes.device)
    diff = source_prototypes[valid_mask] - target_prototypes[valid_mask]
    return torch.linalg.vector_norm(diff, ord=2, dim=1).mean()


def _batch_metadata_to_rows(meta: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(meta.keys())
    batch_size = len(meta[keys[0]])
    rows = []
    for i in range(batch_size):
        row = {}
        for key in keys:
            value = meta[key][i]
            if hasattr(value, "item"):
                value = value.item()
            row[key] = value
        rows.append(row)
    return rows

