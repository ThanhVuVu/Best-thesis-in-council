from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.metrics import classification_metrics


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_embeddings: bool = False,
) -> dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    probs = []
    embeddings = []
    metadata = []

    for batch in loader:
        if len(batch) == 3:
            x, y, meta = batch
            metadata.extend(_batch_metadata_to_rows(meta))
        else:
            x, y = batch
        x = x.to(device)
        y = y.to(device)
        if collect_embeddings:
            logits, emb = model(x, return_embedding=True)
            embeddings.append(emb.detach().cpu().numpy())
        else:
            logits = model(x)
        p = torch.softmax(logits, dim=1)
        pred = p.argmax(dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())

    result = {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "probabilities": np.concatenate(probs),
        "metadata": metadata,
    }
    if collect_embeddings:
        result["embeddings"] = np.concatenate(embeddings)
    result["metrics"] = classification_metrics(result["y_true"], result["y_pred"])
    return result


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
