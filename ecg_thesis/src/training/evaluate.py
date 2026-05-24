from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import classification_metrics


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_embeddings: bool = False,
    desc: str = "evaluate",
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    model_device = next(model.parameters()).device
    print(f"{desc}: model_device={model_device}, target_device={device}, batches={len(loader)}")
    y_true = []
    y_pred = []
    probs = []
    embeddings = []
    metadata = []

    progress = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True, mininterval=1.0)
    for batch in progress:
        if len(batch) == 3 and isinstance(batch[2], dict):
            x, y, meta = batch
            metadata.extend(_batch_metadata_to_rows(meta))
            inputs = (x.to(device, non_blocking=True),)
        elif len(batch) == 4:
            x, rr, y, meta = batch
            metadata.extend(_batch_metadata_to_rows(meta))
            inputs = (x.to(device, non_blocking=True), rr.to(device, non_blocking=True))
        elif len(batch) == 3:
            x, rr, y = batch
            inputs = (x.to(device, non_blocking=True), rr.to(device, non_blocking=True))
        else:
            x, y = batch
            inputs = (x.to(device, non_blocking=True),)
        y = y.to(device, non_blocking=True)
        if collect_embeddings:
            logits, emb = model(*inputs, return_embedding=True)
            embeddings.append(emb.detach().cpu().numpy())
        else:
            logits = model(*inputs)
        p = torch.softmax(logits, dim=1)
        pred = p.argmax(dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())
        progress.set_postfix(batch_size=int(inputs[0].shape[0]), device=str(inputs[0].device), refresh=False)

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
