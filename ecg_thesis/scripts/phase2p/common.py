from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import cfg_path, device_from_torch, load_phase1_config  # noqa: E402
from src.data.datasets import ECGBeatTimeDataset  # noqa: E402
from src.data.splits import mitbih_fit_val_records  # noqa: E402
from src.models.catnet_biclassifier import CATNetBiClassifier  # noqa: E402
from src.training.metrics import classification_metrics  # noqa: E402
from src.training.train import compute_class_weights  # noqa: E402
from src.utils.io import ensure_dir, write_json  # noqa: E402


def model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "d_model",
        "num_heads",
        "dff",
        "num_transformer_layers",
        "attention_reduction",
        "dropout",
        "time_feature_dim",
        "classifier_hidden_dim",
        "classifier_dropout",
    }
    return {key: config["model"][key] for key in allowed if key in config["model"]}


def build_phase2p_model(config: dict[str, Any], device: torch.device) -> CATNetBiClassifier:
    model = CATNetBiClassifier(num_classes=int(config["data"]["num_classes"]), **model_kwargs(config))
    return model.to(device)


def load_phase2p_checkpoint(path: str | Path, config: dict[str, Any], device: torch.device) -> tuple[CATNetBiClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model = build_phase2p_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def fit_val_datasets(config: dict[str, Any], use_duplicated: bool = False):
    source_path = cfg_path(config, "data", "source_train_duplicated" if use_duplicated else "source_train")
    full = ECGBeatTimeDataset(source_path)
    if use_duplicated:
        records = np.asarray([str(r) for r in full.records])
        fit_records, val_records = mitbih_fit_val_records()
        fit_idx = [i for i, rec in enumerate(records) if rec in set(fit_records)]
        source_val = ECGBeatTimeDataset(cfg_path(config, "data", "source_train"))
        val_records_arr = np.asarray([str(r) for r in source_val.records])
        val_idx = [i for i, rec in enumerate(val_records_arr) if rec in set(val_records)]
        return Subset(full, fit_idx), Subset(source_val, val_idx)
    fit_records, val_records = mitbih_fit_val_records()
    records = np.asarray([str(r) for r in full.records])
    fit_idx = [i for i, rec in enumerate(records) if rec in set(fit_records)]
    val_idx = [i for i, rec in enumerate(records) if rec in set(val_records)]
    return Subset(full, fit_idx), Subset(full, val_idx)


def maybe_subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(int(max_samples), len(dataset)))))


def dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent = dataset_labels(dataset.dataset)
        return parent[np.asarray(dataset.indices)]
    return dataset.y


def loader(dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")


def batch_to_device(batch, device: torch.device):
    if len(batch) == 4:
        x, time_features, y, meta = batch
        return x.to(device), time_features.to(device), y.to(device), meta
    x, time_features, y = batch
    return x.to(device), time_features.to(device), y.to(device), None


@torch.no_grad()
def evaluate_model(model, dataset, device: torch.device, batch_size: int = 128, max_samples: int | None = None):
    ds = maybe_subset(dataset, max_samples)
    dl = loader(ds, batch_size, False, device)
    model.to(device)
    model.eval()
    y_true, y_pred, probs, rows = [], [], [], []
    for batch in dl:
        x, time_features, y, meta = batch_to_device(batch, device)
        out = model(x, time_features, return_all=True)
        p = torch.softmax(out["logits"], dim=1)
        pred = p.argmax(dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        probs.append(p.cpu().numpy())
        if meta is not None:
            rows.extend(_batch_metadata_to_rows(meta))
    result = {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "probabilities": np.concatenate(probs),
        "metadata": rows,
    }
    result["metrics"] = classification_metrics(result["y_true"], result["y_pred"])
    return result


def save_predictions(result: dict[str, Any], path: str | Path, class_names: list[str]) -> None:
    rows = result.get("metadata") or [{} for _ in range(len(result["y_true"]))]
    probs = result["probabilities"]
    out_rows = []
    for i, row in enumerate(rows):
        item = dict(row)
        true = int(result["y_true"][i])
        pred = int(result["y_pred"][i])
        item.update({"true": true, "pred": pred, "true_class": class_names[true], "pred_class": class_names[pred]})
        for cls, name in enumerate(class_names):
            item[f"prob_{name}"] = float(probs[i, cls])
        out_rows.append(item)
    ensure_dir(Path(path).parent)
    pd.DataFrame(out_rows).to_csv(path, index=False)


def save_checkpoint(model, optimizer, config, epoch: int, best_f1: float, history: list[dict[str, Any]], path: str | Path) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "model_name": "catnet_biclassifier",
        "epoch": int(epoch),
        "best_macro_f1": float(best_f1),
        "history": history,
        "model_state_fingerprint": state_dict_fingerprint(model.state_dict()),
    }
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def write_history(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def class_weights_for(dataset, config, device):
    if not config.get("use_class_weights", True):
        return None
    return compute_class_weights(dataset_labels(dataset), int(config.get("num_classes", 3))).to(device)


def state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]


def write_eval_outputs(result: dict[str, Any], output: Path, name: str, class_names: list[str]) -> None:
    write_json(result["metrics"], output / "metrics" / f"{name}_metrics.json")
    save_predictions(result, output / "predictions" / f"{name}_predictions.csv", class_names)


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

