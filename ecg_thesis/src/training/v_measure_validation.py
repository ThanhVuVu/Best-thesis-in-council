from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_unlabeled_logits(
    loader: DataLoader,
    device: torch.device,
    infer_batch: Callable[[Any], torch.Tensor],
) -> dict[str, np.ndarray]:
    logits: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    offset = 0
    for batch in loader:
        values = infer_batch(batch)
        if values.ndim != 2:
            raise ValueError(f"Expected [N,C] logits, got {tuple(values.shape)}")
        values_cpu = values.detach().to(device="cpu", dtype=torch.float32).numpy()
        logits.append(values_cpu)
        batch_size = len(values_cpu)
        indices.append(np.arange(offset, offset + batch_size, dtype=np.int64))
        offset += batch_size
    if not logits:
        raise ValueError("V-Measure validation loader is empty")
    return {"logits": np.concatenate(logits), "indices": np.concatenate(indices)}


def ericsson_v_measure(
    source_logits: np.ndarray,
    source_labels: np.ndarray,
    target_logits: np.ndarray,
    *,
    num_classes: int,
    random_state: int = 42,
    beta: float = 1.0,
) -> dict[str, Any]:
    source_logits = _matrix(source_logits, "source_logits", num_classes)
    target_logits = _matrix(target_logits, "target_logits", num_classes)
    source_labels = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    if len(source_labels) != len(source_logits):
        raise ValueError("source logits/labels length mismatch")
    combined_logits = np.concatenate((target_logits, source_logits), axis=0)
    target_predictions = target_logits.argmax(axis=1).astype(np.int64)
    reference_labels = np.concatenate((target_predictions, source_labels), axis=0)
    clustering = KMeans(
        n_clusters=int(num_classes), init="k-means++", n_init=10, random_state=int(random_state)
    ).fit(combined_logits)
    cluster_labels = clustering.labels_.astype(np.int64)
    homogeneity = float(homogeneity_score(reference_labels, cluster_labels))
    completeness = float(completeness_score(reference_labels, cluster_labels))
    denominator = beta * homogeneity + completeness
    harmonic = 0.0 if denominator == 0.0 else float((1.0 + beta) * homogeneity * completeness / denominator)
    direct = float(v_measure_score(reference_labels, cluster_labels, beta=beta))
    if not np.isclose(harmonic, direct, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"manual V-Measure {harmonic} != sklearn {direct}")
    active_clusters = int(np.unique(cluster_labels).size)
    return {
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": direct,
        "v_measure_manual": harmonic,
        "valid": active_clusters == int(num_classes),
        "active_clusters": active_clusters,
        "active_reference_classes": int(np.unique(reference_labels).size),
        "num_source_val": int(len(source_logits)),
        "num_target_val": int(len(target_logits)),
        "reference_labels": reference_labels,
        "cluster_labels": cluster_labels,
        "target_predictions": target_predictions,
        "domain": np.asarray(["target_val"] * len(target_logits) + ["source_val"] * len(source_logits)),
        "sample_index": np.concatenate((np.arange(len(target_logits)), np.arange(len(source_logits)))).astype(np.int64),
    }


def save_v_measure_assignments(path: str | Path, result: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        reference_labels=np.asarray(result["reference_labels"], dtype=np.int64),
        cluster_labels=np.asarray(result["cluster_labels"], dtype=np.int64),
        target_predictions=np.asarray(result["target_predictions"], dtype=np.int64),
        domain=np.asarray(result["domain"]),
        sample_index=np.asarray(result["sample_index"], dtype=np.int64),
    )


def aggregate_v_measure(result: dict[str, Any]) -> dict[str, Any]:
    raw = {"reference_labels", "cluster_labels", "target_predictions", "domain", "sample_index"}
    return {key: value for key, value in result.items() if key not in raw}


def _matrix(value: np.ndarray, name: str, num_classes: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != int(num_classes) or len(matrix) == 0:
        raise ValueError(f"{name} must have shape [N,{num_classes}], got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix
