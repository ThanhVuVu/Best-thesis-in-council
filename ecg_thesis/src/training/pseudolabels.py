from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def select_confident_pseudolabels(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    probabilities1: np.ndarray,
    probabilities2: np.ndarray,
    source_prototypes: np.ndarray,
    source_stats: dict[str, Any],
    confidence_thresholds: dict[str, float],
    class_names: list[str],
    distance_quantile_key: str = "q95",
    min_target_per_class: int = 20,
    metadata: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    discrepancy = np.linalg.norm(probabilities1 - probabilities2, axis=1)
    source_mean_discrepancy = float(source_stats.get("mean_classifier_discrepancy", np.inf))
    distances = np.linalg.norm(embeddings - source_prototypes[pred], axis=1)
    selected = np.zeros(len(pred), dtype=bool)

    rows = []
    for idx in range(len(pred)):
        cls = int(pred[idx])
        name = class_names[cls]
        threshold = float(confidence_thresholds.get(name, 0.99))
        intra = source_stats["intra_class_distance"].get(str(cls)) or {}
        distance_threshold = float(intra.get(distance_quantile_key, np.inf))
        ok = bool(confidence[idx] >= threshold and distances[idx] <= distance_threshold and discrepancy[idx] <= source_mean_discrepancy)
        selected[idx] = ok
        row = {
            "index": int(idx),
            "pseudo_label": cls,
            "pseudo_class": name,
            "confidence": float(confidence[idx]),
            "distance_to_source_prototype": float(distances[idx]),
            "classifier_discrepancy": float(discrepancy[idx]),
            "selected": ok,
        }
        if metadata and idx < len(metadata):
            row.update(metadata[idx])
        rows.append(row)
    df = pd.DataFrame(rows)
    stats = {
        "total": int(len(df)),
        "selected_total": int(selected.sum()),
        "mean_classifier_discrepancy": float(discrepancy.mean()) if len(discrepancy) else None,
        "source_mean_classifier_discrepancy_threshold": source_mean_discrepancy,
        "per_class": {},
        "classes_below_min_target": [],
    }
    for cls, name in enumerate(class_names):
        class_mask = pred == cls
        selected_mask = selected & class_mask
        count = int(selected_mask.sum())
        stats["per_class"][name] = {
            "predicted": int(class_mask.sum()),
            "selected": count,
            "mean_confidence_selected": float(confidence[selected_mask].mean()) if count else None,
            "median_confidence_selected": float(np.median(confidence[selected_mask])) if count else None,
            "mean_distance_selected": float(distances[selected_mask].mean()) if count else None,
        }
        if count < int(min_target_per_class):
            stats["classes_below_min_target"].append(name)
    return df, selected, stats

