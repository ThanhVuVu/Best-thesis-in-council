from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)


def entropy_from_probs(probs: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    return -(values * np.log(np.clip(values, eps, 1.0))).sum(axis=1)


def confidence_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs, dtype=np.float64).max(axis=1)


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict[str, float]]]:
    values = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    confidence = confidence_from_probs(values)
    predicted = values.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    for idx in range(int(n_bins)):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == int(n_bins) - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_start": float(lo), "bin_end": float(hi), "count": 0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(acc - conf)
        ece += float(count / len(labels)) * gap
        rows.append({"bin_start": float(lo), "bin_end": float(hi), "count": count, "accuracy": acc, "confidence": conf, "gap": gap})
    return float(ece), rows


def classwise_predicted_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    n_bins: int = 15,
) -> dict[str, Any]:
    predicted = np.asarray(probs).argmax(axis=1)
    out: dict[str, Any] = {}
    for cls, name in enumerate(class_names):
        mask = predicted == cls
        if not mask.any():
            out[name] = {"count": 0, "ece": None}
            continue
        ece, rows = expected_calibration_error(np.asarray(probs)[mask], np.asarray(y_true)[mask], n_bins=n_bins)
        out[name] = {"count": int(mask.sum()), "ece": ece, "bins": rows}
    return out


def error_pair_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    entropy = entropy_from_probs(probs)
    confidence = confidence_from_probs(probs)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        if int(true) != int(pred):
            groups[(int(true), int(pred))].append(idx)
    rows = []
    for (true, pred), indices in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        idx_arr = np.asarray(indices, dtype=np.int64)
        rows.append(
            {
                "true_class": class_names[true],
                "pred_class": class_names[pred],
                "count": int(len(indices)),
                "avg_confidence": float(confidence[idx_arr].mean()),
                "avg_entropy": float(entropy[idx_arr].mean()),
                **{f"avg_prob_{name}": float(np.asarray(probs)[idx_arr, cls].mean()) for cls, name in enumerate(class_names)},
            }
        )
    return rows


def per_record_rows(
    records: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    by_record: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_record[str(record)].append(idx)
    rows = []
    confidence = confidence_from_probs(probs)
    for record, indices in sorted(by_record.items()):
        idx_arr = np.asarray(indices, dtype=np.int64)
        yt = np.asarray(y_true)[idx_arr]
        yp = np.asarray(y_pred)[idx_arr]
        row: dict[str, Any] = {
            "record": record,
            "support": int(len(idx_arr)),
            "accuracy": float((yt == yp).mean()),
            "avg_confidence": float(confidence[idx_arr].mean()),
        }
        f1_values = []
        for cls, name in enumerate(class_names):
            tp = int(((yt == cls) & (yp == cls)).sum())
            fp = int(((yt != cls) & (yp == cls)).sum())
            fn = int(((yt == cls) & (yp != cls)).sum())
            support = int((yt == cls).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1.0e-12) if support > 0 or fp > 0 else 0.0
            if support > 0:
                f1_values.append(f1)
            row[f"{name}_support"] = support
            row[f"{name}_f1"] = float(f1)
        row["macro_f1_present_classes"] = float(np.mean(f1_values)) if f1_values else 0.0
        rows.append(row)
    return rows


def threshold_sweep_rows(
    y_true: np.ndarray,
    scores: np.ndarray,
    positive_class: int,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    labels = (np.asarray(y_true, dtype=np.int64) == int(positive_class)).astype(np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for threshold in thresholds:
        pred = values >= float(threshold)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
        rows.append({"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1})
    return rows


def pr_roc_summary(y_true: np.ndarray, scores: np.ndarray, positive_class: int) -> dict[str, Any]:
    labels = (np.asarray(y_true, dtype=np.int64) == int(positive_class)).astype(np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auprc": None, "auroc": None, "pr_curve": [], "roc_curve": []}
    precision, recall, pr_thresholds = precision_recall_curve(labels, values)
    fpr, tpr, roc_thresholds = roc_curve(labels, values)
    return {
        "auprc": float(average_precision_score(labels, values)),
        "auroc": float(roc_auc_score(labels, values)),
        "pr_curve": [
            {"precision": float(p), "recall": float(r), "threshold": float(pr_thresholds[i]) if i < len(pr_thresholds) else None}
            for i, (p, r) in enumerate(zip(precision, recall))
        ],
        "roc_curve": [
            {"fpr": float(f), "tpr": float(t), "threshold": float(th)}
            for f, t, th in zip(fpr, tpr, roc_thresholds)
        ],
    }


def centroid_distance_matrix(features: np.ndarray, labels: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, list[str]]:
    centers = []
    names = []
    values = np.asarray(features, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    for cls, name in enumerate(class_names):
        mask = labels_arr == cls
        if mask.any():
            centers.append(values[mask].mean(axis=0))
            names.append(name)
    if not centers:
        return np.zeros((0, 0), dtype=np.float64), []
    centers_arr = np.stack(centers)
    diff = centers_arr[:, None, :] - centers_arr[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1)), names


def safe_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float | None:
    labels_arr = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels_arr)) < 2 or len(labels_arr) < 3:
        return None
    counts = Counter(labels_arr.tolist())
    if min(counts.values()) < 2:
        return None
    return float(silhouette_score(np.asarray(features), labels_arr))
