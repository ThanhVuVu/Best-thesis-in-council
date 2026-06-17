from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


EPS = 1.0e-12


@dataclass(frozen=True)
class ClassPair:
    negative: int
    positive: int
    negative_name: str
    positive_name: str


def effect_size_rows(features: dict[str, np.ndarray], y: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_name, values in features.items():
        values = np.asarray(values, dtype=np.float64)
        for cls in range(1, len(class_names)):
            n_values = values[y == 0]
            c_values = values[y == cls]
            if len(n_values) == 0 or len(c_values) == 0:
                continue
            rows.append(
                {
                    "feature": feature_name,
                    "class_pair": f"{class_names[cls]}_vs_N",
                    "minority_class": class_names[cls],
                    "n_mean": float(np.mean(n_values)),
                    "minority_mean": float(np.mean(c_values)),
                    "mean_diff": float(np.mean(c_values) - np.mean(n_values)),
                    "cohen_d": float(cohen_d(c_values, n_values)),
                    "ks_statistic": float(ks_2samp(c_values, n_values).statistic),
                    "n_support": int(len(n_values)),
                    "minority_support": int(len(c_values)),
                }
            )
    return rows


def temporal_contrast_rows(windows: np.ndarray, y: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected windows with shape [N, T], got {values.shape}")
    n_values = values[y == 0]
    if len(n_values) == 0:
        return rows
    n_mean = n_values.mean(axis=0)
    for cls in range(1, len(class_names)):
        c_values = values[y == cls]
        if len(c_values) == 0:
            continue
        c_mean = c_values.mean(axis=0)
        c_std = c_values.std(axis=0)
        n_std = n_values.std(axis=0)
        pooled = np.sqrt((c_std**2 + n_std**2) / 2.0)
        d_curve = (c_mean - n_mean) / np.maximum(pooled, EPS)
        for idx in range(values.shape[1]):
            rows.append(
                {
                    "minority_class": class_names[cls],
                    "class_pair": f"{class_names[cls]}_vs_N",
                    "time_index": int(idx),
                    "minority_mean": float(c_mean[idx]),
                    "n_mean": float(n_mean[idx]),
                    "abs_mean_diff": float(abs(c_mean[idx] - n_mean[idx])),
                    "cohen_d": float(d_curve[idx]),
                }
            )
    return rows


def pairwise_separability_rows(features_by_layer: dict[str, np.ndarray], y: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, features in features_by_layer.items():
        x = _finite_2d(features)
        centers = []
        scatters = []
        for cls in range(len(class_names)):
            cls_x = x[y == cls]
            if len(cls_x) == 0:
                centers.append(None)
                scatters.append(np.nan)
            else:
                center = cls_x.mean(axis=0)
                centers.append(center)
                scatters.append(float(np.linalg.norm(cls_x - center, axis=1).mean()))
        n_center = centers[0]
        if n_center is None:
            continue
        for cls in range(1, len(class_names)):
            c_center = centers[cls]
            if c_center is None:
                continue
            distance = float(np.linalg.norm(c_center - n_center))
            within = float(np.nanmean([scatters[0], scatters[cls]]))
            rows.append(
                {
                    "layer": layer,
                    "class_pair": f"{class_names[cls]}_vs_N",
                    "minority_class": class_names[cls],
                    "centroid_distance_to_N": distance,
                    "within_scatter_mean": within,
                    "fisher_ratio_proxy": float(distance / max(within, EPS)),
                    "n_support": int(np.sum(y == 0)),
                    "minority_support": int(np.sum(y == cls)),
                }
            )
    return rows


def knn_purity_rows(features_by_layer: dict[str, np.ndarray], y: np.ndarray, class_names: list[str], k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(y) <= 1:
        return rows
    n_neighbors = min(max(int(k) + 1, 2), len(y))
    for layer, features in features_by_layer.items():
        x = StandardScaler().fit_transform(_finite_2d(features))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(x)
        indices = nbrs.kneighbors(x, return_distance=False)[:, 1:]
        neighbor_labels = y[indices]
        for cls, name in enumerate(class_names):
            mask = y == cls
            if not mask.any():
                continue
            same = (neighbor_labels[mask] == cls).mean(axis=1)
            n_fraction = (neighbor_labels[mask] == 0).mean(axis=1)
            rows.append(
                {
                    "layer": layer,
                    "class_name": name,
                    "knn_k": int(n_neighbors - 1),
                    "support": int(mask.sum()),
                    "same_class_purity": float(np.mean(same)),
                    "n_neighbor_fraction": float(np.mean(n_fraction)),
                }
            )
    return rows


def linear_probe_rows(
    features_by_layer: dict[str, np.ndarray],
    y: np.ndarray,
    class_names: list[str],
    test_size: float,
    random_seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(np.unique(y)) < 2:
        return rows
    stratify = y if min(np.bincount(y, minlength=len(class_names))) >= 2 else None
    for layer, features in features_by_layer.items():
        x = StandardScaler().fit_transform(_finite_2d(features))
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=float(test_size),
                random_state=int(random_seed),
                stratify=stratify,
            )
            clf = LogisticRegression(max_iter=int(max_iter), class_weight="balanced", multi_class="auto")
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
        except Exception as exc:
            rows.append({"layer": layer, "error": repr(exc)})
            continue
        recalls = recall_score(y_test, pred, labels=list(range(len(class_names))), average=None, zero_division=0)
        rows.append(
            {
                "layer": layer,
                "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                **{f"{name}_recall": float(recalls[idx]) for idx, name in enumerate(class_names)},
                "test_support": int(len(y_test)),
            }
        )
    return rows


def nearest_neighbor_rows(
    features: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    metadata: list[dict[str, Any]],
    class_names: list[str],
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(y) <= 1:
        return rows
    x = StandardScaler().fit_transform(_finite_2d(features))
    n_mask = y == 0
    same_by_class = {cls: np.where(y == cls)[0] for cls in range(len(class_names))}
    n_indices = np.where(n_mask)[0]
    n_model = NearestNeighbors(n_neighbors=min(int(k), len(n_indices)), metric="euclidean").fit(x[n_indices]) if len(n_indices) else None
    same_models = {
        cls: NearestNeighbors(n_neighbors=min(int(k), len(indices)), metric="euclidean").fit(x[indices])
        for cls, indices in same_by_class.items()
        if len(indices)
    }
    error_indices = np.where((y != 0) & (y_pred == 0))[0]
    for query_idx in error_indices:
        query_cls = int(y[query_idx])
        for group, model, source_indices in (
            ("nearest_N", n_model, n_indices),
            ("nearest_same_class", same_models.get(query_cls), same_by_class.get(query_cls, np.zeros(0, dtype=int))),
        ):
            if model is None or len(source_indices) == 0:
                continue
            distances, local_idx = model.kneighbors(x[query_idx : query_idx + 1], return_distance=True)
            for rank, (distance, local) in enumerate(zip(distances[0], local_idx[0]), start=1):
                neighbor_idx = int(source_indices[int(local)])
                if neighbor_idx == int(query_idx):
                    continue
                rows.append(
                    {
                        "query_index": int(query_idx),
                        "query_true": class_names[query_cls],
                        "query_pred": class_names[int(y_pred[query_idx])],
                        "neighbor_group": group,
                        "neighbor_rank": int(rank),
                        "neighbor_index": neighbor_idx,
                        "neighbor_true": class_names[int(y[neighbor_idx])],
                        "distance": float(distance),
                        "query_record": metadata_value(metadata, query_idx, "record"),
                        "query_symbol": metadata_value(metadata, query_idx, "symbol"),
                        "neighbor_record": metadata_value(metadata, neighbor_idx, "record"),
                        "neighbor_symbol": metadata_value(metadata, neighbor_idx, "symbol"),
                    }
                )
    return rows


def metadata_value(metadata: list[dict[str, Any]], idx: int, key: str) -> Any:
    if idx >= len(metadata):
        return ""
    return metadata[idx].get(key, "")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0) if len(a) > 1 and len(b) > 1 else 0.0
    return float((a.mean() - b.mean()) / max(pooled, EPS))


def _finite_2d(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim > 2:
        x = x.reshape((x.shape[0], -1))
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got {x.shape}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
