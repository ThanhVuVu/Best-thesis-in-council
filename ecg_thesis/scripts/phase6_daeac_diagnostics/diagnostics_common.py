from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, resolve_path


def load_diagnostics_config(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    diag = load_phase1_config(path)
    base_config = resolve_path(diag["analysis"]["base_config"], diag["_base_dir"])
    base = load_phase1_config(str(base_config))
    return diag, base


def method_name(config: dict[str, Any], arg_method: str | None = None) -> str:
    return str(arg_method or config["analysis"].get("method_name", "daeac_diagnostics"))


def checkpoint_path(config: dict[str, Any], arg_checkpoint: str | None = None) -> Path:
    value = arg_checkpoint or config["analysis"]["default_checkpoint"]
    return resolve_path(value, config["_base_dir"])


def output_dir(config: dict[str, Any]) -> Path:
    return ensure_dir(cfg_path(config, "paths", "output_dir"))


def selected_datasets(base_config: dict[str, Any], dataset: str, configured: list[str] | None = None) -> list[tuple[str, Path]]:
    external = dict(base_config.get("data", {}).get("external_targets", {}))
    selected: list[tuple[str, Path]] = []
    if dataset == "configured":
        dataset_names = configured or ["source_eval", "target_test", *external.keys()]
        for name in dataset_names:
            selected.extend(selected_datasets(base_config, name, configured=None))
        return selected
    if dataset in {"source", "source_eval", "both", "all"}:
        selected.append(("source_eval", cfg_path(base_config, "data", "source_eval")))
    if dataset in {"target", "target_test", "both", "all"}:
        selected.append(("target_test", cfg_path(base_config, "data", "target_test")))
    if dataset in {"external", "all"}:
        selected.extend((name, _resolve_base_path(base_config, value)) for name, value in external.items())
    elif dataset in external:
        selected.append((dataset, _resolve_base_path(base_config, external[dataset])))
    if not selected:
        valid = ["source", "source_eval", "target", "target_test", "both", "external", "all", *external.keys()]
        raise ValueError(f"Unknown dataset '{dataset}'. Valid values: {valid}")
    deduped = []
    seen = set()
    for name, path in selected:
        if name not in seen:
            deduped.append((name, path))
            seen.add(name)
    return deduped


def prediction_path(config: dict[str, Any], method: str, dataset: str) -> Path:
    return output_dir(config) / "predictions" / f"{method}_{dataset}_predictions.csv"


def embedding_path(config: dict[str, Any], method: str, dataset: str) -> Path:
    return output_dir(config) / "embeddings" / f"{method}_{dataset}_embeddings.npz"


def read_predictions(path: Path, class_names: list[str]) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    probs = np.asarray([[float(row[f"prob_{name}"]) for name in class_names] for row in rows], dtype=np.float64)
    records = [row.get("record", "") for row in rows]
    return {"rows": rows, "y_true": y_true, "y_pred": y_pred, "probs": probs, "records": records}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def class_names_from(base_config: dict[str, Any]) -> list[str]:
    return list(base_config["data"]["class_names"])


def _resolve_base_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path
