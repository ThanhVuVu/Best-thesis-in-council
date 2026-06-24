from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.daeac_preprocess import CLASS_NAMES, map_symbol_daeac
from src.data.physionet import read_annotation


def record_class_counts(raw_dir: str | Path, records: list[str]) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for record in sorted(map(str, records)):
        labels = [map_symbol_daeac(symbol) for symbol in read_annotation(raw_dir, record).symbol]
        counts[record] = np.bincount([x for x in labels if x is not None], minlength=len(CLASS_NAMES)).astype(int).tolist()
    return counts


def balanced_record_split(
    counts: dict[str, list[int]], split_sizes: dict[str, int], *, seed: int = 42, trials: int = 10000
) -> dict[str, list[str]]:
    records = np.asarray(sorted(counts), dtype=object)
    if sum(split_sizes.values()) != len(records):
        raise ValueError("split sizes must cover every record exactly once")
    matrix = np.asarray([counts[str(record)] for record in records], dtype=np.float64)
    total = matrix.sum(axis=0)
    rng = np.random.default_rng(seed)
    split_names = list(split_sizes)
    target_fractions = np.asarray([split_sizes[name] / len(records) for name in split_names])
    best_score = float("inf")
    best: dict[str, list[str]] | None = None
    for _ in range(max(1, int(trials))):
        order = rng.permutation(len(records))
        candidate: dict[str, list[str]] = {}
        offset = 0
        score = 0.0
        for idx, name in enumerate(split_names):
            chosen = order[offset : offset + split_sizes[name]]
            offset += split_sizes[name]
            candidate[name] = sorted(str(records[i]) for i in chosen)
            observed = matrix[chosen].sum(axis=0)
            expected = total * target_fractions[idx]
            score += float(np.mean(np.abs(observed - expected) / np.maximum(expected, 1.0)))
            if name != "train":
                score += float(np.count_nonzero((total > 0) & (observed == 0))) * 10.0
        signature = tuple(tuple(candidate[name]) for name in split_names)
        best_signature = tuple(tuple(best[name]) for name in split_names) if best else None
        if score < best_score or (score == best_score and (best_signature is None or signature < best_signature)):
            best_score, best = score, candidate
    assert best is not None
    return best


def audit_record_split(
    counts: dict[str, list[int]], splits: dict[str, list[str]], expected_sizes: dict[str, int]
) -> dict[str, Any]:
    flattened = [record for values in splits.values() for record in values]
    overlaps = sorted({record for record in flattened if flattened.count(record) > 1})
    missing = sorted(set(counts) - set(flattened))
    extra = sorted(set(flattened) - set(counts))
    result: dict[str, Any] = {
        "valid": not overlaps and not missing and not extra,
        "record_overlap": overlaps,
        "missing_records": missing,
        "extra_records": extra,
        "splits": {},
    }
    for name, records in splits.items():
        class_counts = np.asarray([counts[record] for record in records], dtype=np.int64).sum(axis=0)
        result["splits"][name] = {
            "records": list(records),
            "num_records": len(records),
            "expected_records": int(expected_sizes[name]),
            "class_counts": {cls: int(class_counts[i]) for i, cls in enumerate(CLASS_NAMES)},
        }
        result["valid"] = result["valid"] and len(records) == int(expected_sizes[name])
    return result


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
