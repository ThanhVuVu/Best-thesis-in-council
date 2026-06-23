from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.data.splits import MITBIH_TEST_RECORDS, mitbih_fit_val_records

PAPER_CLASS_NAMES = ["N", "S", "V", "F"]
REFERENCE_REPO_CLASS_NAMES = ["N", "V", "S", "F"]
AUTO_INPUT_KEYS = ("x_daeac", "x_macnn", "x", "X", "inputs", "data", "samples", "beats")


class DAEACDataset(Dataset):
    """Paper-faithful DAEAC dataset for preprocessed [N, 1, 3, 128] arrays."""

    def __init__(
        self,
        npz_path: str | Path,
        input_key: str = "auto",
        label_key: str = "y",
        require_labels: bool = True,
        return_index: bool = False,
        return_metadata: bool = False,
        class_names: list[str] | None = None,
    ):
        self.path = Path(npz_path)
        requested_input_key = str(input_key)
        self.label_key = str(label_key)
        self.require_labels = bool(require_labels)
        self.return_index = bool(return_index)
        self.return_metadata = bool(return_metadata)
        self.class_names = list(class_names or PAPER_CLASS_NAMES)

        self.data = np.load(self.path, allow_pickle=True)
        self.input_key = _resolve_input_key(self.data, self.path, requested_input_key)
        self.x = _normalize_input_array(self.data[self.input_key].astype(np.float32), self.path, self.input_key)
        _validate_input_shape(self.x, self.path, self.input_key)

        self.y: np.ndarray | None = None
        if self.label_key in self.data:
            self.y = self.data[self.label_key].astype(np.int64)
            if len(self.y) != len(self.x):
                raise ValueError(f"{self.path}: {self.input_key}/y length mismatch: {len(self.x)} vs {len(self.y)}")
            _validate_labels(self.y, self.path, self.class_names)
        elif self.require_labels:
            raise KeyError(f"{self.path} does not contain required label key '{self.label_key}'.")

        _validate_class_names(self.data, self.path, self.class_names)

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx])
        values: list[Any] = [x]
        if self.y is not None and self.require_labels:
            values.append(torch.tensor(int(self.y[idx]), dtype=torch.long))
        if self.return_index:
            values.append(torch.tensor(int(idx), dtype=torch.long))
        if self.return_metadata:
            values.append(self.metadata(idx))
        if len(values) == 1:
            return values[0]
        return tuple(values)

    def metadata(self, idx: int) -> dict[str, Any]:
        fields = [
            "record",
            "record_id",
            "symbol",
            "sample",
            "r_peak_sample",
            "r_peak_time_sec",
            "fs",
            "domain",
            "lead_index",
            "lead_name",
        ]
        rows: dict[str, Any] = {}
        for field in fields:
            if field not in self.data:
                continue
            value = self.data[field][idx]
            rows[field] = value.item() if hasattr(value, "item") else value
        return rows

    @property
    def records(self) -> np.ndarray | None:
        if "record" in self.data:
            return self.data["record"]
        if "record_id" in self.data:
            return self.data["record_id"]
        return None

    def close(self) -> None:
        self.data.close()


class DAEACTargetUnlabeledDataset(DAEACDataset):
    """Target adaptation dataset that intentionally never exposes target labels."""

    def __init__(
        self,
        npz_path: str | Path,
        input_key: str = "auto",
        label_key: str = "y",
        return_index: bool = True,
        class_names: list[str] | None = None,
    ):
        super().__init__(
            npz_path,
            input_key=input_key,
            label_key=label_key,
            require_labels=False,
            return_index=return_index,
            return_metadata=False,
            class_names=class_names,
        )


def subset_first(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    n = min(int(max_samples), len(dataset))
    return Subset(dataset, list(range(n)))


def load_daeac_source_fit_val(
    source_path: str | Path,
    eval_path: str | Path,
    input_key: str = "auto",
    label_key: str = "y",
    class_names: list[str] | None = None,
    split_same_path: bool = True,
    full_source_fit: bool = True,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    source_ds = DAEACDataset(source_path, input_key=input_key, label_key=label_key, class_names=class_names)
    same_path = Path(source_path).resolve() == Path(eval_path).resolve()
    if same_path and split_same_path:
        fit_ds, val_ds, split_summary = split_daeac_source_fit_val(source_ds)
        if full_source_fit:
            all_records = sorted(set(np.asarray(source_ds.records).astype(str))) if source_ds.records is not None else []
            summary = {
                **split_summary,
                "mode": "full_source_fit_with_overlapping_monitor_subset",
                "fit_records": all_records,
                "fit_samples": len(source_ds),
                "source_monitor_overlaps_fit": True,
                "source_path": str(source_path),
                "eval_path": str(eval_path),
                "split_applied": True,
            }
            return source_ds, val_ds, summary
        summary = {
            **split_summary,
            "mode": "disjoint_record_fit_validation",
            "source_monitor_overlaps_fit": False,
            "source_path": str(source_path),
            "eval_path": str(eval_path),
            "split_applied": True,
        }
        return fit_ds, val_ds, summary

    val_ds = DAEACDataset(eval_path, input_key=input_key, label_key=label_key, class_names=class_names)
    summary = {
        "source_path": str(source_path),
        "eval_path": str(eval_path),
        "split_applied": False,
        "same_path": same_path,
        "fit_samples": len(source_ds),
        "val_samples": len(val_ds),
    }
    return source_ds, val_ds, summary


def split_daeac_source_fit_val(dataset: DAEACDataset) -> tuple[Subset, Subset, dict[str, Any]]:
    records = dataset.records
    if records is None:
        raise ValueError("DAEAC source dataset has no record metadata; cannot create record-wise train/validation split.")

    record_strings = np.asarray([str(value) for value in records])
    fit_records, val_records = mitbih_fit_val_records()
    # For the MITBIH source-pool experiments, DS1+DS2 is the source corpus.
    # Keep the four-record DS1 validation set and add every DS2 record to fit.
    present_set = set(record_strings)
    ds2_present = [record for record in MITBIH_TEST_RECORDS if record in present_set]
    if ds2_present:
        fit_records = [*fit_records, *ds2_present]
    fit_set = set(fit_records)
    val_set = set(val_records)
    fit_idx = [idx for idx, rec in enumerate(record_strings) if rec in fit_set]
    val_idx = [idx for idx, rec in enumerate(record_strings) if rec in val_set]
    if not fit_idx or not val_idx:
        present = sorted(set(record_strings))
        raise ValueError(
            "DAEAC source fit/validation split is empty. "
            f"Expected fit records={fit_records}, val records={val_records}, present records={present}."
        )

    fit_present = set(record_strings[fit_idx])
    val_present = set(record_strings[val_idx])
    overlap = sorted(fit_present & val_present)
    if overlap:
        raise ValueError(f"Record overlap between DAEAC source fit and validation splits: {overlap}")

    summary = {
        "mode": "mitbih_fit_val_records",
        "fit_records": fit_records,
        "val_records": val_records,
        "fit_samples": len(fit_idx),
        "val_samples": len(val_idx),
        "record_overlap": overlap,
    }
    return Subset(dataset, fit_idx), Subset(dataset, val_idx), summary


class DAEACPseudoLabeledDataset(Dataset):
    """Immutable per-epoch view of confidently pseudo-labeled target samples."""

    def __init__(
        self,
        target_dataset: Dataset,
        positions: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor | None = None,
        normalized_entropy: torch.Tensor | None = None,
    ):
        self.target_dataset = target_dataset
        self.positions = torch.as_tensor(positions, dtype=torch.long).cpu()
        self.labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        self.confidence = _optional_snapshot_tensor(confidence, len(self.labels), default=1.0)
        self.normalized_entropy = _optional_snapshot_tensor(normalized_entropy, len(self.labels), default=0.0)
        if len(self.positions) != len(self.labels):
            raise ValueError("Pseudo-label positions and labels must have equal length.")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        item = self.target_dataset[int(self.positions[idx])]
        x = item[0] if isinstance(item, (tuple, list)) else item
        return x, self.labels[idx], self.confidence[idx], self.normalized_entropy[idx]


def _optional_snapshot_tensor(value: torch.Tensor | None, length: int, default: float) -> torch.Tensor:
    if value is None:
        return torch.full((length,), float(default), dtype=torch.float32)
    result = torch.as_tensor(value, dtype=torch.float32).cpu()
    if len(result) != length:
        raise ValueError("Pseudo-label metadata must have the same length as labels.")
    return result


def class_counts_from_dataset(dataset: DAEACDataset, num_classes: int = 4) -> np.ndarray:
    if dataset.y is None:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount(dataset.y.astype(np.int64), minlength=num_classes)


def inspect_daeac_npz(
    npz_path: str | Path,
    input_key: str = "auto",
    label_key: str = "y",
    class_names: list[str] | None = None,
    require_labels: bool = True,
) -> dict[str, Any]:
    ds = DAEACDataset(
        npz_path,
        input_key=input_key,
        label_key=label_key,
        require_labels=require_labels,
        class_names=class_names,
    )
    counts = class_counts_from_dataset(ds, len(ds.class_names)).astype(int).tolist()
    records = ds.records
    return {
        "path": str(ds.path),
        "input_key": ds.input_key,
        "shape": list(ds.x.shape),
        "has_labels": ds.y is not None,
        "class_names": ds.class_names,
        "class_counts": {name: counts[idx] for idx, name in enumerate(ds.class_names)},
        "num_records": int(len(set(records.astype(str)))) if records is not None else None,
        "num_samples": int(len(ds)),
    }


def _resolve_input_key(data: np.lib.npyio.NpzFile, path: Path, requested: str) -> str:
    if requested != "auto" and requested in data:
        return requested

    detected = _detect_input_key(data)
    if detected is not None:
        return detected

    available = ", ".join(data.files)
    candidates = ", ".join(AUTO_INPUT_KEYS)
    if requested == "auto":
        raise KeyError(f"{path}: could not auto-detect DAEAC input key. Available keys: [{available}]. Tried: [{candidates}].")
    raise KeyError(
        f"{path} does not contain input key '{requested}', and auto-detection failed. "
        f"Available keys: [{available}]. Tried: [{candidates}]."
    )


def _detect_input_key(data: np.lib.npyio.NpzFile) -> str | None:
    for key in AUTO_INPUT_KEYS:
        if key in data and _looks_like_daeac_input(data[key]):
            return key
    for key in data.files:
        if _looks_like_daeac_input(data[key]):
            return key
    return None


def _looks_like_daeac_input(value: np.ndarray) -> bool:
    shape = tuple(value.shape)
    return (len(shape) == 4 and shape[1:] == (1, 3, 128)) or (len(shape) == 3 and shape[1:] == (3, 128))


def _normalize_input_array(x: np.ndarray, path: Path, input_key: str) -> np.ndarray:
    if x.ndim == 3 and tuple(x.shape[1:]) == (3, 128):
        return x[:, None, :, :]
    return x


def _validate_input_shape(x: np.ndarray, path: Path, input_key: str) -> None:
    if x.ndim != 4 or tuple(x.shape[1:]) != (1, 3, 128):
        raise ValueError(f"{path}: expected {input_key} shape [N, 1, 3, 128], got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: {input_key} contains NaN or Inf.")


def _validate_labels(y: np.ndarray, path: Path, class_names: list[str]) -> None:
    if y.ndim != 1:
        raise ValueError(f"{path}: labels must be 1-D, got {y.shape}.")
    valid = set(range(len(class_names)))
    observed = set(int(v) for v in np.unique(y))
    invalid = sorted(observed - valid)
    if invalid:
        raise ValueError(f"{path}: labels contain invalid ids {invalid}; expected {sorted(valid)}.")


def _validate_class_names(data: np.lib.npyio.NpzFile, path: Path, expected: list[str]) -> None:
    if "class_names" not in data:
        return
    found = [str(v) for v in data["class_names"].tolist()]
    if found == expected:
        return
    if found == REFERENCE_REPO_CLASS_NAMES:
        raise ValueError(
            f"{path}: class_names are {found}, which matches the DAEAC reference repo order. "
            f"Paper-faithful phase requires {expected}; convert labels before training."
        )
    raise ValueError(f"{path}: class_names are {found}; expected {expected}.")
