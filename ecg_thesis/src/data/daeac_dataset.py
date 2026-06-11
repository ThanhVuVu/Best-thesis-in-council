from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

PAPER_CLASS_NAMES = ["N", "S", "V", "F"]
REFERENCE_REPO_CLASS_NAMES = ["N", "V", "S", "F"]


class DAEACDataset(Dataset):
    """Paper-faithful DAEAC dataset for preprocessed [N, 1, 3, 128] arrays."""

    def __init__(
        self,
        npz_path: str | Path,
        input_key: str = "x_daeac",
        label_key: str = "y",
        require_labels: bool = True,
        return_index: bool = False,
        return_metadata: bool = False,
        class_names: list[str] | None = None,
    ):
        self.path = Path(npz_path)
        self.input_key = str(input_key)
        self.label_key = str(label_key)
        self.require_labels = bool(require_labels)
        self.return_index = bool(return_index)
        self.return_metadata = bool(return_metadata)
        self.class_names = list(class_names or PAPER_CLASS_NAMES)

        self.data = np.load(self.path, allow_pickle=True)
        if self.input_key not in self.data:
            raise KeyError(f"{self.path} does not contain input key '{self.input_key}'.")
        self.x = self.data[self.input_key].astype(np.float32)
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


class DAEACTargetUnlabeledDataset(DAEACDataset):
    """Target adaptation dataset that intentionally never exposes target labels."""

    def __init__(
        self,
        npz_path: str | Path,
        input_key: str = "x_daeac",
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


def class_counts_from_dataset(dataset: DAEACDataset, num_classes: int = 4) -> np.ndarray:
    if dataset.y is None:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount(dataset.y.astype(np.int64), minlength=num_classes)


def inspect_daeac_npz(
    npz_path: str | Path,
    input_key: str = "x_daeac",
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
        "input_key": input_key,
        "shape": list(ds.x.shape),
        "has_labels": ds.y is not None,
        "class_names": ds.class_names,
        "class_counts": {name: counts[idx] for idx, name in enumerate(ds.class_names)},
        "num_records": int(len(set(records.astype(str)))) if records is not None else None,
        "num_samples": int(len(ds)),
    }


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
