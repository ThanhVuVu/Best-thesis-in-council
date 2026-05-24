from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class ECGBeatDataset(Dataset):
    def __init__(self, npz_path: str | Path, return_metadata: bool = False):
        self.path = Path(npz_path)
        self.data = np.load(self.path, allow_pickle=True)
        self.x = self.data["x"].astype(np.float32)
        self.y = self.data["y"].astype(np.int64)
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        if not self.return_metadata:
            return x, y
        return x, y, self.metadata(idx)

    def metadata(self, idx: int) -> dict[str, Any]:
        fields = ["record", "symbol", "sample", "fs", "domain", "lead_index", "lead_name"]
        return {field: self.data[field][idx].item() if hasattr(self.data[field][idx], "item") else self.data[field][idx] for field in fields}

    @property
    def records(self) -> np.ndarray:
        return self.data["record"]


class ECGBeatRRDataset(ECGBeatDataset):
    def __init__(self, npz_path: str | Path, return_metadata: bool = False):
        super().__init__(npz_path, return_metadata=return_metadata)
        if "rr_features" not in self.data:
            raise KeyError(f"{self.path} does not contain rr_features. Run scripts/phase3/02_prepare_rr_features.py first.")
        self.rr_features = self.data["rr_features"].astype(np.float32)
        if len(self.rr_features) != len(self.y):
            raise ValueError(
                f"rr_features length mismatch in {self.path}: "
                f"{len(self.rr_features)} rr rows vs {len(self.y)} labels"
            )

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx])
        rr = torch.from_numpy(self.rr_features[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        if not self.return_metadata:
            return x, rr, y
        return x, rr, y, self.metadata(idx)


def subset_by_records(dataset: ECGBeatDataset, records: list[str]) -> Subset:
    wanted = set(records)
    indices = [i for i, rec in enumerate(dataset.records) if str(rec) in wanted]
    return Subset(dataset, indices)


def class_counts(npz_path: str | Path, num_classes: int = 3) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    return np.bincount(data["y"].astype(np.int64), minlength=num_classes)
