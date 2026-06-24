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





def subset_by_records(dataset: ECGBeatDataset, records: list[str]) -> Subset:
    wanted = set(records)
    indices = [i for i, rec in enumerate(dataset.records) if str(rec) in wanted]
    return Subset(dataset, indices)


def class_counts(npz_path: str | Path, num_classes: int = 3) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    return np.bincount(data["y"].astype(np.int64), minlength=num_classes)
