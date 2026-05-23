from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.data.label_mapping import CLASS_NAMES
from src.utils.io import ensure_dir


def plot_confusion_matrix(cm: list[list[int]], output_path: str | Path, title: str) -> None:
    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(5, 4))
    image = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    max_value = max(max(row) for row in cm) if cm else 0
    threshold = max_value / 2.0
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            color = "white" if value > threshold else "black"
            plt.text(j, i, str(value), ha="center", va="center", color=color)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
