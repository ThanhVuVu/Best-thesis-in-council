from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.label_mapping import CLASS_NAMES
from src.utils.io import ensure_dir


def plot_example_beats(npz_paths: list[str | Path], output_path: str | Path, examples_per_class: int = 10, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        domain = str(data["domain"][0])
        for class_id, class_name in enumerate(CLASS_NAMES):
            indices = np.where(data["y"] == class_id)[0]
            if len(indices) == 0:
                continue
            chosen = rng.choice(indices, size=min(examples_per_class, len(indices)), replace=False)
            rows.append((domain, class_name, data["x"][chosen, 0, :]))

    output = Path(output_path)
    ensure_dir(output.parent)
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, max(3, 2.2 * len(rows))), squeeze=False)
    for ax, (domain, class_name, beats) in zip(axes[:, 0], rows):
        for beat in beats:
            ax.plot(beat, alpha=0.7, linewidth=1)
        ax.set_title(f"{domain} class {class_name}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("z-score")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close(fig)
