from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

try:
    from umap import UMAP
except ModuleNotFoundError:
    UMAP = None

from src.data.label_mapping import ID_TO_CLASS
from src.utils.io import ensure_dir


def plot_umap_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    domains: list[str],
    output_path: str | Path,
    seed: int = 42,
) -> None:
    if UMAP is not None:
        coords = UMAP(n_components=2, random_state=seed).fit_transform(embeddings)
        method = "UMAP"
    else:
        coords = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
        method = "PCA fallback"
    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(8, 6))
    class_names = [ID_TO_CLASS[int(y)] for y in labels]
    markers = {"mitbih": "o", "incart": "^"}
    colors = {"N": "tab:blue", "S": "tab:orange", "V": "tab:green"}
    for domain in sorted(set(domains)):
        for class_name in sorted(set(class_names)):
            mask = np.array([(d == domain and c == class_name) for d, c in zip(domains, class_names)])
            if not mask.any():
                continue
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                alpha=0.75,
                linewidths=0,
                marker=markers.get(domain, "o"),
                color=colors.get(class_name, "gray"),
                label=f"{domain}-{class_name}",
            )
    plt.legend(markerscale=2, fontsize=8)
    plt.title(f"ResNet1D embeddings ({method})")
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
