from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, write_json


MERGE_FILES = {
    "method_layer_collapse_comparison.csv": "layer_pairwise_separability.csv",
    "method_minority_to_N_summary.csv": "layer_knn_purity.csv",
    "method_raw_feature_effect_summary.csv": "clinical_proxy_effect_size.csv",
    "method_linear_probe_summary.csv": "layer_linear_probe.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_feature_debug.yaml")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-merge-figures", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = ensure_dir(Path(args.output_dir) if args.output_dir else cfg_path(config, "paths", "output_dir") / "merged")
    input_dirs = _expand_dirs(args.input_dirs)
    summary = {"input_dirs": [str(path) for path in input_dirs], "merged_files": {}}
    for output_name, source_name in MERGE_FILES.items():
        rows = []
        for input_dir in input_dirs:
            source = input_dir / source_name
            if source.exists() and source.stat().st_size > 0:
                rows.extend(_read_csv(source))
        target = output / output_name
        _write_csv(target, rows)
        summary["merged_files"][output_name] = {"rows": len(rows), "path": str(target)}
    if not args.no_merge_figures:
        figure_summary = _merge_figure_comparisons(input_dirs, output / "figure_comparisons")
        summary["figure_comparisons"] = figure_summary
    write_json(summary, output / "merge_summary.json")
    print(f"merged reports written to {output}")


def _expand_dirs(values: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for value in values:
        matches = [Path(match) for match in glob.glob(value)]
        dirs.extend(matches if matches else [Path(value)])
    return [path for path in dirs if path.exists() and path.is_dir()]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merge_figure_comparisons(input_dirs: list[Path], output_dir: Path) -> dict:
    groups: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for input_dir in input_dirs:
        figures_dir = input_dir / "figures"
        if not figures_dir.exists():
            continue
        for image_path in sorted(figures_dir.rglob("*.png")):
            parsed = _parse_figure_path(figures_dir, image_path)
            if parsed is None:
                continue
            method, dataset, figure_key = parsed
            groups.setdefault((dataset, figure_key), []).append((method, image_path))

    output_dir = ensure_dir(output_dir)
    created = []
    for (dataset, figure_key), items in sorted(groups.items()):
        deduped = _dedupe_method_images(items)
        if len(deduped) < 2:
            continue
        target = output_dir / dataset / f"{_safe_name(figure_key)}_comparison.png"
        _write_image_grid(deduped, target, title=f"{dataset}: {figure_key}")
        created.append({"dataset": dataset, "figure": figure_key, "methods": [m for m, _ in deduped], "path": str(target)})
    return {"created": len(created), "items": created}


def _parse_figure_path(figures_dir: Path, image_path: Path) -> tuple[str, str, str] | None:
    rel = image_path.relative_to(figures_dir)
    if len(rel.parts) < 2:
        return None
    folder = rel.parts[0]
    dataset = None
    method = None
    for suffix in ("target", "incart", "svdb"):
        marker = f"_{suffix}"
        if folder.endswith(marker):
            dataset = suffix
            method = folder[: -len(marker)]
            break
    if dataset is None or not method:
        return None
    figure_key = "/".join(rel.parts[1:])
    return method, dataset, figure_key


def _dedupe_method_images(items: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    by_method: dict[str, Path] = {}
    for method, path in items:
        by_method[method] = path
    return [(method, by_method[method]) for method in sorted(by_method)]


def _write_image_grid(items: list[tuple[str, Path]], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    n = len(items)
    fig_width = max(4.0 * n, 8.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 4.5), squeeze=False)
    for ax, (method, path) in zip(axes[0], items):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(method, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _safe_name(value: str) -> str:
    safe = []
    for char in value:
        safe.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(safe).strip("_")


if __name__ == "__main__":
    main()
