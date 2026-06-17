from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

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


if __name__ == "__main__":
    main()
