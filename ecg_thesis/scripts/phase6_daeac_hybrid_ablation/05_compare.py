from __future__ import annotations

import argparse
import csv
from pathlib import Path

import common  # noqa: F401 - adds the project root to sys.path
from src.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/phase6_daeac_hybrid_ablation_comparison")
    args = parser.parse_args()
    rows = []
    for directory in args.input_dirs:
        for path in sorted((Path(directory) / "metrics").glob("*_best_*_metrics.json")):
            metric = read_json(path)
            if metric.get("checkpoint_kind") != "best" or metric.get("dataset") not in {"target_after5", "incart", "svdb"}:
                continue
            per = metric.get("per_class", {})
            row = {"method": metric.get("setting"), "dataset": metric.get("dataset"), "accuracy": metric.get("accuracy"), "macro_f1": metric.get("macro_f1")}
            for name in ("N", "S", "V", "F"):
                for key in ("precision", "recall", "f1"):
                    row[f"{name}_{key}"] = per.get(name, {}).get(key)
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No best target_after5/INCART/SVDB metric files found in input directories.")
    output = ensure_dir(args.output_dir)
    csv_path = output / "phase6_hybrid_ablation_best_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
