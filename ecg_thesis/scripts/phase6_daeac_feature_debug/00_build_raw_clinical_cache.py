from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import cfg_path, load_phase1_config
from src.data.daeac_raw_debug import build_raw_cache_for_dataset, dataset_key_to_raw_key, load_labeled_daeac_dataset
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_feature_debug.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--mitdb-raw-dir", default=None)
    parser.add_argument("--incart-raw-dir", default=None)
    parser.add_argument("--svdb-raw-dir", default=None)
    parser.add_argument("--dataset", default="all", help="target, incart, svdb, or all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-raw-windows", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = ensure_dir(Path(args.output_dir) if args.output_dir else cfg_path(config, "paths", "raw_cache_dir"))
    processed_dir = Path(args.processed_dir) if args.processed_dir else cfg_path(config, "paths", "processed_dir")
    raw_dirs = {
        "mitdb": args.mitdb_raw_dir,
        "incart": args.incart_raw_dir,
        "svdb": args.svdb_raw_dir,
    }
    selected = _selected_datasets(args.dataset)
    raw_cfg = dict(config["raw"])
    if args.save_raw_windows:
        raw_cfg["save_raw_windows"] = True
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    all_rows = []
    window_payload = {}
    summary = {"datasets": {}, "processed_dir": str(processed_dir), "output_dir": str(output)}
    for dataset_key in selected:
        raw_key = dataset_key_to_raw_key(dataset_key)
        raw_dir = raw_dirs.get(raw_key)
        if raw_dir in (None, ""):
            raise ValueError(f"Missing --{raw_key}-raw-dir for dataset={dataset_key}")
        ds = load_labeled_daeac_dataset(processed_dir, dataset_key, input_key=input_key, label_key=label_key, class_names=class_names)
        preferred = list(raw_cfg["lead_selection"][raw_key])
        rows, windows = build_raw_cache_for_dataset(
            dataset_key=dataset_key,
            dataset=ds,
            raw_dir=raw_dir,
            preferred_leads=preferred,
            fallback_lead_index=int(raw_cfg.get("fallback_lead_index", 0)),
            cfg=raw_cfg,
            max_samples=args.max_samples,
        )
        all_rows.extend(rows)
        if windows is not None:
            window_payload[f"{dataset_key}_windows"] = windows
        summary["datasets"][dataset_key] = {
            "processed_path": str(ds.path),
            "raw_dir": str(raw_dir),
            "samples_cached": len(rows),
            "raw_windows_saved": windows is not None,
        }
        ds.close()

    csv_path = output / "raw_clinical_features.csv"
    _write_csv(csv_path, all_rows)
    if window_payload:
        np.savez_compressed(output / "raw_window_cache.npz", **window_payload)
    summary["raw_clinical_features"] = str(csv_path)
    summary["total_samples_cached"] = len(all_rows)
    write_json(summary, output / "raw_cache_summary.json")
    print(f"raw cache written to {output}")


def _selected_datasets(dataset: str) -> list[str]:
    if dataset == "all":
        return ["target", "incart", "svdb"]
    if dataset in {"target", "incart", "svdb"}:
        return [dataset]
    raise ValueError(f"Unknown dataset {dataset!r}; expected target, incart, svdb, or all")


def _write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
