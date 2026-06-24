from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import cfg_path, load_phase1_config
from src.data.daeac_preprocess import preprocess_daeac_records
from src.data.physionet import discover_records
from src.data.record_splits import audit_record_split, balanced_record_split, record_class_counts, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_record_splits.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    seed = int(config.get("seed", 42))
    trials = int(config["split"].get("trials", 10000))
    bundle = cfg_path(config, "paths", "bundle_dir")
    bundle.mkdir(parents=True, exist_ok=True)
    mitdb = cfg_path(config, "paths", "mitdb_raw_dir")
    domains = {
        "ds1": (mitdb, list(map(str, config["mitdb"]["ds1"])), "source", "mitdb"),
        "mitbih": (mitdb, [*map(str, config["mitdb"]["ds1"]), *map(str, config["mitdb"]["ds2"])], "source", "mitdb"),
        "ds2": (mitdb, list(map(str, config["mitdb"]["ds2"])), "target", "mitdb"),
        "incart": (cfg_path(config, "paths", "incart_raw_dir"), discover_records(cfg_path(config, "paths", "incart_raw_dir")), "target", "incart"),
        "svdb": (cfg_path(config, "paths", "svdb_raw_dir"), discover_records(cfg_path(config, "paths", "svdb_raw_dir")), "target", "svdb"),
    }
    manifest = {"schema_version": 1, "seed": seed, "domains": {}, "preprocessing": config["preprocessing"]}
    for name, (raw_dir, records, role, dataset) in domains.items():
        sizes = _sizes(len(records), role)
        counts = record_class_counts(raw_dir, records)
        splits = balanced_record_split(counts, sizes, seed=seed, trials=trials)
        audit = audit_record_split(counts, splits, sizes)
        if not audit["valid"]:
            raise RuntimeError(f"Invalid record split for {name}: {audit}")
        manifest["domains"][name] = {"role": role, "raw_dir": str(raw_dir), "audit": audit}
        for split_name, split_records in splits.items():
            output = bundle / f"{name}_{split_name}.npz"
            preprocess_daeac_records(raw_dir, split_records, output, dataset, config, split_rule="all", force=args.force)
            if role == "target" and split_name in {"train", "val"}:
                _strip_labels(output)
    write_manifest(bundle / "record_split_manifest.json", manifest)
    print(bundle / "record_split_manifest.json")


def _sizes(total: int, role: str) -> dict[str, int]:
    if role == "source":
        train = int(round(total * 0.8))
        return {"train": train, "val": total - train}
    train, val = int(round(total * 0.6)), int(round(total * 0.2))
    return {"train": train, "val": val, "test": total - train - val}


def _strip_labels(path: Path) -> None:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files if key != "y"}
    np.savez_compressed(path, **payload)


if __name__ == "__main__":
    main()
