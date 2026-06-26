from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import cfg_path, load_phase1_config
from src.data.daeac_preprocess import CLASS_TO_ID, preprocess_daeac_records
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
    drop_classes_by_domain = {
        str(domain): [str(cls) for cls in classes]
        for domain, classes in dict(config.get("filter", {}).get("drop_classes_by_domain", {})).items()
    }
    output_class_names = [str(name) for name in config.get("data", {}).get("class_names", [])]
    for name, (raw_dir, records, role, dataset) in domains.items():
        counts = record_class_counts(raw_dir, records)
        if name == "ds1":
            sizes = {"train": 17, "val": 5}
            val_records = ["114", "124", "201", "205", "223"]
            train_records = [r for r in records if r not in val_records]
            splits = {"train": sorted(train_records), "val": sorted(val_records)}
        elif name == "ds2":
            sizes = {"train": 14, "val": 4, "test": 4}
            val_records = ["100", "202", "210", "214"]
            test_records = ["200", "219", "222", "233"]
            train_records = [r for r in records if r not in val_records and r not in test_records]
            splits = {
                "train": sorted(train_records),
                "val": sorted(val_records),
                "test": sorted(test_records)
            }
        else:
            sizes = _sizes(len(records), role)
            splits = balanced_record_split(counts, sizes, seed=seed, trials=trials)
        audit = audit_record_split(counts, splits, sizes)
        if not audit["valid"]:
            raise RuntimeError(f"Invalid record split for {name}: {audit}")
        manifest["domains"][name] = {"role": role, "raw_dir": str(raw_dir), "audit": audit}
        if name in drop_classes_by_domain:
            manifest["domains"][name]["sample_filter"] = {"drop_classes": drop_classes_by_domain[name], "splits": {}}
        for split_name, split_records in splits.items():
            output = bundle / f"{name}_{split_name}.npz"
            preprocess_daeac_records(raw_dir, split_records, output, dataset, config, split_rule="all", force=args.force)
            if name in drop_classes_by_domain:
                removed = _drop_classes(output, drop_classes_by_domain[name], output_class_names=output_class_names)
                manifest["domains"][name]["sample_filter"]["splits"][split_name] = {
                    "removed_samples": removed,
                    "class_counts": _labeled_class_counts(output),
                }
                print(f"{output}: removed {removed} samples for classes {drop_classes_by_domain[name]}")
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


def _drop_classes(path: Path, class_names: list[str], output_class_names: list[str] | None = None) -> int:
    unknown = sorted(set(class_names) - set(CLASS_TO_ID))
    if unknown:
        raise ValueError(f"Unknown DAEAC classes in drop filter: {unknown}")
    drop_ids = {CLASS_TO_ID[name] for name in class_names}
    with np.load(path, allow_pickle=True) as data:
        if "y" not in data.files:
            return 0
        y = np.asarray(data["y"], dtype=np.int64)
        keep = ~np.isin(y, list(drop_ids))
        payload = {}
        for key in data.files:
            value = data[key]
            if value.shape[:1] == y.shape[:1]:
                payload[key] = value[keep]
            else:
                payload[key] = value
        if output_class_names:
            output_class_names = [str(name) for name in output_class_names]
            output_class_to_id = {name: CLASS_TO_ID[name] for name in output_class_names}
            payload["class_names"] = np.asarray(output_class_names, dtype=object)
            payload["class_to_id_json"] = np.asarray(json.dumps(output_class_to_id, sort_keys=True), dtype=object)
            if "config_json" in payload:
                cfg = json.loads(str(payload["config_json"].tolist()))
                cfg["class_names"] = output_class_names
                cfg["class_to_id"] = output_class_to_id
                payload["config_json"] = np.asarray(json.dumps(cfg, sort_keys=True), dtype=object)
    np.savez_compressed(path, **payload)
    return int((~keep).sum())


def _labeled_class_counts(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=True) as data:
        if "y" not in data.files:
            return {}
        counts = np.bincount(np.asarray(data["y"], dtype=np.int64), minlength=len(CLASS_TO_ID))
    return {name: int(counts[idx]) for name, idx in CLASS_TO_ID.items()}


if __name__ == "__main__":
    main()
