from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.data.rr_features import add_rr_features_to_npz, apply_rr_normalizer, fit_rr_normalizer
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_rr_dann.yaml")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    paths = {
        "source_train": cfg_path(config, "data", "source_train"),
        "source_test": cfg_path(config, "data", "source_test"),
        "target_unlabeled": cfg_path(config, "data", "target_unlabeled"),
        "target_test": cfg_path(config, "data", "target_test"),
    }
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    metrics_dir = ensure_dir(output / "metrics")

    raw_train_rr, _ = add_rr_features_to_npz(paths["source_train"])
    stats = fit_rr_normalizer(raw_train_rr)
    add_rr_features_to_npz(paths["source_train"], normalizer=stats)
    for name in ("source_test", "target_unlabeled", "target_test"):
        add_rr_features_to_npz(paths[name], normalizer=stats)

    write_json(stats, metrics_dir / "phase3_rr_normalization.json")
    write_json({key: str(path) for key, path in paths.items()}, metrics_dir / "phase3_rr_prepared_files.json")
    print(f"Saved RR normalization stats to {metrics_dir / 'phase3_rr_normalization.json'}")


if __name__ == "__main__":
    main()
