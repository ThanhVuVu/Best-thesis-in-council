from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.data.preprocess import all_incart_records, preprocess_records
from src.data.splits import MITBIH_TEST_RECORDS, MITBIH_TRAIN_RECORDS
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)

    mit_dir = cfg_path(config, "paths", "mitbih_raw_dir")
    inc_dir = cfg_path(config, "paths", "incart_raw_dir")
    processed_dir = ensure_dir(cfg_path(config, "paths", "processed_dir"))
    metrics_dir = ensure_dir(cfg_path(config, "paths", "output_dir") / "metrics")

    summaries = {}
    summaries["mitbih_train"] = preprocess_records(
        mit_dir,
        MITBIH_TRAIN_RECORDS,
        processed_dir / "mitbih_train.npz",
        "mitbih",
        config,
        force=args.force,
    )
    summaries["mitbih_test"] = preprocess_records(
        mit_dir,
        MITBIH_TEST_RECORDS,
        processed_dir / "mitbih_test.npz",
        "mitbih",
        config,
        force=args.force,
    )
    summaries["incart_test"] = preprocess_records(
        inc_dir,
        all_incart_records(inc_dir),
        processed_dir / "incart_test.npz",
        "incart",
        config,
        force=args.force,
    )
    write_json(summaries, metrics_dir / "preprocess_summary.json")
    for name, summary in summaries.items():
        print(f"\n{name}: {summary}")


if __name__ == "__main__":
    main()
