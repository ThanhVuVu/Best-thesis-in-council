from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.data.physionet import discover_records
from src.data.splits import MITBIH_TEST_RECORDS, MITBIH_TRAIN_RECORDS
from src.data.window_5s import preprocess_5s_windows
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4a_ecgfm_leadbridge.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--mitbih-raw-dir", default=None)
    parser.add_argument("--incart-raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    if args.mitbih_raw_dir is not None:
        config["paths"]["mitbih_raw_dir"] = args.mitbih_raw_dir
    if args.incart_raw_dir is not None:
        config["paths"]["incart_raw_dir"] = args.incart_raw_dir
    if args.processed_dir is not None:
        config["paths"]["processed_dir"] = args.processed_dir
    if args.output_dir is not None:
        config["paths"]["output_dir"] = args.output_dir

    processed_dir = ensure_dir(cfg_path(config, "paths", "processed_dir"))
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    metrics_dir = ensure_dir(output / "metrics")
    mit_dir = cfg_path(config, "paths", "mitbih_raw_dir")
    inc_dir = cfg_path(config, "paths", "incart_raw_dir")
    _validate_raw_dir(mit_dir, "MIT-BIH")
    _validate_raw_dir(inc_dir, "INCART")

    summaries = {}
    summaries["mitbih_train"] = preprocess_5s_windows(
        mit_dir,
        _limit(MITBIH_TRAIN_RECORDS, args.max_records),
        processed_dir / "mitbih_train_5s.npz",
        "mitbih",
        config,
        force=args.force,
    )
    summaries["mitbih_test"] = preprocess_5s_windows(
        mit_dir,
        _limit(MITBIH_TEST_RECORDS, args.max_records),
        processed_dir / "mitbih_test_5s.npz",
        "mitbih",
        config,
        force=args.force,
    )
    incart_records = set(discover_records(inc_dir))
    adapt_records = [rec for rec in config["incart_split"]["adapt_records"] if rec in incart_records]
    test_records = [rec for rec in config["incart_split"]["test_records"] if rec in incart_records]
    summaries["incart_unlabeled"] = preprocess_5s_windows(
        inc_dir,
        _limit(adapt_records, args.max_records),
        processed_dir / "incart_unlabeled_5s.npz",
        "incart",
        config,
        force=args.force,
    )
    summaries["incart_heldout"] = preprocess_5s_windows(
        inc_dir,
        _limit(test_records, args.max_records),
        processed_dir / "incart_test_heldout_5s.npz",
        "incart",
        config,
        force=args.force,
    )
    write_json(summaries, metrics_dir / "phase4a_preprocess_summary.json")
    for name, summary in summaries.items():
        print(f"\n{name}: {summary}")


def _limit(records: list[str], max_records: int | None) -> list[str]:
    if max_records is None:
        return list(records)
    return list(records)[: int(max_records)]


def _validate_raw_dir(raw_dir, name: str) -> None:
    records = discover_records(raw_dir)
    if records:
        print(f"{name} raw dir ok: {raw_dir} ({len(records)} records discovered)")
        return
    raise FileNotFoundError(
        f"No WFDB records discovered in {raw_dir}. Expected .hea, .dat, and .atr files "
        f"to be in this exact folder. Check that --{name.lower()}-raw-dir points to the "
        "directory containing files like 100.hea/100.dat/100.atr for MIT-BIH or "
        "I01.hea/I01.dat/I01.atr for INCART."
    )


if __name__ == "__main__":
    main()
