from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.data.paper_preprocess import compute_source_mean_rr_samples, fit_time_normalizer, preprocess_paper_records
from src.data.splits import MITBIH_TEST_RECORDS, MITBIH_TRAIN_RECORDS
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))

    processed_dir = ensure_dir(cfg_path(config, "paths", "processed_dir"))
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    metrics_dir = ensure_dir(output / "metrics")
    mit_dir = cfg_path(config, "paths", "mitbih_raw_dir")
    inc_dir = cfg_path(config, "paths", "incart_raw_dir")
    mit_train = _limit(MITBIH_TRAIN_RECORDS, args.max_records)
    mit_test = _limit(MITBIH_TEST_RECORDS, args.max_records)
    inc_adapt = _limit(config["incart_split"]["adapt_records"], args.max_records)
    inc_test = _limit(config["incart_split"]["test_records"], args.max_records)

    target_fs = int(config["paper_preprocess"]["resampling"]["target_fs"])
    segment_length = compute_source_mean_rr_samples(mit_dir, mit_train, target_fs)
    summaries = {"segment_length": int(segment_length), "target_fs": target_fs}

    train_path = cfg_path(config, "data", "source_train")
    train_summary_raw, train_time_raw = preprocess_paper_records(
        mit_dir, mit_train, train_path, "mitbih", config, segment_length, time_normalizer=None, force=True
    )
    time_stats = fit_time_normalizer(train_time_raw)
    train_summary, _ = preprocess_paper_records(
        mit_dir, mit_train, train_path, "mitbih", config, segment_length, time_normalizer=time_stats, force=True
    )
    summaries["mitbih_train_raw_first_pass"] = train_summary_raw
    summaries["mitbih_train"] = train_summary
    summaries["time_normalization"] = time_stats

    jobs = [
        ("mitbih_test", mit_dir, mit_test, cfg_path(config, "data", "source_test"), "mitbih"),
        ("incart_unlabeled", inc_dir, inc_adapt, cfg_path(config, "data", "target_unlabeled"), "incart"),
        ("incart_heldout", inc_dir, inc_test, cfg_path(config, "data", "target_test"), "incart"),
    ]
    for name, raw_dir, records, path, dataset in jobs:
        summary, _ = preprocess_paper_records(
            raw_dir, records, path, dataset, config, segment_length, time_normalizer=time_stats, force=args.force
        )
        summaries[name] = summary

    write_json(summaries, metrics_dir / "phase2p_preprocess_summary.json")
    write_json({"processed_dir": str(processed_dir), "files": {k: str(cfg_path(config, "data", k)) for k in config["data"] if k.startswith("source") or k.startswith("target")}}, metrics_dir / "phase2p_preprocessed_files.json")
    print(summaries)


def _limit(records: list[str], max_records: int | None) -> list[str]:
    records = list(records)
    if max_records is None:
        return records
    return records[: int(max_records)]


if __name__ == "__main__":
    main()

