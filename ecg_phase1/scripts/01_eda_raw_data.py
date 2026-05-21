from __future__ import annotations

import argparse
from collections import Counter

from common import cfg_path, load_phase1_config
from src.data.physionet import summarize_raw_dataset
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)

    mit_dir = cfg_path(config, "paths", "mitbih_raw_dir")
    inc_dir = cfg_path(config, "paths", "incart_raw_dir")
    out_dir = ensure_dir(cfg_path(config, "paths", "output_dir") / "metrics")

    mit = summarize_raw_dataset(
        mit_dir,
        "mitbih",
        config["lead_selection"]["preferred_leads"]["mitbih"],
        int(config["lead_selection"]["fallback_lead_index"]),
    )
    inc = summarize_raw_dataset(
        inc_dir,
        "incart",
        config["lead_selection"]["preferred_leads"]["incart"],
        int(config["lead_selection"]["fallback_lead_index"]),
    )
    write_json({"mitbih": mit, "incart": inc}, out_dir / "raw_eda.json")

    for summary in (mit, inc):
        print(f"\n{summary['dataset'].upper()}")
        print(f"Records read: {summary['records_read_ok']} / {summary['records_discovered']}")
        print(f"FS counts: {summary['fs_counts']}")
        print(f"Signal counts: {summary['n_signal_counts']}")
        print(f"Selected leads: {summary['selected_lead_counts']}")
        print(f"Fallback records: {summary['fallback_records']}")
        print(f"Mapped N/S/V: {summary['mapped_nsv_counts']}")
        print(f"Ignored: {summary['total_ignored']}")
        print("Top symbols:", Counter(summary["symbol_counts"]).most_common(15))


if __name__ == "__main__":
    main()
