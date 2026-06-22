from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.data.daeac_protocol import audit_daeac_disjoint, create_daeac_after_time_split, create_daeac_before_time_split
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    threshold = float(config["data"]["target_split_seconds"])
    full_target = cfg_path(config, "data", "target_full_transductive")
    before5 = create_daeac_before_time_split(
        full_target,
        cfg_path(config, "data", "target_unlabeled"),
        threshold_sec=threshold,
        force=args.force,
    )
    after5 = create_daeac_after_time_split(
        full_target,
        cfg_path(config, "data", "target_test"),
        threshold_sec=threshold,
        force=args.force,
    )
    overlap = audit_daeac_disjoint(
        cfg_path(config, "data", "target_unlabeled"),
        cfg_path(config, "data", "target_test"),
    )
    if not overlap["disjoint"]:
        raise ValueError(f"Target first5/after5 overlap detected: {overlap['overlap_count']} samples.")
    report = {"before5": before5, "after5": after5, "overlap_audit": overlap}
    output = cfg_path(config, "paths", "output_dir") / "diagnostics" / "after5_prepare.json"
    write_json(report, output)
    print(report)


if __name__ == "__main__":
    main()
