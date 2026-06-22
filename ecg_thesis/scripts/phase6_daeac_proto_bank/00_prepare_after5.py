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
    protocol = str(config["data"].get("target_protocol", "first5_adapt_after5_test"))
    full_target = cfg_path(config, "data", "target_full_transductive")
    if protocol == "full_target_transductive":
        target_unlabeled = cfg_path(config, "data", "target_unlabeled")
        target_test = cfg_path(config, "data", "target_test")
        if target_unlabeled.resolve() != full_target.resolve() or target_test.resolve() != full_target.resolve():
            raise ValueError("full_target_transductive requires target_unlabeled and target_test to equal the full target file.")
        before5 = None
        after5 = None
    else:
        before5 = create_daeac_before_time_split(
            full_target,
            cfg_path(config, "data", "target_unlabeled"),
            threshold_sec=threshold,
            force=args.force,
        )
        after5 = None
        if protocol == "first5_adapt_after5_test":
            after5 = create_daeac_after_time_split(
                full_target,
                cfg_path(config, "data", "target_test"),
                threshold_sec=threshold,
                force=args.force,
            )
        elif protocol != "first5_adapt_full_test":
            raise ValueError(f"Unknown data.target_protocol: {protocol}")
    overlap = audit_daeac_disjoint(
        cfg_path(config, "data", "target_unlabeled"),
        cfg_path(config, "data", "target_test"),
    )
    if protocol == "first5_adapt_after5_test" and not overlap["disjoint"]:
        raise ValueError(f"Target first5/after5 overlap detected: {overlap['overlap_count']} samples.")
    report = {"target_protocol": protocol, "before5": before5, "after5": after5, "overlap_audit": overlap}
    output = cfg_path(config, "paths", "output_dir") / "diagnostics" / "after5_prepare.json"
    write_json(report, output)
    print(report)


if __name__ == "__main__":
    main()
