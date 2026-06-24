from __future__ import annotations

from workflow import prepare_train_run, train_parser
from src.training.train_daeac_adversarial import train_daeac_cdan
from src.utils.io import write_json


def main() -> None:
    parser = train_parser("configs/phase6_daeac_cdan.yaml")
    parser.add_argument("--method", choices=["cdan", "cdan_e"], default=None)
    args = parser.parse_args()
    config, source_ds, val_ds, target_ds, target_val_ds, output, device = prepare_train_run(args, "cdan")
    if args.method is not None:
        config["cdan"]["method"] = str(args.method)
        if args.checkpoint_prefix is None and args.domain_pair is None:
            config["training"]["checkpoint_prefix"] = "daeac_cdan" if args.method == "cdan" else "daeac_cdan_e"
    summary = train_daeac_cdan(source_ds, val_ds, target_ds, target_val_ds, config, output, device)
    prefix = config["training"]["checkpoint_prefix"]
    write_json(summary, output / "metrics" / f"{prefix}_train_summary.json")


if __name__ == "__main__":
    main()
