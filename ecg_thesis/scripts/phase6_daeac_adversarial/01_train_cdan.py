from __future__ import annotations

from workflow import prepare_train_run, train_parser
from src.training.train_daeac_adversarial import train_daeac_cdan
from src.utils.io import write_json


def _should_override_cdan_prefix(checkpoint_prefix: str | None, domain_pair: str | None, current_prefix: str) -> bool:
    return checkpoint_prefix is None and domain_pair is None and current_prefix in {"daeac_cdan", "daeac_cdan_e"}


def main() -> None:
    parser = train_parser("configs/phase6_daeac_cdan.yaml")
    parser.add_argument("--method", choices=["cdan", "cdan_e"], default=None)
    args = parser.parse_args()
    config, source_ds, val_ds, target_ds, target_val_ds, output, device = prepare_train_run(args, "cdan")
    if args.method is not None:
        config["cdan"]["method"] = str(args.method)
        current_prefix = str(config["training"].get("checkpoint_prefix", ""))
        if _should_override_cdan_prefix(args.checkpoint_prefix, args.domain_pair, current_prefix):
            config["training"]["checkpoint_prefix"] = "daeac_cdan" if args.method == "cdan" else "daeac_cdan_e"
    summary = train_daeac_cdan(source_ds, val_ds, target_ds, target_val_ds, config, output, device)
    prefix = config["training"]["checkpoint_prefix"]
    write_json(summary, output / "metrics" / f"{prefix}_train_summary.json")


if __name__ == "__main__":
    main()
