from __future__ import annotations

from workflow import prepare_train_run, train_parser
from src.training.train_daeac_adversarial import train_daeac_dann
from src.utils.io import write_json


def main() -> None:
    parser = train_parser("configs/phase6_daeac_dann.yaml")
    args = parser.parse_args()
    config, source_ds, val_ds, target_ds, target_val_ds, output, device = prepare_train_run(args, "dann")
    summary = train_daeac_dann(source_ds, val_ds, target_ds, target_val_ds, config, output, device)
    prefix = config["training"]["checkpoint_prefix"]
    write_json(summary, output / "metrics" / f"{prefix}_train_summary.json")


if __name__ == "__main__":
    main()
