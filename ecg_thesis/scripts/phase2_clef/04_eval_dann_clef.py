from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from scripts_eval_common import evaluate_and_save
from src.data.datasets import ECGBeatDataset
from src.training.train_dann import load_dann_from_checkpoint
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_clef_dann.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", choices=["mitbih", "incart", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clef-checkpoint", default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    if args.clef_checkpoint is not None:
        config["model"]["clef_checkpoint_path"] = args.clef_checkpoint
    device = device_from_torch()
    print(f"Using device: {device}")

    prefix = config["training"].get("checkpoint_prefix", "clef_dann")
    checkpoint = args.checkpoint or f"outputs/phase2_clef_dann/checkpoints/{prefix}_best.pt"
    checkpoint_path = cfg_path({"paths": {"checkpoint": checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")
    model, ckpt = load_dann_from_checkpoint(
        checkpoint_path,
        device,
        model_kwargs_override={"clef_checkpoint_path": args.clef_checkpoint},
    )
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = {"checkpoint": str(checkpoint_path), "mode": prefix, "datasets": {}, "checkpoint_epoch": ckpt.get("epoch")}

    if args.dataset in ("mitbih", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "source_test"), return_metadata=True)
        dataset, name = _maybe_subset(dataset, f"{prefix}_mitbih_test", args.max_samples)
        summary["datasets"]["mitbih_test"] = evaluate_and_save(model, dataset, device, output, name, prefix)

    if args.dataset in ("incart", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "target_test"), return_metadata=True)
        dataset, name = _maybe_subset(dataset, f"{prefix}_incart_heldout", args.max_samples)
        summary["datasets"]["incart_heldout"] = evaluate_and_save(model, dataset, device, output, name, prefix)

    write_json(summary, output / "metrics" / f"{prefix}_eval_summary.json")


def _maybe_subset(dataset, name: str, max_samples: int | None):
    if max_samples is None:
        return dataset, name
    n = min(int(max_samples), len(dataset))
    print(f"Evaluating subset {name}: {n} samples")
    return Subset(dataset, list(range(n))), f"{name}_max_samples_{n}"


if __name__ == "__main__":
    main()
