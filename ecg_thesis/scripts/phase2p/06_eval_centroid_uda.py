from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, device_from_torch, evaluate_model, load_phase1_config, load_phase2p_checkpoint, write_eval_outputs
from src.data.datasets import ECGBeatTimeDataset
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--method-name", default="phase2p_centroid_uda")
    parser.add_argument("--dataset", choices=["both", "source", "target"], default="both")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    device = device_from_torch()
    checkpoint = Path(args.checkpoint or (output / "checkpoints" / f"{config['uda']['checkpoint_prefix']}_best.pt"))
    if not checkpoint.is_absolute():
        checkpoint = cfg_path({"_base_dir": config["_base_dir"], "path": str(checkpoint)}, "path")
    model, ckpt = load_phase2p_checkpoint(checkpoint, config, device)
    batch_size = int(config["evaluation"]["batch_size"])
    class_names = list(config["data"]["class_names"])
    summary = {"checkpoint": str(checkpoint), "method_name": args.method_name, "datasets": {}}

    if args.dataset in {"both", "source"}:
        source_ds = ECGBeatTimeDataset(cfg_path(config, "data", "source_test"), return_metadata=True)
        result = evaluate_model(model, source_ds, device, batch_size=batch_size, max_samples=args.max_samples)
        name = f"{args.method_name}_mitbih_test"
        write_eval_outputs(result, output, name, class_names)
        summary["datasets"]["mitbih_test"] = result["metrics"]

    if args.dataset in {"both", "target"}:
        target_ds = ECGBeatTimeDataset(cfg_path(config, "data", "target_test"), return_metadata=True)
        result = evaluate_model(model, target_ds, device, batch_size=batch_size, max_samples=args.max_samples)
        name = f"{args.method_name}_incart_heldout"
        write_eval_outputs(result, output, name, class_names)
        summary["datasets"]["incart_heldout"] = result["metrics"]

    summary["checkpoint_epoch"] = ckpt.get("epoch")
    summary["checkpoint_best_macro_f1"] = ckpt.get("best_macro_f1")
    write_json(summary, output / "metrics" / f"{args.method_name}_eval_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
