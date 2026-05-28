from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from common import load_phase1_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4a_ecgfm_leadbridge.yaml")
    parser.add_argument("--windows", nargs="+", type=float, default=[2.5, 5.0, 10.0])
    parser.add_argument("--target-fs", type=int, default=None)
    parser.add_argument("--mitbih-raw-dir", default=None)
    parser.add_argument("--incart-raw-dir", default=None)
    parser.add_argument("--ecgfm-checkpoint", default=None)
    parser.add_argument("--fairseq-signals-path", default=None)
    parser.add_argument("--max-records", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-fit-samples", type=int, default=32)
    parser.add_argument("--max-val-samples", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--output-root", default="outputs/phase4a_window_ablation")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    config = load_phase1_config(args.config)
    base_dir = Path(config["_base_dir"])
    output_root = _resolve(args.output_root, base_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seconds in args.windows:
        variant = _variant_name(seconds)
        variant_root = output_root / variant
        processed_dir = variant_root / "data"
        output_dir = variant_root / "outputs"
        cfg_path = variant_root / f"phase4a_{variant}.yaml"
        variant_root.mkdir(parents=True, exist_ok=True)

        target_fs = int(args.target_fs or config["data"]["target_fs"])
        window_samples = int(round(seconds * target_fs))
        variant_cfg = _make_variant_config(
            config=config,
            seconds=seconds,
            target_fs=target_fs,
            window_samples=window_samples,
            processed_dir=processed_dir,
            output_dir=output_dir,
            checkpoint_prefix=f"source_only_ecgfm_leadbridge_{variant}",
            raw_mitbih=args.mitbih_raw_dir,
            raw_incart=args.incart_raw_dir,
            ecgfm_checkpoint=args.ecgfm_checkpoint,
            fairseq_signals_path=args.fairseq_signals_path,
        )
        cfg_path.write_text(yaml.safe_dump(variant_cfg, sort_keys=False), encoding="utf-8")

        print(f"\n=== Phase 4A window ablation: {variant} ({seconds}s, {window_samples} samples) ===", flush=True)
        prep_cmd = [
            sys.executable,
            str(base_dir / "scripts" / "phase4a" / "01_prepare_5s_windows.py"),
            "--config",
            str(cfg_path),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(output_dir),
            "--max-records",
            str(args.max_records),
        ]
        if args.force_preprocess:
            prep_cmd.append("--force")
        _run(prep_cmd, cwd=base_dir)

        checkpoint = output_dir / "checkpoints" / f"source_only_ecgfm_leadbridge_{variant}_best.pt"
        if not args.skip_train:
            train_cmd = [
                sys.executable,
                str(base_dir / "scripts" / "phase4a" / "02_train_source_ecgfm_leadbridge.py"),
                "--config",
                str(cfg_path),
                "--epochs",
                str(args.epochs),
                "--max-fit-samples",
                str(args.max_fit_samples),
                "--max-val-samples",
                str(args.max_val_samples),
            ]
            if args.ecgfm_checkpoint is not None:
                train_cmd += ["--ecgfm-checkpoint", args.ecgfm_checkpoint]
            if args.fairseq_signals_path is not None:
                train_cmd += ["--fairseq-signals-path", args.fairseq_signals_path]
            _run(train_cmd, cwd=base_dir)

        if not args.skip_eval and checkpoint.exists():
            eval_cmd = [
                sys.executable,
                str(base_dir / "scripts" / "phase4a" / "03_eval_source_ecgfm_leadbridge.py"),
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                "both",
                "--max-samples",
                str(args.max_eval_samples),
            ]
            if args.ecgfm_checkpoint is not None:
                eval_cmd += ["--ecgfm-checkpoint", args.ecgfm_checkpoint]
            if args.fairseq_signals_path is not None:
                eval_cmd += ["--fairseq-signals-path", args.fairseq_signals_path]
            _run(eval_cmd, cwd=base_dir)

        summaries.append(_collect_variant_summary(variant, cfg_path, processed_dir, output_dir))

    summary_path = output_root / "window_ablation_smoke_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote ablation summary: {summary_path}", flush=True)


def _make_variant_config(
    config: dict,
    seconds: float,
    target_fs: int,
    window_samples: int,
    processed_dir: Path,
    output_dir: Path,
    checkpoint_prefix: str,
    raw_mitbih: str | None,
    raw_incart: str | None,
    ecgfm_checkpoint: str | None,
    fairseq_signals_path: str | None,
) -> dict:
    cfg = json.loads(json.dumps(config, default=str))
    cfg.pop("_base_dir", None)
    cfg["paths"]["processed_dir"] = str(processed_dir)
    cfg["paths"]["output_dir"] = str(output_dir)
    if raw_mitbih is not None:
        cfg["paths"]["mitbih_raw_dir"] = raw_mitbih
    if raw_incart is not None:
        cfg["paths"]["incart_raw_dir"] = raw_incart
    cfg["data"]["window_seconds"] = float(seconds)
    cfg["data"]["target_fs"] = int(target_fs)
    cfg["data"]["window_samples"] = int(window_samples)
    cfg["data"]["source_train"] = str(processed_dir / "mitbih_train_5s.npz")
    cfg["data"]["source_test"] = str(processed_dir / "mitbih_test_5s.npz")
    cfg["data"]["target_unlabeled"] = str(processed_dir / "incart_unlabeled_5s.npz")
    cfg["data"]["target_test"] = str(processed_dir / "incart_test_heldout_5s.npz")
    cfg["source_only"]["checkpoint_prefix"] = checkpoint_prefix
    if ecgfm_checkpoint is not None:
        cfg["ecgfm"]["checkpoint_path"] = ecgfm_checkpoint
        cfg["model"]["ecgfm_checkpoint_path"] = ecgfm_checkpoint
    if fairseq_signals_path is not None:
        cfg["ecgfm"]["fairseq_signals_path"] = fairseq_signals_path
        cfg["model"]["fairseq_signals_path"] = fairseq_signals_path
    return cfg


def _collect_variant_summary(variant: str, cfg_path: Path, processed_dir: Path, output_dir: Path) -> dict:
    metrics_dir = output_dir / "metrics"
    return {
        "variant": variant,
        "config": str(cfg_path),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "preprocess_summary": _read_json(metrics_dir / "phase4a_preprocess_summary.json"),
        "train_summary": _read_json(metrics_dir / "source_only_ecgfm_leadbridge_train_summary.json"),
        "mitbih_metrics": _first_json(metrics_dir, f"source_only_ecgfm_leadbridge_mitbih_test_max_samples_*_metrics.json"),
        "incart_metrics": _first_json(metrics_dir, f"source_only_ecgfm_leadbridge_incart_heldout_max_samples_*_metrics.json"),
    }


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _first_json(folder: Path, pattern: str):
    matches = sorted(folder.glob(pattern))
    if not matches:
        return None
    return _read_json(matches[-1])


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _resolve(path: str, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def _variant_name(seconds: float) -> str:
    text = f"{seconds:g}".replace(".", "p")
    return f"w{text}s"


if __name__ == "__main__":
    main()
