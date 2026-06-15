from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from diagnostics_common import class_names_from, load_diagnostics_config, method_name, output_dir, write_csv
from src.utils.io import ensure_dir, read_json, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--train-summary", default=None)
    args = parser.parse_args()

    diag_config, base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    class_names = class_names_from(base_config)
    tables_dir = ensure_dir(output_dir(diag_config) / "diagnostics" / "pseudo")
    figures_dir = ensure_dir(output_dir(diag_config) / "figures" / "pseudo")
    summary_path = _summary_path(diag_config, args.train_summary)

    if not summary_path.exists():
        summary = {
            "available": False,
            "reason": "train summary not found",
            "train_summary": str(summary_path),
        }
        write_json(summary, tables_dir / f"{method}_pseudo_summary.json")
        print(f"pseudo-label audit skipped; train summary not found: {summary_path}")
        return

    train_summary = read_json(summary_path)
    history = list(train_summary.get("history", []))
    rows = [_epoch_row(row, class_names) for row in history]
    write_csv(tables_dir / f"{method}_pseudo_epoch_counts.csv", rows)
    if rows:
        _plot_counts(rows, class_names, "pseudo_count_", figures_dir / f"{method}_pseudo_counts.png", f"{method} selected pseudo labels")
        _plot_counts(rows, class_names, "target_pred_count_", figures_dir / f"{method}_target_pred_counts.png", f"{method} target prediction distribution")

    summary = {
        "available": True,
        "train_summary": str(summary_path),
        "best_epoch": train_summary.get("best_epoch"),
        "best_val_macro_f1": train_summary.get("best_val_macro_f1"),
        "epochs": len(history),
        "final_epoch": rows[-1] if rows else None,
        "pseudo_behavior": _pseudo_behavior(rows, class_names),
    }
    write_json(summary, tables_dir / f"{method}_pseudo_summary.json")
    print(f"pseudo-label audit written under {tables_dir} and {figures_dir}")


def _summary_path(config: dict[str, Any], override: str | None) -> Path:
    value = override or config["analysis"].get("train_summary")
    if not value:
        return output_dir(config) / "missing_train_summary.json"
    return resolve_path(value, config["_base_dir"])


def _epoch_row(row: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "epoch": int(row.get("epoch", -1)),
        "val_macro_f1": row.get("val_macro_f1"),
        "pseudo_selected_avg_per_step": row.get("pseudo_selected"),
        "target_entropy": row.get("target_entropy"),
        "loss_mcc": row.get("loss_mcc"),
    }
    pseudo_counts = list(row.get("pseudo_counts", []))
    target_counts = list(row.get("target_pred_counts", []))
    for idx, name in enumerate(class_names):
        out[f"pseudo_count_{name}"] = int(pseudo_counts[idx]) if idx < len(pseudo_counts) else 0
        out[f"target_pred_count_{name}"] = int(target_counts[idx]) if idx < len(target_counts) else 0
    out["pseudo_total"] = int(sum(out[f"pseudo_count_{name}"] for name in class_names))
    out["target_pred_total"] = int(sum(out[f"target_pred_count_{name}"] for name in class_names))
    return out


def _pseudo_behavior(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    final = rows[-1]
    pseudo_total = max(int(final.get("pseudo_total", 0)), 1)
    target_total = max(int(final.get("target_pred_total", 0)), 1)
    return {
        "available": True,
        "final_pseudo_ratios": {
            name: float(final.get(f"pseudo_count_{name}", 0) / pseudo_total)
            for name in class_names
        },
        "final_target_pred_ratios": {
            name: float(final.get(f"target_pred_count_{name}", 0) / target_total)
            for name in class_names
        },
        "low_or_zero_pseudo_classes": [
            name for name in class_names if int(final.get(f"pseudo_count_{name}", 0)) == 0
        ],
    }


def _plot_counts(rows: list[dict[str, Any]], class_names: list[str], prefix: str, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(7, 4))
    has_any = False
    for name in class_names:
        values = [int(row.get(f"{prefix}{name}", 0)) for row in rows]
        if any(values):
            has_any = True
        plt.plot(epochs, values, marker="o", linewidth=1.5, label=name)
    if not has_any:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
