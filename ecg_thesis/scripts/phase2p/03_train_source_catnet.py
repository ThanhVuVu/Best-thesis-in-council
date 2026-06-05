from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
from tqdm import tqdm

from common import (
    build_phase2p_model,
    class_weights_for,
    device_from_torch,
    evaluate_model,
    fit_val_datasets,
    loader,
    load_phase2p_checkpoint,
    maybe_subset,
    save_checkpoint,
    write_eval_outputs,
    write_history,
)
from common import cfg_path, load_phase1_config
from src.training.dro import ClassGroupDROLoss, classifier_discrepancy
from src.training.metrics import classification_metrics
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-fit-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    train_cfg = copy.deepcopy(config["source_pretrain"])
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    device = device_from_torch()
    print(f"Using device: {device}")

    fit_ds, val_ds = fit_val_datasets(config, use_duplicated=bool(train_cfg.get("use_duplicated_source", True)))
    fit_ds = maybe_subset(fit_ds, args.max_fit_samples)
    val_ds = maybe_subset(val_ds, args.max_val_samples)
    fit_loader = loader(fit_ds, int(train_cfg["batch_size"]), True, device)
    val_batch_size = int(config["evaluation"]["batch_size"])
    model = build_phase2p_model(config, device)
    class_weights = class_weights_for(fit_ds, {"use_class_weights": train_cfg.get("use_class_weights", True), "num_classes": 3}, device)
    criterion = ClassGroupDROLoss(num_classes=3, class_weights=class_weights, eta=float(train_cfg.get("dro_eta", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))

    ckpt_dir = ensure_dir(output / "checkpoints")
    log_dir = ensure_dir(output / "logs")
    prefix = str(train_cfg["checkpoint_prefix"])
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_f1 = -1.0
    best_epoch = -1
    stale = 0
    patience = int(train_cfg["early_stopping_patience"])
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses, cls_losses, disc_losses = [], [], []
        y_true, y_pred = [], []
        for x, time_features, y in tqdm(fit_loader, desc=f"phase2p source epoch {epoch}", dynamic_ncols=True):
            x = x.to(device)
            time_features = time_features.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x, time_features, return_all=True)
            cls_loss, dro_stats = criterion(out["logits"], y)
            disc_loss = classifier_discrepancy(out["probabilities1"], out["probabilities2"])
            loss = cls_loss + float(train_cfg.get("lambda_disc", 0.1)) * disc_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            cls_losses.append(float(cls_loss.detach().cpu()))
            disc_losses.append(float(disc_loss.detach().cpu()))
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(out["logits"].argmax(dim=1).detach().cpu().numpy())
        train_metrics = classification_metrics(np.concatenate(y_true), np.concatenate(y_pred))
        val_result = evaluate_model(model, val_ds, device, batch_size=val_batch_size)
        val_metrics = val_result["metrics"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "cls_loss": float(np.mean(cls_losses)),
            "disc_loss": float(np.mean(disc_losses)),
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            **dro_stats,
        }
        history.append(row)
        print(row)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            save_checkpoint(model, optimizer, config, epoch, best_f1, history, best_path)
        else:
            stale += 1
        save_checkpoint(model, optimizer, config, epoch, best_f1, history, latest_path)
        if stale >= patience:
            break

    write_history(history, log_dir / f"{prefix}_train_log.csv")
    summary = {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
    }
    write_json(summary, output / "metrics" / "phase2p_catnet_source_train_summary.json")
    best_model, _ = load_phase2p_checkpoint(best_path, config, device)
    write_eval_outputs(evaluate_model(best_model, val_ds, device, batch_size=val_batch_size), output, "phase2p_source_validation", config["data"]["class_names"])


if __name__ == "__main__":
    main()
