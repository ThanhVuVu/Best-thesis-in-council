from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.daeac_dataset import DAEACDataset
from src.models.daeac_adversarial import DAEACADDAModel, DAEACCDANModel, DAEACDANNModel, entropy
from src.training.daeac_losses import build_daeac_classification_loss
from src.training.metrics import classification_metrics
from src.training.train_daeac_paper import build_daeac_model
from src.training.v_measure_validation import aggregate_v_measure, ericsson_v_measure, save_v_measure_assignments
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb, should_log_artifacts


def train_daeac_dann(
    source_dataset: DAEACDataset,
    source_val_dataset: DAEACDataset,
    target_dataset,
    target_val_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    train_cfg = config["training"]
    dann_cfg = config["dann"]
    output_dir = Path(output_dir)
    prefix = str(train_cfg.get("checkpoint_prefix", "daeac_dann"))
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    backup_dir = _checkpoint_backup_dir(config)
    model = build_daeac_dann_model(config, device, init_checkpoint=train_cfg.get("init_checkpoint")).to(device)
    wandb_run = init_wandb(config, job_type="train_daeac_dann", default_name=prefix)

    source_loader = _loader(source_dataset, int(train_cfg["source_batch_size"]), True, device)
    target_loader = _loader(target_dataset, int(train_cfg["target_batch_size"]), True, device)
    val_loader = _loader(source_val_dataset, int(train_cfg["source_batch_size"]), False, device)
    target_val_loader = _loader(target_val_dataset, int(train_cfg["target_batch_size"]), False, device)
    cls_loss_fn = _classification_loss(source_dataset, config, train_cfg, device)
    domain_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = _optimizer(
        model,
        train_cfg,
        [
            ("encoder", model.feature_extractor.parameters(), train_cfg.get("encoder_lr")),
            ("classifier", model.classifier.parameters(), train_cfg.get("classifier_lr")),
            ("domain", model.domain_classifier.parameters(), train_cfg.get("domain_lr")),
        ],
    )
    scheduler = _scheduler(optimizer, train_cfg)

    total_epochs = int(train_cfg["epochs"])
    steps_per_epoch = len(target_loader)
    total_steps = max(total_epochs * steps_per_epoch, 1)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    global_step = 0
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"

    for epoch in range(1, total_epochs + 1):
        model.train()
        rows = []
        source_true, source_pred, domain_true, domain_pred = [], [], [], []
        source_iter = cycle(source_loader)
        target_iter = iter(target_loader)
        progress = tqdm(range(steps_per_epoch), desc=f"{prefix} epoch {epoch}/{total_epochs}", dynamic_ncols=True)
        for _ in progress:
            global_step += 1
            lambd = adversarial_lambda(global_step, total_steps, dann_cfg)
            alpha = float(dann_cfg.get("alpha", 1.0))
            if epoch <= int(dann_cfg.get("warmup_epochs", 0)):
                lambd = 0.0
                alpha = 0.0
            x_s, rr_s, y_s = _source_batch_with_optional_rr(next(source_iter), device)
            x_t, rr_t = _target_batch_with_optional_rr(next(target_iter), device)
            x_domain = torch.cat([x_s, x_t], dim=0)
            _cat_optional_rr(rr_s, rr_t)
            y_domain = torch.cat(
                [
                    torch.zeros(x_s.shape[0], dtype=torch.long),
                    torch.ones(x_t.shape[0], dtype=torch.long),
                ],
                dim=0,
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            # One mixed-domain encoder pass keeps shared BatchNorm statistics
            # symmetric and avoids updating them twice with source samples.
            raw_features_domain = model.extract_raw_features(x_domain)
            logits_s = model.class_logits(raw_features_domain[: x_s.shape[0]], rr_s)
            domain_features = model.domain_features(raw_features_domain)
            domain_logits = model.forward_domain_from_features(domain_features, lambd=lambd)
            loss_cls = cls_loss_fn(logits_s, y_s)
            loss_domain = domain_loss_fn(domain_logits, y_domain)
            loss = loss_cls + alpha * loss_domain
            loss.backward()
            optimizer.step()
            _step_scheduler(scheduler, None)

            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "loss_cls": float(loss_cls.detach().cpu()),
                    "loss_domain": float(loss_domain.detach().cpu()),
                    "lambda": float(lambd),
                    "alpha": float(alpha),
                }
            )
            source_true.append(y_s.detach().cpu().numpy())
            source_pred.append(logits_s.argmax(dim=1).detach().cpu().numpy())
            domain_true.append(y_domain.detach().cpu().numpy())
            domain_pred.append(domain_logits.argmax(dim=1).detach().cpu().numpy())
            progress.set_postfix(loss=f"{rows[-1]['loss']:.4f}", dom=f"{rows[-1]['loss_domain']:.4f}", refresh=False)

        val_result = evaluate_daeac_adversarial_model(model, val_loader, device, config["data"]["class_names"], desc=f"{prefix} val")
        train_metrics = classification_metrics(np.concatenate(source_true), np.concatenate(source_pred), config["data"]["class_names"])
        epoch_row = _mean_rows(rows)
        epoch_row.update(
            {
                "epoch": epoch,
                "source_train_accuracy": train_metrics["accuracy"],
                "source_train_macro_f1": train_metrics["macro_f1"],
                "source_val_accuracy": val_result["metrics"]["accuracy"],
                "source_val_macro_f1": val_result["metrics"]["macro_f1"],
                "domain_accuracy": float((np.concatenate(domain_true) == np.concatenate(domain_pred)).mean()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        _update_v_measure(epoch_row, model, val_result, target_val_loader, device, config, ckpt_dir, prefix)
        _step_scheduler(scheduler, epoch_row["source_val_macro_f1"])
        history.append(epoch_row)
        wandb_run.log({f"train/{k}": v for k, v in epoch_row.items() if k != "epoch"}, step=epoch)
        print(f"{prefix} epoch {epoch}: v_measure={epoch_row['v_measure']:.4f}, val_macro_f1={epoch_row['source_val_macro_f1']:.4f}, domain_acc={epoch_row['domain_accuracy']:.4f}")
        best_f1, best_epoch, stale_epochs = _save_epoch_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            "daeac_dann",
            epoch_row,
            history,
            best_f1,
            best_epoch,
            stale_epochs,
            epoch,
            best_path,
            latest_path,
            backup_dir,
        )
        if epoch >= int(config.get("validation", {}).get("min_epochs", 10)) and stale_epochs >= int(config.get("validation", {}).get("patience", 10)):
            break

    summary = _finish_training(prefix, best_path, latest_path, history, best_epoch, best_f1, log_dir, backup_dir, wandb_run, config)
    return summary


def train_daeac_cdan(
    source_dataset: DAEACDataset,
    source_val_dataset: DAEACDataset,
    target_dataset,
    target_val_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    train_cfg = config["training"]
    cdan_cfg = config["cdan"]
    output_dir = Path(output_dir)
    prefix = str(train_cfg.get("checkpoint_prefix", "daeac_cdan_e"))
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    backup_dir = _checkpoint_backup_dir(config)
    model = build_daeac_cdan_model(config, device, init_checkpoint=train_cfg.get("init_checkpoint")).to(device)
    wandb_run = init_wandb(config, job_type="train_daeac_cdan", default_name=prefix)

    source_loader = _loader(source_dataset, int(train_cfg["source_batch_size"]), True, device)
    target_loader = _loader(target_dataset, int(train_cfg["target_batch_size"]), True, device)
    val_loader = _loader(source_val_dataset, int(train_cfg["source_batch_size"]), False, device)
    target_val_loader = _loader(target_val_dataset, int(train_cfg["target_batch_size"]), False, device)
    cls_loss_fn = _classification_loss(source_dataset, config, train_cfg, device)
    domain_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = _optimizer(
        model,
        train_cfg,
        [
            ("encoder", model.feature_extractor.parameters(), train_cfg.get("encoder_lr")),
            ("classifier", model.classifier.parameters(), train_cfg.get("classifier_lr")),
            ("domain", model.domain_classifier.parameters(), train_cfg.get("domain_lr")),
        ],
    )
    scheduler = _scheduler(optimizer, train_cfg)

    total_epochs = int(train_cfg["epochs"])
    steps_per_epoch = len(target_loader)
    total_steps = max(total_epochs * steps_per_epoch, 1)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    global_step = 0
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"

    for epoch in range(1, total_epochs + 1):
        model.train()
        rows = []
        source_true, source_pred, domain_true, domain_pred = [], [], [], []
        source_iter = cycle(source_loader)
        target_iter = iter(target_loader)
        progress = tqdm(range(steps_per_epoch), desc=f"{prefix} epoch {epoch}/{total_epochs}", dynamic_ncols=True)
        for _ in progress:
            global_step += 1
            lambd = adversarial_lambda(global_step, total_steps, cdan_cfg)
            lambda_base = float(cdan_cfg.get("lambda_base", cdan_cfg.get("alpha", 1.0)))
            if epoch <= int(cdan_cfg.get("warmup_epochs", 0)):
                lambd = 0.0
                lambda_base = 0.0
            x_s, rr_s, y_s = _source_batch_with_optional_rr(next(source_iter), device)
            x_t, rr_t = _target_batch_with_optional_rr(next(target_iter), device)
            rr_all = _cat_optional_rr(rr_s, rr_t)

            optimizer.zero_grad(set_to_none=True)
            # Source and target share one encoder and therefore one mixed BN
            # update per iteration; sequential passes would bias running stats
            # toward whichever domain is forwarded last.
            raw_features_all = model.extract_raw_features(torch.cat([x_s, x_t], dim=0))
            features_all = model.domain_features(raw_features_all)
            raw_s = raw_features_all[: x_s.shape[0]]
            raw_t = raw_features_all[x_s.shape[0] :]
            f_s = features_all[: x_s.shape[0]]
            f_t = features_all[x_s.shape[0] :]
            rr_s_for_logits = rr_all[: x_s.shape[0]] if rr_all is not None else None
            rr_t_for_logits = rr_all[x_s.shape[0] :] if rr_all is not None else None
            logits_s = model.class_logits(raw_s, rr_s_for_logits)
            logits_t = model.class_logits(raw_t, rr_t_for_logits)
            loss_cls = cls_loss_fn(logits_s, y_s)
            logits_all = torch.cat([logits_s, logits_t], dim=0)
            domain_logits = model.forward_domain_from_features(
                features_all,
                logits_all,
                lambd=lambd,
                detach_softmax=bool(cdan_cfg.get("detach_softmax_in_T", True)),
            )
            y_domain = torch.cat(
                [
                    torch.ones(f_s.shape[0], 1, dtype=torch.float32),
                    torch.zeros(f_t.shape[0], 1, dtype=torch.float32),
                ],
                dim=0,
            ).to(device)
            probs_all = torch.softmax(logits_all, dim=1)
            loss_domain = _cdan_domain_loss(domain_loss_fn, domain_logits, y_domain, entropy(probs_all), f_s.shape[0], cdan_cfg)
            loss = loss_cls + lambda_base * loss_domain
            loss.backward()
            optimizer.step()
            _step_scheduler(scheduler, None)

            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "loss_cls": float(loss_cls.detach().cpu()),
                    "loss_domain": float(loss_domain.detach().cpu()),
                    "source_entropy": float(entropy(torch.softmax(logits_s, dim=1)).mean().detach().cpu()),
                    "target_entropy": float(entropy(torch.softmax(logits_t, dim=1)).mean().detach().cpu()),
                    "lambda": float(lambd),
                    "lambda_base": float(lambda_base),
                }
            )
            source_true.append(y_s.detach().cpu().numpy())
            source_pred.append(logits_s.argmax(dim=1).detach().cpu().numpy())
            domain_true.append(y_domain.detach().cpu().numpy().reshape(-1))
            domain_pred.append((torch.sigmoid(domain_logits) >= 0.5).long().detach().cpu().numpy().reshape(-1))
            progress.set_postfix(loss=f"{rows[-1]['loss']:.4f}", dom=f"{rows[-1]['loss_domain']:.4f}", refresh=False)

        val_result = evaluate_daeac_adversarial_model(model, val_loader, device, config["data"]["class_names"], desc=f"{prefix} val")
        train_metrics = classification_metrics(np.concatenate(source_true), np.concatenate(source_pred), config["data"]["class_names"])
        epoch_row = _mean_rows(rows)
        epoch_row.update(
            {
                "epoch": epoch,
                "method": str(cdan_cfg.get("method", "cdan_e")),
                "source_train_accuracy": train_metrics["accuracy"],
                "source_train_macro_f1": train_metrics["macro_f1"],
                "source_val_accuracy": val_result["metrics"]["accuracy"],
                "source_val_macro_f1": val_result["metrics"]["macro_f1"],
                "domain_accuracy": float((np.concatenate(domain_true) == np.concatenate(domain_pred)).mean()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        _update_v_measure(epoch_row, model, val_result, target_val_loader, device, config, ckpt_dir, prefix)
        _step_scheduler(scheduler, epoch_row["source_val_macro_f1"])
        history.append(epoch_row)
        wandb_run.log({f"train/{k}": v for k, v in epoch_row.items() if k != "epoch"}, step=epoch)
        print(f"{prefix} epoch {epoch}: v_measure={epoch_row['v_measure']:.4f}, val_macro_f1={epoch_row['source_val_macro_f1']:.4f}, domain_acc={epoch_row['domain_accuracy']:.4f}")
        best_f1, best_epoch, stale_epochs = _save_epoch_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            "daeac_cdan",
            epoch_row,
            history,
            best_f1,
            best_epoch,
            stale_epochs,
            epoch,
            best_path,
            latest_path,
            backup_dir,
        )
        if epoch >= int(config.get("validation", {}).get("min_epochs", 10)) and stale_epochs >= int(config.get("validation", {}).get("patience", 10)):
            break

    return _finish_training(prefix, best_path, latest_path, history, best_epoch, best_f1, log_dir, backup_dir, wandb_run, config)


def train_daeac_adda(
    source_dataset: DAEACDataset,
    source_val_dataset: DAEACDataset,
    target_dataset,
    target_val_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    train_cfg = config["training"]
    output_dir = Path(output_dir)
    prefix = str(train_cfg.get("checkpoint_prefix", "daeac_adda"))
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    backup_dir = _checkpoint_backup_dir(config)
    model = build_daeac_adda_model(config, device, init_checkpoint=train_cfg.get("init_checkpoint")).to(device)
    wandb_run = init_wandb(config, job_type="train_daeac_adda", default_name=prefix)

    source_loader = _loader(source_dataset, int(train_cfg["batch_size"]), True, device)
    target_loader = _loader(target_dataset, int(train_cfg["batch_size"]), True, device)
    val_loader = _loader(source_val_dataset, int(train_cfg["batch_size"]), False, device)
    target_val_loader = _loader(target_val_dataset, int(train_cfg["batch_size"]), False, device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer_d = torch.optim.AdamW(
        model.domain_discriminator.parameters(),
        lr=float(train_cfg["discriminator_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    optimizer_m = torch.optim.AdamW(
        [p for p in model.target_encoder.parameters() if p.requires_grad],
        lr=float(train_cfg["target_encoder_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler_m = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_m, mode="max", factor=0.5, patience=3)

    total_epochs = int(train_cfg["epochs"])
    steps_per_epoch = len(target_loader)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    class_names = list(config["data"]["class_names"])

    for epoch in range(1, total_epochs + 1):
        model.train()
        rows = []
        domain_true, domain_pred = [], []
        target_pseudo_counts = np.zeros(int(config["data"]["num_classes"]), dtype=np.int64)
        source_iter = cycle(source_loader)
        target_iter = iter(target_loader)
        progress = tqdm(range(steps_per_epoch), desc=f"{prefix} epoch {epoch}/{total_epochs}", dynamic_ncols=True)
        for _ in progress:
            x_s = _batch_x(next(source_iter), device)
            x_t = _target_batch(next(target_iter), device)

            _set_requires_grad(model.domain_discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                f_s = model.forward_source_features(x_s)
                f_t = model.forward_target_features(x_t).detach()
            logits_s = model.forward_domain_from_features(f_s)
            logits_t = model.forward_domain_from_features(f_t)
            y_s = torch.ones_like(logits_s)
            y_t = torch.zeros_like(logits_t)
            loss_d = loss_fn(logits_s, y_s) + loss_fn(logits_t, y_t)
            loss_d.backward()
            optimizer_d.step()

            with torch.no_grad():
                prob_s = torch.sigmoid(logits_s)
                prob_t = torch.sigmoid(logits_t)
                pred_s = (prob_s >= 0.5).long()
                pred_t = (prob_t >= 0.5).long()
                domain_true.append(torch.cat([torch.ones_like(pred_s), torch.zeros_like(pred_t)]).cpu().numpy())
                domain_pred.append(torch.cat([pred_s, pred_t]).cpu().numpy())
                target_as_source = float(pred_t.float().mean().cpu())
                source_prob = float(prob_s.mean().cpu())
                target_prob = float(prob_t.mean().cpu())

            _set_requires_grad(model.domain_discriminator, False)
            model.domain_discriminator.eval()
            optimizer_m.zero_grad(set_to_none=True)
            f_t = model.forward_target_features(x_t)
            logits_t_for_m = model.forward_domain_from_features(f_t)
            loss_m = loss_fn(logits_t_for_m, torch.ones_like(logits_t_for_m))
            loss_m.backward()
            optimizer_m.step()
            _set_requires_grad(model.domain_discriminator, True)
            model.domain_discriminator.train()

            with torch.no_grad():
                target_logits = model.class_logits(f_t.detach())
                target_probs = torch.softmax(target_logits, dim=1)
                target_entropy = float(entropy(target_probs).mean().cpu())
                pseudo = target_probs.argmax(dim=1).detach().cpu().numpy()
                target_pseudo_counts += np.bincount(pseudo, minlength=len(target_pseudo_counts))

            rows.append(
                {
                    "loss_d": float(loss_d.detach().cpu()),
                    "loss_m": float(loss_m.detach().cpu()),
                    "target_as_source_rate": target_as_source,
                    "source_domain_prob_mean": source_prob,
                    "target_domain_prob_mean": target_prob,
                    "target_prediction_entropy": target_entropy,
                }
            )
            progress.set_postfix(loss_d=f"{rows[-1]['loss_d']:.4f}", loss_m=f"{rows[-1]['loss_m']:.4f}", refresh=False)

        val_result = evaluate_daeac_adversarial_model(model, val_loader, device, class_names, desc=f"{prefix} val")
        epoch_row = _mean_rows(rows)
        epoch_row.update(
            {
                "epoch": epoch,
                "domain_accuracy": float((np.concatenate(domain_true) == np.concatenate(domain_pred)).mean()),
                "source_val_accuracy": val_result["metrics"]["accuracy"],
                "source_val_macro_f1": val_result["metrics"]["macro_f1"],
                "target_encoder_lr": float(optimizer_m.param_groups[0]["lr"]),
                "discriminator_lr": float(optimizer_d.param_groups[0]["lr"]),
            }
        )
        total_pseudo = max(1, int(target_pseudo_counts.sum()))
        for idx, name in enumerate(class_names):
            epoch_row[f"target_pseudo_count_{name}"] = int(target_pseudo_counts[idx])
            epoch_row[f"target_pseudo_rate_{name}"] = float(target_pseudo_counts[idx] / total_pseudo)
        _update_v_measure(epoch_row, model, val_result, target_val_loader, device, config, ckpt_dir, prefix)
        scheduler_m.step(epoch_row["source_val_macro_f1"])
        history.append(epoch_row)
        wandb_run.log({f"train/{k}": v for k, v in epoch_row.items() if k != "epoch"}, step=epoch)
        print(f"{prefix} epoch {epoch}: v_measure={epoch_row['v_measure']:.4f}, val_macro_f1={epoch_row['source_val_macro_f1']:.4f}, domain_acc={epoch_row['domain_accuracy']:.4f}")
        best_f1, best_epoch, stale_epochs = _save_epoch_checkpoint(
            model,
            {"discriminator": optimizer_d, "target_encoder": optimizer_m},
            scheduler_m,
            config,
            "daeac_adda",
            epoch_row,
            history,
            best_f1,
            best_epoch,
            stale_epochs,
            epoch,
            best_path,
            latest_path,
            backup_dir,
        )
        if epoch >= int(config.get("validation", {}).get("min_epochs", 10)) and stale_epochs >= int(config.get("validation", {}).get("patience", 10)):
            break

    return _finish_training(prefix, best_path, latest_path, history, best_epoch, best_f1, log_dir, backup_dir, wandb_run, config)


@torch.no_grad()
def evaluate_daeac_adversarial_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    desc: str = "evaluate",
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    y_true, y_pred, probs_all, features_all, logits_all = [], [], [], [], []
    for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
        x, rr_features, y = _source_batch_with_optional_rr(batch, device)
        if rr_features is None:
            logits, features = model(x, return_embedding=True)
        else:
            logits, features = model(x, rr_features=rr_features, return_embedding=True)
        probs = torch.softmax(logits, dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(probs.argmax(dim=1).detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())
        features_all.append(features.detach().cpu().numpy())
    true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=np.int64)
    return {
        "y_true": true,
        "y_pred": pred,
        "probabilities": np.concatenate(probs_all) if probs_all else np.zeros((0, len(class_names)), dtype=np.float32),
        "logits": np.concatenate(logits_all) if logits_all else np.zeros((0, len(class_names)), dtype=np.float32),
        "features": np.concatenate(features_all) if features_all else np.zeros((0, 0), dtype=np.float32),
        "metrics": _daeac_metrics(true, pred, class_names),
    }


def load_daeac_adversarial_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = _torch_load(Path(checkpoint_path), device)
    config = checkpoint["config"]
    method = str(checkpoint.get("method", "")).lower()
    if method == "daeac_dann":
        model = build_daeac_dann_model(config, device, init_checkpoint=None)
    elif method == "daeac_cdan":
        model = build_daeac_cdan_model(config, device, init_checkpoint=None)
    elif method == "daeac_adda":
        model = build_daeac_adda_model(config, device, init_checkpoint=None)
    else:
        raise ValueError(f"Unsupported DAEAC adversarial checkpoint method: {method!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def build_daeac_dann_model(config: dict[str, Any], device: torch.device, init_checkpoint: str | Path | None = None) -> DAEACDANNModel:
    base = _base_daeac_from_checkpoint(config, device, init_checkpoint)
    cfg = config.get("dann", {})
    return DAEACDANNModel(
        feature_extractor=base.feature_extractor,
        classifier=base.classifier,
        feature_dim=int(base.feature_dim),
        num_classes=int(config["data"]["num_classes"]),
        num_domains=int(cfg.get("num_domains", 2)),
        domain_hidden_dim=cfg.get("domain_hidden_dim"),
        dropout=float(cfg.get("dropout", config["model"].get("dropout", 0.3))),
    )


def build_daeac_cdan_model(config: dict[str, Any], device: torch.device, init_checkpoint: str | Path | None = None) -> DAEACCDANModel:
    base = _base_daeac_from_checkpoint(config, device, init_checkpoint)
    cfg = config.get("cdan", {})
    return DAEACCDANModel(
        feature_extractor=base.feature_extractor,
        classifier=base.classifier,
        feature_dim=int(base.feature_dim),
        num_classes=int(config["data"]["num_classes"]),
        conditioning=str(cfg.get("conditioning", "auto")),
        randomized_threshold=int(cfg.get("randomized_threshold", 4096)),
        random_dim=int(cfg.get("random_dim", 1024)),
        domain_hidden_dim=cfg.get("domain_hidden_dim"),
        dropout=float(cfg.get("dropout", config["model"].get("dropout", 0.3))),
    )


def build_daeac_adda_model(config: dict[str, Any], device: torch.device, init_checkpoint: str | Path | None = None) -> DAEACADDAModel:
    base = _base_daeac_from_checkpoint(config, device, init_checkpoint)
    cfg = config.get("adda", {})
    return DAEACADDAModel(
        source_encoder=base.feature_extractor,
        classifier=base.classifier,
        feature_dim=int(config["model"]["feature_dim"]),
        discriminator_hidden_dims=cfg.get("discriminator_hidden_dims", cfg.get("discriminator_hidden_dim", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


def adversarial_lambda(step: int, total_steps: int, cfg: dict[str, Any]) -> float:
    schedule = str(cfg.get("lambda_schedule", "progressive")).lower()
    if schedule == "fixed":
        return float(cfg.get("fixed_lambda", 1.0))
    p = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def _base_daeac_from_checkpoint(config: dict[str, Any], device: torch.device, checkpoint_value: str | Path | None):
    model = build_daeac_model(config, device)
    if checkpoint_value in (None, "", "null", "None"):
        return model
    checkpoint_path = _resolve_path(checkpoint_value, config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DAEAC base checkpoint not found: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _classification_loss(dataset: DAEACDataset, config: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    weights = _class_weights(dataset, config, cfg, device) if bool(cfg.get("use_class_weights", True)) else None
    return build_daeac_classification_loss(cfg, int(config["data"]["num_classes"]), weights).to(device)


def _class_weights(dataset, config: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    labels = _dataset_labels(dataset)
    if labels is None:
        return None
    num_classes = int(config["data"]["num_classes"])
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    mode = str(cfg.get("class_weight_mode", "inverse")).lower()
    if mode in {"inverse", "balanced"}:
        pass
    elif mode in {"sqrt", "sqrt_inverse", "sqrt_balanced"}:
        weights = np.sqrt(weights)
    else:
        raise ValueError(f"Unsupported class_weight_mode: {mode}")
    if cfg.get("class_weight_cap") is not None:
        weights = np.minimum(weights, float(cfg["class_weight_cap"]))
    for name, scale in dict(cfg.get("class_weight_scales", {})).items():
        class_names = list(config["data"]["class_names"])
        if str(name) not in class_names:
            raise ValueError(f"Unknown class in class_weight_scales: {name}")
        weights[class_names.index(str(name))] *= float(scale)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _dataset_labels(dataset) -> np.ndarray | None:
    if isinstance(dataset, Subset):
        labels = _dataset_labels(dataset.dataset)
        return None if labels is None else labels[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, "y"):
        return dataset.y
    return None


def _loader(dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")


def _source_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, torch.Tensor):
        raise ValueError("Source batch must include labels.")
    x, y = batch[0], batch[1]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _source_batch_with_optional_rr(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if isinstance(batch, torch.Tensor):
        raise ValueError("Source batch must include labels.")
    if len(batch) >= 3:
        x, rr_features, y = batch[0], batch[1], batch[2]
        return x.to(device, non_blocking=True), rr_features.to(device, non_blocking=True), y.to(device, non_blocking=True)
    x, y = batch[0], batch[1]
    return x.to(device, non_blocking=True), None, y.to(device, non_blocking=True)


def _target_batch(batch, device: torch.device) -> torch.Tensor:
    return _batch_x(batch, device)


def _target_batch_with_optional_rr(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True), None
    if len(batch) >= 3:
        x, rr_features = batch[0], batch[1]
        return x.to(device, non_blocking=True), rr_features.to(device, non_blocking=True)
    return batch[0].to(device, non_blocking=True), None


def _cat_optional_rr(rr_source: torch.Tensor | None, rr_target: torch.Tensor | None) -> torch.Tensor | None:
    if rr_source is None and rr_target is None:
        return None
    if rr_source is None or rr_target is None:
        raise ValueError("Source and target batches must both include rr_features for RR late-fusion adversarial training.")
    return torch.cat([rr_source, rr_target], dim=0)


def _batch_x(batch, device: torch.device) -> torch.Tensor:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    return x.to(device, non_blocking=True)


def _optimizer(model: torch.nn.Module, cfg: dict[str, Any], groups: list[tuple[str, Any, Any]]) -> torch.optim.Optimizer:
    base_lr = float(cfg["lr"])
    params = []
    seen: set[int] = set()
    for name, parameters, lr_value in groups:
        unique = []
        for param in parameters:
            if param.requires_grad and id(param) not in seen:
                unique.append(param)
                seen.add(id(param))
        if unique:
            params.append({"params": unique, "lr": float(lr_value if lr_value is not None else base_lr), "name": name})
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    return optimizer_cls(params, lr=base_lr, weight_decay=float(cfg["weight_decay"]))


def _scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]):
    name = str(cfg.get("scheduler", "step")).lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.get("lr_decay_every_steps", 200)),
        gamma=float(cfg.get("lr_decay_gamma", 0.99)),
    )


def _step_scheduler(scheduler, metric: float | None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is not None:
            scheduler.step(metric)
        return
    if metric is None:
        scheduler.step()


def _cdan_domain_loss(domain_loss_fn, domain_logits, domain_labels, entropy_values, source_batch_size: int, cfg: dict[str, Any]) -> torch.Tensor:
    loss_each = domain_loss_fn(domain_logits, domain_labels)
    method = str(cfg.get("method", "cdan_e")).lower()
    if method in {"cdan_e", "cdan+e", "cdane"}:
        weights = (1.0 + torch.exp(-entropy_values.detach())).view(-1, 1)
        if bool(cfg.get("normalize_entropy_weights", True)):
            weights = weights.clone()
            weights[:source_batch_size] = weights[:source_batch_size] / weights[:source_batch_size].mean().clamp_min(1e-6)
            weights[source_batch_size:] = weights[source_batch_size:] / weights[source_batch_size:].mean().clamp_min(1e-6)
        return (weights * loss_each).mean()
    if method == "cdan":
        return loss_each.mean()
    raise ValueError(f"Unsupported CDAN method: {method!r}")


def _daeac_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_pred, class_names)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    per_class = {}
    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        se = tp / max(tp + fn, 1)
        pp = tp / max(tp + fp, 1)
        f1 = 2 * se * pp / max(se + pp, 1e-12)
        per_class[name] = {"Se": float(se), "Pp": float(pp), "F1": float(f1), "support": int(cm[idx, :].sum())}
    metrics["paper_metrics"] = {"accuracy": metrics["accuracy"], "per_class": per_class}
    return metrics


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


@torch.no_grad()
def _update_v_measure(epoch_row, model, source_result, target_val_loader, device, config, ckpt_dir, prefix) -> None:
    model.eval()
    target_logits = []
    for batch in target_val_loader:
        x_t, rr_t = _target_batch_with_optional_rr(batch, device)
        if rr_t is None:
            logits, _ = model(x_t, return_embedding=True)
        else:
            logits, _ = model(x_t, rr_features=rr_t, return_embedding=True)
        target_logits.append(logits.detach().cpu().numpy())
    result = ericsson_v_measure(
        source_result["logits"],
        source_result["y_true"],
        np.concatenate(target_logits),
        num_classes=int(config["data"]["num_classes"]),
        random_state=int(config.get("seed", 42)),
        beta=float(config.get("validation", {}).get("beta", 1.0)),
    )
    epoch_row.update(aggregate_v_measure(result))
    save_v_measure_assignments(ckpt_dir / f"{prefix}_latest_v_measure_assignments.npz", result)


def _save_epoch_checkpoint(
    model,
    optimizer,
    scheduler,
    config,
    method: str,
    row: dict[str, Any],
    history: list[dict[str, Any]],
    best_f1: float,
    best_epoch: int,
    stale_epochs: int,
    epoch: int,
    best_path: Path,
    latest_path: Path,
    backup_dir: Path | None,
) -> tuple[float, int, int]:
    current_f1 = float(row["v_measure"])
    min_delta = float(config.get("validation", {}).get("min_delta", 1e-4))
    if bool(row.get("valid", False)) and current_f1 > best_f1 + min_delta:
        best_f1 = current_f1
        best_epoch = epoch
        stale_epochs = 0
        _save_checkpoint(_payload(model, optimizer, scheduler, config, method, epoch, row, best_f1, best_epoch, stale_epochs, history), best_path, backup_dir)
        prefix = best_path.stem.removesuffix("_best")
        latest_assignments = best_path.parent / f"{prefix}_latest_v_measure_assignments.npz"
        if latest_assignments.exists():
            shutil.copy2(latest_assignments, best_path.parent / f"{prefix}_best_v_measure_assignments.npz")
    else:
        stale_epochs += 1
    _save_checkpoint(_payload(model, optimizer, scheduler, config, method, epoch, row, best_f1, best_epoch, stale_epochs, history), latest_path, backup_dir)
    return best_f1, best_epoch, stale_epochs


def _payload(model, optimizer, scheduler, config, method: str, epoch: int, row: dict[str, Any], best_f1: float, best_epoch: int, stale_epochs: int, history):
    state_dict = model.state_dict()
    if isinstance(optimizer, dict):
        optimizer_state = {key: value.state_dict() for key, value in optimizer.items()}
    else:
        optimizer_state = optimizer.state_dict()
    return {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "method": method,
        "model_name": method,
        "epoch": int(epoch),
        "metrics": row,
        "best_metric": float(best_f1),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale_epochs),
        "history": history,
        "config": config,
        "class_names": config["data"]["class_names"],
        "model_state_fingerprint": _state_dict_fingerprint(state_dict),
    }


def _save_checkpoint(payload: dict[str, Any], path: str | Path, backup_dir: Path | None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
    print(f"Saved checkpoint {path} (epoch={payload['epoch']}, best_epoch={payload['best_epoch']}, best_metric={payload['best_metric']:.6f})")
    if backup_dir is not None:
        ensure_dir(backup_dir)
        shutil.copy2(path, backup_dir / path.name)


def _finish_training(prefix, best_path, latest_path, history, best_epoch, best_f1, log_dir, backup_dir, wandb_run, config):
    train_log = log_dir / f"{prefix}_train_log.csv"
    _write_history_csv(history, train_log)
    if backup_dir is not None and train_log.exists():
        shutil.copy2(train_log, backup_dir / train_log.name)
    summary = {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_backup_dir": str(backup_dir) if backup_dir is not None else None,
        "best_epoch": int(best_epoch),
        "best_v_measure": float(best_f1),
        "selection_policy": "maximum_ericsson_v_measure_source_val_plus_target_val_logits",
        "history": history,
    }
    wandb_run.summary_update(summary)
    if should_log_artifacts(config):
        wandb_run.log_artifact(best_path, name=f"{prefix}_best", artifact_type="model")
        wandb_run.log_artifact(train_log, name=f"{prefix}_train_log", artifact_type="train_log")
    wandb_run.finish()
    return summary


def _write_history_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(value)


def _resolve_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config.get("_base_dir", ".")) / path).resolve()


def _checkpoint_backup_dir(config: dict[str, Any]) -> Path | None:
    value = os.environ.get("ECG_PHASE6_ADV_CHECKPOINT_BACKUP_DIR") or config.get("paths", {}).get("checkpoint_backup_dir")
    if value in (None, "", "null", "None"):
        return None
    return _resolve_path(value, config)


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]
