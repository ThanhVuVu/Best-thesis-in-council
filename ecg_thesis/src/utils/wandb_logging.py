from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class WandbRun:
    def __init__(self, run: Any | None = None):
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(_wandb_safe(data), step=step)

    def summary_update(self, data: dict[str, Any]) -> None:
        if self.run is None:
            return
        self.run.summary.update(_wandb_safe(data))

    def log_artifact(self, path: str | Path, name: str, artifact_type: str = "artifact") -> None:
        if self.run is None:
            return
        path = Path(path)
        if not path.exists():
            return
        artifact = _wandb().Artifact(name=name, type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_wandb(
    config: dict[str, Any],
    job_type: str,
    default_name: str,
    extra_config: dict[str, Any] | None = None,
) -> WandbRun:
    wandb_cfg = _wandb_config(config)
    if not _as_bool(wandb_cfg.get("enabled", False)):
        return WandbRun()

    wandb = _wandb()
    project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT") or "ecg-thesis"
    entity = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY") or None
    name = wandb_cfg.get("run_name") or default_name
    group = wandb_cfg.get("group") or None
    tags = wandb_cfg.get("tags") or []
    mode = wandb_cfg.get("mode") or os.environ.get("WANDB_MODE") or None

    init_kwargs = {
        "project": project,
        "entity": entity,
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": tags,
        "config": _wandb_safe({**config, **(extra_config or {})}),
        "reinit": True,
    }
    if mode:
        init_kwargs["mode"] = mode
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    return WandbRun(wandb.init(**init_kwargs))


def add_wandb_args(parser) -> None:
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for this run.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=None)


def apply_wandb_overrides(config: dict[str, Any], args) -> None:
    if not hasattr(args, "wandb"):
        return
    logging_cfg = config.setdefault("logging", {})
    wandb_cfg = logging_cfg.setdefault("wandb", {})
    if args.wandb:
        wandb_cfg["enabled"] = True
    for attr, key in (
        ("wandb_project", "project"),
        ("wandb_entity", "entity"),
        ("wandb_run_name", "run_name"),
        ("wandb_group", "group"),
        ("wandb_mode", "mode"),
        ("wandb_tags", "tags"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            wandb_cfg[key] = value


def log_eval_metrics(run: WandbRun, metrics: dict[str, Any], prefix: str) -> None:
    if not run.enabled:
        return
    flat = {
        f"{prefix}/accuracy": metrics.get("accuracy"),
        f"{prefix}/macro_f1": metrics.get("macro_f1"),
    }
    for class_name, values in metrics.get("per_class", {}).items():
        for metric_name, metric_value in values.items():
            flat[f"{prefix}/{class_name}_{metric_name}"] = metric_value
    run.log(flat)


def should_log_artifacts(config: dict[str, Any]) -> bool:
    return _as_bool(_wandb_config(config).get("log_artifacts", False))


def _wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("logging", {}).get("wandb", {}))


def _wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging is enabled, but wandb is not installed. Run: pip install wandb") from exc
    return wandb


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _wandb_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _wandb_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wandb_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
