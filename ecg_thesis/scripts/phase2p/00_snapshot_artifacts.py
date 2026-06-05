from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = cfg_path(config, "paths", "output_dir")
    backup_dir = _backup_dir(config, args.backup_dir)
    ensure_dir(backup_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.final and Path("/kaggle/working").exists():
        zip_path = Path("/kaggle/working/phase2p_catnet_paper_uda_final.zip")
    else:
        zip_path = backup_dir / f"phase2p_{_safe(args.stage)}_{timestamp}_snapshot.zip"

    files = _collect_files(config, output)
    manifest = {
        "stage": args.stage,
        "timestamp_utc": timestamp,
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output),
        "backup_dir": str(backup_dir),
        "file_count": len(files),
        "files": [{"path": str(path), "size_bytes": int(path.stat().st_size)} for path in files if path.exists()],
    }
    manifest_path = backup_dir / f"phase2p_{_safe(args.stage)}_{timestamp}_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname="snapshot_manifest.json")
        for path in files:
            if path.exists() and path.is_file():
                zf.write(path, arcname=_arcname(path, config))
    print(f"Wrote snapshot: {zip_path}")
    print(f"Wrote manifest: {manifest_path}")


def _backup_dir(config: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    env_dir = os.environ.get("PHASE2P_ARTIFACT_BACKUP_DIR")
    if env_dir:
        return Path(env_dir)
    configured = config.get("paths", {}).get("artifact_backup_dir")
    if configured:
        return cfg_path(config, "paths", "artifact_backup_dir")
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/phase2p_backup")
    if Path("/content").exists():
        raise RuntimeError(
            "Colab runtime detected but no backup dir is configured. "
            "Mount Drive and pass --backup-dir /content/drive/MyDrive/thesis-runs/phase2p_catnet_paper_uda."
        )
    return cfg_path(config, "paths", "output_dir") / "phase2p_backup"


def _collect_files(config: dict, output: Path) -> list[Path]:
    roots_and_patterns = [
        (output / "checkpoints", "phase2p*.pt"),
        (output / "metrics", "phase2p*.json"),
        (output / "logs", "phase2p*.csv"),
        (output / "predictions", "phase2p*.csv"),
        (output / "prototypes", "phase2p*.pt"),
        (output, "phase2p_catnet_paper_uda_report.md"),
    ]
    files = []
    for root, pattern in roots_and_patterns:
        if root.exists():
            files.extend(sorted(root.glob(pattern)))
    processed = cfg_path(config, "paths", "processed_dir")
    if processed.exists():
        files.extend(sorted(processed.glob("*summary*.json")))
        files.extend(sorted(processed.glob("*manifest*.json")))
    config_path = Path(config["_config_path"])
    files.append(config_path)
    phase2p_scripts = Path(__file__).resolve().parent
    files.extend(sorted(phase2p_scripts.glob("*.py")))
    return _dedupe([path for path in files if path.exists() and path.is_file()])


def _arcname(path: Path, config: dict) -> str:
    candidates = [Path(config["_base_dir"]).resolve(), Path.cwd().resolve(), path.parent.resolve()]
    for base in candidates:
        try:
            return str(path.resolve().relative_to(base)).replace("\\", "/")
        except ValueError:
            continue
    return path.name


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def _dedupe(paths: list[Path]) -> list[Path]:
    out = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


if __name__ == "__main__":
    main()
