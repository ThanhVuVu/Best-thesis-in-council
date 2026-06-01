from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = [
    "configs",
    "docs",
    "scripts",
    "scripts/phase1",
    "scripts/phase2",
    "scripts/phase3",
    "scripts/phase4a",
    "scripts/phase4b",
    "scripts/phase4c",
    "scripts/phase5_macnn",
    "src",
    "src/data",
    "src/models",
    "src/training",
    "src/utils",
    "src/visualization",
]


def main() -> None:
    errors: list[str] = []
    errors.extend(_check_dirs())
    errors.extend(_check_python_syntax())
    errors.extend(_check_configs())
    if errors:
        print("Repo check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Repo check passed.")
    print(f"Checked root: {ROOT}")
    print(f"Python files: {len(list(ROOT.rglob('*.py')))}")
    print(f"Config files: {len(list((ROOT / 'configs').glob('*.yaml')))}")


def _check_dirs() -> list[str]:
    errors = []
    for rel_path in REQUIRED_DIRS:
        path = ROOT / rel_path
        if not path.is_dir():
            errors.append(f"Missing directory: {rel_path}")
    return errors


def _check_python_syntax() -> list[str]:
    errors = []
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            rel = path.relative_to(ROOT)
            errors.append(f"Syntax error in {rel}: line {exc.lineno}: {exc.msg}")
    return errors


def _check_configs() -> list[str]:
    errors = []
    config_dir = ROOT / "configs"
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(f"Invalid YAML in {path.relative_to(ROOT)}: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"Config is not a mapping: {path.relative_to(ROOT)}")
            continue
        for key in ("paths", "data"):
            if key not in data:
                errors.append(f"Config missing '{key}': {path.relative_to(ROOT)}")
        training_sections = ("training", "source_only", "dann", "source_free")
        if not any(section in data for section in training_sections):
            rel = path.relative_to(ROOT)
            errors.append(f"Config has no training section: {rel}")
    return errors


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
