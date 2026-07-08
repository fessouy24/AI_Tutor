from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_kaggle() -> bool:
    return Path("/kaggle/input").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def is_lightning() -> bool:
    return Path("/teamspace/studios/this_studio").exists() or Path("/teamspace").exists()


def _replace_tokens(value: str, root: Path) -> str:
    lightning_working = Path("/teamspace/studios/this_studio")
    if not lightning_working.exists():
        lightning_working = Path("/teamspace") if Path("/teamspace").exists() else root
    tokens = {
        "{project_root}": str(root),
        "{cwd}": str(Path.cwd()),
        "{kaggle_input}": "/kaggle/input",
        "{kaggle_working}": "/kaggle/working",
        "{lightning_working}": str(lightning_working),
        "{lightning_datasets}": "/teamspace/datasets",
        "{lightning_s3}": "/teamspace/s3_connections",
    }
    for token, replacement in tokens.items():
        value = value.replace(token, replacement)
    return os.path.expandvars(os.path.expanduser(value))


def resolve_path(value: str | Path | None, root: Path | None = None) -> Path | None:
    if value is None:
        return None
    root = root or project_root()
    path = Path(_replace_tokens(str(value), root))
    if path.is_absolute():
        return path
    return root / path


def resolve_path_list(values: list[str] | None, root: Path | None = None) -> list[Path]:
    if not values:
        return []
    return [p for p in (resolve_path(v, root) for v in values) if p is not None]


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    root = project_root()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("project_root", str(root))
    cfg.setdefault("runtime", {})
    cfg["runtime"]["is_kaggle"] = is_kaggle()
    cfg["runtime"]["is_lightning"] = is_lightning()
    cfg["runtime"]["project_root"] = str(root)
    return cfg


def get_path(cfg: dict[str, Any], key: str, default: str | None = None) -> Path:
    value = cfg.get("paths", {}).get(key, default)
    path = resolve_path(value, project_root())
    if path is None:
        raise ValueError(f"Missing path config: paths.{key}")
    return path


def first_existing_path(candidates: list[str | Path]) -> Path | None:
    for candidate in candidates:
        path = resolve_path(candidate, project_root())
        if path and path.exists():
            return path
    return None


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in ["checkpoints_dir", "outputs_dir", "logs_dir", "processed_dir", "rag_index_dir"]:
        value = cfg.get("paths", {}).get(key)
        if not value:
            continue
        path = resolve_path(value, project_root())
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
