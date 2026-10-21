from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and attach resolved project metadata."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def project_root(config: dict[str, Any]) -> Path:
    return Path(config.get("_project_root", ".")).expanduser().resolve()


def resolve_path(config: dict[str, Any], path: str | Path) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return project_root(config) / path_obj


def data_dir(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, config["data"][key])


def split_alias(config: dict[str, Any], split: str) -> str:
    aliases = config.get("data", {}).get("split_aliases", {}) or {}
    return aliases.get(split, split)


class DotDict(dict):
    """Small optional dot-access helper for ad hoc scripts."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        return value
