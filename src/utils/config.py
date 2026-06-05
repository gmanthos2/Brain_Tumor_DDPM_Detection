"""
Configuration loader utility.
Loads YAML config files and provides dot-notation access.
"""

import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """Hierarchical config object with dot-notation and dict access."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(config_path: str | Path) -> Config:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


def merge_configs(base: Config, override: Config) -> Config:
    """Merge two configs, with override taking precedence."""
    base_dict = base.to_dict()
    override_dict = override.to_dict()

    def deep_merge(d1: dict, d2: dict) -> dict:
        merged = d1.copy()
        for key, value in d2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    return Config(deep_merge(base_dict, override_dict))
