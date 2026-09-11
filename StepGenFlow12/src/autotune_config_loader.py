"""Load autotune configs with recursive YAML inheritance."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


def _merge_recursive(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_recursive(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_config(configs: dict, name: str) -> dict:
    """Resolve one named config, applying ``base`` entries recursively."""
    assert name in configs, (
        f"autotune config {name!r} not found; available: {sorted(configs)}"
    )
    raw = configs[name]
    assert isinstance(raw, dict), (
        f"autotune config {name!r} must be a mapping, got {type(raw).__name__}"
    )

    bases = raw.get("base", [])
    if isinstance(bases, str):
        bases = [bases]
    assert isinstance(bases, list), (
        f"autotune config {name!r}: 'base' must be a string or list of strings"
    )

    resolved: dict = {}
    for base_name in bases:
        assert isinstance(base_name, str), (
            f"autotune config {name!r}: base entries must be strings"
        )
        assert base_name != name, (
            f"autotune config {name!r}: config cannot inherit from itself"
        )
        resolved = _merge_recursive(resolved, resolve_config(configs, base_name))

    override = {k: v for k, v in raw.items() if k != "base"}
    return _merge_recursive(resolved, override)


def load_autotune_config(path: str | Path, config_name: str | None = None) -> dict:
    """Load a JSON config or a named config from an inherited YAML file."""
    path = Path(path)
    if not path.exists() and path.suffix == "":
        yaml_path = path.with_name("autotune_configs.yaml")
        assert yaml_path.exists(), (
            f"autotune config not found: {path}; also looked for named "
            f"config {path.name!r} in {yaml_path}"
        )
        return load_autotune_config(yaml_path, path.name)
    assert path.exists(), f"autotune config not found: {path}"
    if path.suffix == ".json":
        return json.loads(path.read_text())

    assert path.suffix in (".yaml", ".yml"), (
        f"autotune config must be .json, .yaml, or .yml: {path}"
    )
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict), f"{path}: YAML root must be a mapping"
    configs = data.get("configs", data)
    assert isinstance(configs, dict), f"{path}: 'configs' must be a mapping"
    name = config_name or data.get("default")
    assert isinstance(name, str) and name, (
        f"{path}: pass --autotune-config-name or set a non-empty 'default'"
    )
    return resolve_config(configs, name)
