"""Gold reference cache.

Extracted from orchestrator so synthetic tree-node kernel names (created by
the planner phase) can populate the cache via ``_inject_gold`` without a
circular import between orchestrator and planner.
"""

import json

from src.tools import _validate_functional_mod

_GOLD_CACHE: dict = {}


def _gold_key(kernel_name: str, dims: dict) -> tuple:
    return (kernel_name, json.dumps(dims, sort_keys=True, default=str))


def _get_gold(kernel_name, dims):
    key = _gold_key(kernel_name, dims)
    cached = _GOLD_CACHE.get(key)
    if cached is not None:
        return cached
    config = _validate_functional_mod.load_config()
    gold = _validate_functional_mod.run_reference(kernel_name, dims, config)
    _GOLD_CACHE[key] = gold
    return gold


def _inject_gold(kernel_name: str, dims: dict, gold) -> None:
    """Pre-populate the cache for a synthetic kernel name (planner flow)."""
    key = _gold_key(kernel_name, dims)
    _GOLD_CACHE[key] = gold
