"""Gold reference cache, extracted from orchestrator to break a circular
import between ``src.orchestrator`` and ``src.subdivide``.

``_get_gold`` memoizes the deterministic PyTorch reference output for
``(kernel_name, dims)`` so heavy kernels (multi-GiB tensors) are not
re-allocated on every gate. ``_inject_gold`` lets the subdivide flow
pre-populate the cache for synthetic kernel names so the standard
``_compare_against_gold`` path works for sub-tasks unchanged.
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
    """Pre-populate the cache for a synthetic kernel name (subdivide flow)."""
    key = _gold_key(kernel_name, dims)
    _GOLD_CACHE[key] = gold
