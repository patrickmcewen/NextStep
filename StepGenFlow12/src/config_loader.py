"""Resolve and load an LLM config profile for the runners."""

import json
from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_llm_config(config_path, model: str) -> dict:
    """Return the LLM config dict for this invocation.

    If ``config_path`` is given, load that file directly.
    Otherwise load ``<NextStep>/configs/<model>.json`` resolved relative to
    this file (so the loader works from any CWD).
    """
    path = Path(config_path) if config_path is not None else _CONFIGS_DIR / f"{model}.json"
    assert path.exists(), f"LLM config not found: {path}"
    with open(path) as f:
        return json.load(f)
