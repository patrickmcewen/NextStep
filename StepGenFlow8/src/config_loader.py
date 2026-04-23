"""Resolve and load an LLM config profile for the runners."""

import json
import os


def load_llm_config(config_path: str | None, model: str) -> dict:
    """Return the LLM config dict for this invocation.

    If ``config_path`` is given, load that file directly.
    Otherwise load ``configs/<model>.json`` relative to the current working directory.
    """
    path = config_path if config_path is not None else f"configs/{model}.json"
    assert os.path.exists(path), f"LLM config not found: {path}"
    with open(path) as f:
        return json.load(f)
