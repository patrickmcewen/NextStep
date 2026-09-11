from pathlib import Path

import yaml

from src.autotune_config_loader import load_autotune_config, resolve_config


def test_resolve_config_recursively_merges_base_configs():
    configs = {
        "base": {
            "hw_config": {
                "hbm_channels": 32,
                "hbm_channel_latency": 2,
            },
            "attempt_budgets": [1.0, 0.5],
            "passes": [{"name": "tiling"}],
        },
        "child": {
            "base": "base",
            "hw_config": {"hbm_channel_latency": 4},
            "passes": [{"name": "parallel"}],
        },
    }

    resolved = resolve_config(configs, "child")

    assert resolved == {
        "hw_config": {
            "hbm_channels": 32,
            "hbm_channel_latency": 4,
        },
        "attempt_budgets": [1.0, 0.5],
        "passes": [{"name": "parallel"}],
    }


def test_resolve_config_accepts_base_chain():
    configs = {
        "hw": {"hw_config": {"hbm_channels": 32}},
        "memory": {"max_on_chip_memory": 1024},
        "combined": {
            "base": ["hw", "memory"],
            "max_on_chip_memory": 2048,
        },
    }

    resolved = resolve_config(configs, "combined")

    assert resolved == {
        "hw_config": {"hbm_channels": 32},
        "max_on_chip_memory": 2048,
    }


def test_load_autotune_config_reads_named_yaml_config(tmp_path: Path):
    path = tmp_path / "autotune_configs.yaml"
    path.write_text(yaml.safe_dump({
        "configs": {
            "base": {"hw_config": {"hbm_channels": 32}},
            "selected": {
                "base": "base",
                "hw_config": {"hbm_init_interval": 2},
            },
        },
    }))

    resolved = load_autotune_config(path, "selected")

    assert resolved == {
        "hw_config": {
            "hbm_channels": 32,
            "hbm_init_interval": 2,
        },
    }


def test_load_autotune_config_treats_missing_suffixless_path_as_yaml_name(tmp_path: Path):
    path = tmp_path / "autotune_configs.yaml"
    path.write_text(yaml.safe_dump({
        "configs": {
            "base": {"hw_config": {"hbm_channels": 32}},
            "autotune_config_general": {
                "base": "base",
                "fewshot": "general",
            },
        },
    }))

    resolved = load_autotune_config(tmp_path / "autotune_config_general")

    assert resolved == {
        "hw_config": {"hbm_channels": 32},
        "fewshot": "general",
    }
