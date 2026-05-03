import json
from pathlib import Path

from src.orchestrator import _load_autotune_progress


def test_load_autotune_progress_returns_dict_when_file_exists(tmp_path: Path):
    (tmp_path / "progress.json").write_text(json.dumps({
        "baseline_cycles": 1000, "best_cycles": 900,
        "turn": 3, "last_status": "NEW_BEST",
    }))
    out = _load_autotune_progress(tmp_path)
    assert out == {
        "baseline_cycles": 1000, "best_cycles": 900,
        "turn": 3, "last_status": "NEW_BEST",
    }


def test_load_autotune_progress_returns_empty_dict_when_missing(tmp_path: Path):
    assert _load_autotune_progress(tmp_path) == {}


def test_load_autotune_progress_returns_empty_dict_when_dir_missing(tmp_path: Path):
    assert _load_autotune_progress(tmp_path / "does-not-exist") == {}
