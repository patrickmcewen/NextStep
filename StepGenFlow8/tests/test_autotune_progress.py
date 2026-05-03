import json
from pathlib import Path

from src.autotune import _write_progress


def test_write_progress_creates_file_with_expected_schema(tmp_path: Path):
    _write_progress(tmp_path, baseline_cycles=1000, best_cycles=900,
                    turn=2, last_status="NEW_BEST")
    data = json.loads((tmp_path / "progress.json").read_text())
    assert data == {
        "baseline_cycles": 1000,
        "best_cycles": 900,
        "turn": 2,
        "last_status": "NEW_BEST",
    }


def test_write_progress_overwrites(tmp_path: Path):
    _write_progress(tmp_path, baseline_cycles=1000, best_cycles=1000,
                    turn=-1, last_status="BASELINE")
    _write_progress(tmp_path, baseline_cycles=1000, best_cycles=950,
                    turn=0, last_status="NEW_BEST")
    data = json.loads((tmp_path / "progress.json").read_text())
    assert data["best_cycles"] == 950
    assert data["turn"] == 0
    assert data["last_status"] == "NEW_BEST"
