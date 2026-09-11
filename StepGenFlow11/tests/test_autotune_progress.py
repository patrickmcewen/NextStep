import json
from pathlib import Path

from src.autotune import _write_progress


def _mi(on_chip: int, off_chip: int) -> dict:
    return {"on_chip_bytes": on_chip, "off_chip_bytes": off_chip,
            "pmu_buffer_bytes": None, "pmu_utilization_pct": None,
            "on_chip_per_node": {}, "off_chip_per_node": {}}


def test_write_progress_creates_file_with_expected_schema(tmp_path: Path):
    _write_progress(tmp_path, baseline_cycles=1000, best_cycles=900,
                    turn=2, last_status="NEW_BEST",
                    baseline_mem_info=_mi(400, 0),
                    best_mem_info=_mi(350, 0),
                    last_mem_info=_mi(350, 0))
    data = json.loads((tmp_path / "progress.json").read_text())
    assert data == {
        "baseline_cycles": 1000,
        "best_cycles": 900,
        "turn": 2,
        "last_status": "NEW_BEST",
        "baseline_on_chip_bytes": 400,
        "baseline_off_chip_bytes": 0,
        "best_on_chip_bytes": 350,
        "best_off_chip_bytes": 0,
        "last_on_chip_bytes": 350,
        "last_off_chip_bytes": 0,
    }


def test_write_progress_overwrites(tmp_path: Path):
    _write_progress(tmp_path, baseline_cycles=1000, best_cycles=1000,
                    turn=-1, last_status="BASELINE",
                    baseline_mem_info=_mi(400, 0),
                    best_mem_info=_mi(400, 0),
                    last_mem_info=_mi(400, 0))
    _write_progress(tmp_path, baseline_cycles=1000, best_cycles=950,
                    turn=0, last_status="NEW_BEST",
                    baseline_mem_info=_mi(400, 0),
                    best_mem_info=_mi(380, 0),
                    last_mem_info=_mi(380, 0))
    data = json.loads((tmp_path / "progress.json").read_text())
    assert data["best_cycles"] == 950
    assert data["turn"] == 0
    assert data["last_status"] == "NEW_BEST"
    assert data["best_on_chip_bytes"] == 380
