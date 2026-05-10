from pathlib import Path
from src.planner import _node_turn_dir


def test_default_layout_unchanged(tmp_path):
    d = _node_turn_dir(tmp_path, "root/leaf_1", 0)
    assert d == tmp_path / "root_leaf_1" / "turn_0"
    assert d.is_dir()


def test_pass1_layout(tmp_path):
    d = _node_turn_dir(tmp_path, "root/leaf_1", 0, pass_name="pass1")
    assert d == tmp_path / "root_leaf_1" / "pass1" / "turn_0"
    assert d.is_dir()


def test_pass2_layout(tmp_path):
    # Pass 2 has no turns — pass_name only.
    d = _node_turn_dir(tmp_path, "root/leaf_1", attempt=None, pass_name="pass2")
    assert d == tmp_path / "root_leaf_1" / "pass2"
    assert d.is_dir()


def test_none_root_returns_none():
    d = _node_turn_dir(None, "root/leaf_1", 0)
    assert d is None
