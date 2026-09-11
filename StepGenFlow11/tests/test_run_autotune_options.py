"""Tests for run.py's autotune-options builder."""

import pytest

import run as run_mod


class _FakeArgs:
    def __init__(self, autotune_max_turns=None, autotune_agent="general"):
        self.autotune_max_turns = autotune_max_turns
        self.autotune_agent = autotune_agent


def test_build_options_legacy_no_passes_key_synthesizes_single_pass():
    cfg = {"hw_config": {}, "constraints": {}, "max_turns": 8}
    args = _FakeArgs(autotune_max_turns=None, autotune_agent="parallel")
    opts = run_mod._build_autotune_options(args, cfg)
    assert opts["config"] is cfg
    assert opts["passes"] == [{"agent": "parallel", "max_turns": None}]


def test_build_options_legacy_max_turns_override_threads_into_synthesized_pass():
    cfg = {"hw_config": {}, "constraints": {}, "max_turns": 8}
    args = _FakeArgs(autotune_max_turns=4, autotune_agent="general")
    opts = run_mod._build_autotune_options(args, cfg)
    assert opts["passes"] == [{"agent": "general", "max_turns": 4}]


def test_build_options_passes_key_takes_precedence():
    cfg = {
        "hw_config": {}, "constraints": {},
        "passes": [
            {"agent": "memory", "max_turns": 6,
             "feasibility": {"on_chip_bytes": 262144}},
            {"agent": "general", "max_turns": 12},
        ],
    }
    args = _FakeArgs(autotune_max_turns=99, autotune_agent="parallel")
    opts = run_mod._build_autotune_options(args, cfg)
    # Legacy flags ignored when passes is set.
    assert opts["passes"] == cfg["passes"]


def test_build_options_passes_must_be_nonempty_list():
    cfg = {"hw_config": {}, "constraints": {}, "passes": []}
    args = _FakeArgs()
    with pytest.raises(AssertionError):
        run_mod._build_autotune_options(args, cfg)


def test_build_options_pass_spec_must_have_agent():
    cfg = {"hw_config": {}, "constraints": {},
           "passes": [{"max_turns": 4}]}
    args = _FakeArgs()
    with pytest.raises(AssertionError):
        run_mod._build_autotune_options(args, cfg)
