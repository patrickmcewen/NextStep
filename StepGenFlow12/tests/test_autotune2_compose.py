"""Unit tests for autotune2 source composition + scorer factory.

The analytical scorer's body is NOT exercised here — it requires the
STeP graph machinery (``translate``, ``_exec_build_graph``,
``analyze_timing``). We only validate that the factory builds a
callable and rejects invalid bandwidth budgets.
"""

import pytest

from src.autotune2.compose import (
    _format_per_node_memory_report,
    compose_source,
    make_analytical_scorer,
)


# --- compose_source -----------------------------------------------------------


def test_compose_source_post_order_concat():
    out = compose_source(
        parent_dsl="def parent():\n    return child_a() + child_b()\n",
        descendant_dsls_postorder=[
            "def child_a():\n    return 1\n",
            "def child_b():\n    return 2\n",
        ],
    )
    assert "def child_a" in out
    assert "def child_b" in out
    # parent must come last so child defs are in scope when parent execs
    assert out.rindex("def parent") > out.rindex("def child_b")


def test_compose_source_empty_descendants():
    parent = "def root():\n    pass\n"
    assert compose_source(parent_dsl=parent, descendant_dsls_postorder=[]) == parent


# --- per-node memory report ---------------------------------------------------


def test_per_node_memory_report_caps_large_graphs_at_top_100():
    per_node = [
        (i, f"[{i}] Op{i}", 200 - i)
        for i in range(101)
    ]

    out = _format_per_node_memory_report(per_node, sum(b for _, _, b in per_node))

    assert "across 101 contributing nodes" in out
    assert out.count("Op") == 100
    assert "[99] Op99" in out
    assert "[100] Op100" not in out
    assert "... 1 more contributing nodes omitted" in out


# --- make_analytical_scorer (sanity: factory + lazy import) -------------------


def test_make_analytical_scorer_returns_callable():
    """Smoke test: factory builds a callable without triggering heavy imports.

    The scorer body imports STeP machinery lazily on first call; just
    constructing it should succeed even in environments that lack the
    full STeP install.
    """
    score = make_analytical_scorer(
        dims={"M": 64, "N": 64},
        tensors={},
        hw_config={},
    )
    assert callable(score)


def test_make_analytical_scorer_rejects_zero_budget():
    with pytest.raises(AssertionError, match="max_total_compute_bw"):
        make_analytical_scorer(
            dims={}, tensors={}, hw_config={},
            max_total_compute_bw=0,
        )
