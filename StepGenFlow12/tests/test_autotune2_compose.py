"""Unit tests for autotune2 library composition.

The analytical scorer is NOT exercised here — it requires the STeP
graph machinery (``translate``, ``_exec_build_graph``,
``analyze_timing``). All tests inject a deterministic fake scorer so
the Cartesian product + Pareto admission logic can be validated in
isolation.
"""

import pytest

from src.autotune2.compose import (
    cartesian_compose,
    compose_into_library,
    compose_source,
    make_analytical_scorer,
)
from src.autotune2.contracts import (
    DesignEntry,
    TensorContract,
    library_cell,
    vanilla_contract_for,
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


# --- cartesian_compose --------------------------------------------------------


def _entry(dsl: str, cycles: int = 0, on_chip: int = 0) -> DesignEntry:
    return DesignEntry(
        dsl=dsl,
        input_contracts={},
        output_contracts={},
        cycles=cycles,
        on_chip=on_chip,
    )


def test_cartesian_compose_enumerates_full_product():
    fronts = {
        "a": [_entry("# a0\n"), _entry("# a1\n")],
        "b": [_entry("# b0\n"), _entry("# b1\n"), _entry("# b2\n")],
    }
    seen = set()

    def fake_score(src):
        # Score is deterministic: encode the picks by counting marker lines.
        n_a = sum(1 for ln in src.splitlines() if ln.startswith("# a"))
        n_b = sum(1 for ln in src.splitlines() if ln.startswith("# b"))
        assert n_a == 1 and n_b == 1, f"composition has wrong shape: {src!r}"
        return 0, 0

    for chosen, _, _ in cartesian_compose(
        parent_dsl="# parent\n",
        children_fronts=fronts,
        children_order=["a", "b"],
        score_fn=fake_score,
    ):
        # record the exact (a_idx, b_idx) pair
        a_idx = next(i for i, e in enumerate(fronts["a"]) if e is chosen["a"])
        b_idx = next(i for i, e in enumerate(fronts["b"]) if e is chosen["b"])
        seen.add((a_idx, b_idx))
    assert seen == {(i, j) for i in range(2) for j in range(3)}


def test_cartesian_compose_respects_children_order():
    # parent must come last; children in given order. Verify by string position.
    fronts = {
        "x": [_entry("# X-body\n")],
        "y": [_entry("# Y-body\n")],
    }
    captured = []

    def fake_score(src):
        captured.append(src)
        return 0, 0

    list(cartesian_compose(
        parent_dsl="# PARENT\n",
        children_fronts=fronts,
        children_order=["x", "y"],   # x before y
        score_fn=fake_score,
    ))
    assert len(captured) == 1
    src = captured[0]
    assert src.index("# X-body") < src.index("# Y-body") < src.index("# PARENT")

    # Reverse order
    captured.clear()
    list(cartesian_compose(
        parent_dsl="# PARENT\n",
        children_fronts=fronts,
        children_order=["y", "x"],
        score_fn=fake_score,
    ))
    src = captured[0]
    assert src.index("# Y-body") < src.index("# X-body") < src.index("# PARENT")


def test_cartesian_compose_rejects_empty_front():
    fronts = {"a": [], "b": [_entry("# b\n")]}
    with pytest.raises(AssertionError, match="empty Pareto front"):
        list(cartesian_compose(
            parent_dsl="# p\n",
            children_fronts=fronts,
            children_order=["a", "b"],
            score_fn=lambda src: (0, 0),
        ))


def test_cartesian_compose_rejects_key_mismatch():
    fronts = {"a": [_entry("# a\n")]}
    with pytest.raises(AssertionError, match="must equal"):
        list(cartesian_compose(
            parent_dsl="# p\n",
            children_fronts=fronts,
            children_order=["a", "b"],   # b not in fronts
            score_fn=lambda src: (0, 0),
        ))


# --- compose_into_library -----------------------------------------------------


def test_compose_into_library_admits_only_non_dominated():
    """Two children each with 3 Pareto entries; verify only non-dominated
    compositions land in the parent's library cell."""
    fronts = {
        "a": [
            _entry("# a0\n", cycles=10, on_chip=100),
            _entry("# a1\n", cycles=20, on_chip=50),
            _entry("# a2\n", cycles=30, on_chip=20),
        ],
        "b": [
            _entry("# b0\n", cycles=5, on_chip=100),
            _entry("# b1\n", cycles=15, on_chip=30),
        ],
    }

    # Score the composition as the (cycles, on_chip) sum of the two picks.
    def sum_score(src):
        ac = 0
        ach = 0
        for line in src.splitlines():
            for path, front in fronts.items():
                for entry in front:
                    if entry.dsl.strip() == line.strip():
                        ac += entry.cycles
                        ach += entry.on_chip
        return ac, ach

    lib: dict = {}
    in_c = {"x": vanilla_contract_for((4, 8))}
    out_c = {"out_0": vanilla_contract_for((4, 8))}
    admitted = compose_into_library(
        parent_lib=lib,
        parent_input_contracts=in_c,
        parent_output_contracts=out_c,
        parent_dsl="# parent\n",
        children_fronts=fronts,
        children_order=["a", "b"],
        score_fn=sum_score,
        provenance="test",
    )

    cell = library_cell(lib, in_c, out_c)
    # 6 total combinations; non-dominated count must equal admitted count.
    assert admitted == len(cell)
    # No entry in the cell dominates any other (Pareto invariant).
    for i, e1 in enumerate(cell):
        for j, e2 in enumerate(cell):
            if i == j:
                continue
            assert not (e1.cycles <= e2.cycles and e1.on_chip <= e2.on_chip
                        and (e1.cycles < e2.cycles or e1.on_chip < e2.on_chip)), (
                f"entry[{i}]={e1} dominates entry[{j}]={e2} — Pareto violated"
            )


def test_compose_into_library_provenance_encodes_picks():
    fronts = {
        "alpha": [_entry("# a0\n", 10, 10), _entry("# a1\n", 5, 30)],
        "beta": [_entry("# b0\n", 1, 1)],
    }
    lib: dict = {}
    in_c: dict = {}
    out_c: dict = {}
    compose_into_library(
        parent_lib=lib,
        parent_input_contracts=in_c,
        parent_output_contracts=out_c,
        parent_dsl="# parent\n",
        children_fronts=fronts,
        children_order=["alpha", "beta"],
        score_fn=lambda src: (1, 1),  # all identical -> first admitted wins
        provenance="phase3test",
    )
    cell = library_cell(lib, in_c, out_c)
    assert len(cell) >= 1
    for entry in cell:
        assert entry.provenance.startswith("phase3test;picks=")
        assert "alpha@" in entry.provenance
        assert "beta@" in entry.provenance


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
        hw_config={"pmu_buffer_bytes": 1 << 24},
    )
    assert callable(score)


def test_make_analytical_scorer_rejects_zero_budget():
    with pytest.raises(AssertionError, match="max_total_compute_bw"):
        make_analytical_scorer(
            dims={}, tensors={}, hw_config={},
            max_total_compute_bw=0,
        )
