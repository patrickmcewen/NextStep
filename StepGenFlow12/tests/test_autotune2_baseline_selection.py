"""Unit tests for autotune2.baseline_selection."""

import pytest

from src.autotune2.baseline_selection import (
    STRATEGIES,
    _farthest_point_sample,
    _flatten,
    _pareto_front,
    select_baselines,
)
from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    TensorContract,
    library_cell,
)


def _entry(cycles: int, on_chip: int, *, dsl: str = "x", provenance: str = "") -> DesignEntry:
    return DesignEntry(
        dsl=dsl, cycles=cycles, on_chip=on_chip, provenance=provenance
    )


def _make_lib(entries: list[DesignEntry]) -> NodeLibrary:
    """Pack ``entries`` into one (identity-contract) cell of a fresh library."""
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    cell.extend(entries)
    return lib


def _make_multi_cell_lib(
    cell_a: list[DesignEntry], cell_b: list[DesignEntry]
) -> NodeLibrary:
    """Two cells under distinct input_contract keys, same empty output key."""
    lib: NodeLibrary = {}
    library_cell(lib, {}, {}).extend(cell_a)
    library_cell(
        lib,
        {"x": TensorContract(reshape=(4, 4), permutation=(1, 0))},
        {},
    ).extend(cell_b)
    return lib


# --- _flatten -----------------------------------------------------------------


def test_flatten_single_cell():
    e0 = _entry(100, 1000)
    e1 = _entry(200, 500)
    lib = _make_lib([e0, e1])
    assert _flatten(lib) == [e0, e1]


def test_flatten_multi_cell_unions():
    a0 = _entry(100, 1000)
    b0 = _entry(50, 2000)
    lib = _make_multi_cell_lib([a0], [b0])
    flat = _flatten(lib)
    assert set(map(id, flat)) == {id(a0), id(b0)}


def test_flatten_empty_lib():
    assert _flatten({}) == []


# --- _pareto_front ------------------------------------------------------------


def test_pareto_front_keeps_non_dominated():
    # Three points on the front, one strictly dominated.
    a = _entry(100, 1000)      # cheap on cycles
    b = _entry(50, 2000)       # cheaper on cycles, bigger memory
    c = _entry(200, 500)       # bigger cycles, cheapest memory
    d = _entry(300, 3000)      # strictly worse than a (both axes)
    front = _pareto_front([a, b, c, d])
    assert set(map(id, front)) == {id(a), id(b), id(c)}


def test_pareto_front_keeps_ties():
    # Two entries with identical (cycles, on_chip) tie — neither dominates.
    a = _entry(100, 500)
    b = _entry(100, 500)
    front = _pareto_front([a, b])
    assert set(map(id, front)) == {id(a), id(b)}


def test_pareto_front_singleton():
    a = _entry(100, 500)
    assert _pareto_front([a]) == [a]


# --- _farthest_point_sample --------------------------------------------------


def test_farthest_point_sample_seeds_min_cycles():
    a = _entry(100, 1000)
    b = _entry(50, 2000)
    c = _entry(200, 500)
    picks = _farthest_point_sample([a, b, c], k=1)
    assert picks == [b]  # b has lowest cycles


def test_farthest_point_sample_spreads_picks():
    # Five points along the front: two clusters of two close points + one far.
    pts = [
        _entry(100, 1000),
        _entry(101, 999),    # near-cluster 1
        _entry(500, 500),
        _entry(501, 499),    # near-cluster 2
        _entry(300, 800),    # middle
    ]
    picks = _farthest_point_sample(pts, k=3)
    # Should pick one from each region (low cycles, high cycles, middle).
    cycles_picked = sorted(e.cycles for e in picks)
    assert cycles_picked[0] == 100  # seed = min cycles
    # Second pick must be from the other extreme (max distance) — the 500/501 cluster.
    assert cycles_picked[-1] in (500, 501)
    # Third pick from the middle.
    assert 200 < cycles_picked[1] < 500


def test_farthest_point_sample_degenerate_zero_span():
    # All entries share one axis value — span clamp keeps math safe.
    pts = [_entry(100, 500), _entry(100, 800), _entry(100, 1200)]
    picks = _farthest_point_sample(pts, k=2)
    assert len(picks) == 2
    assert {id(e) for e in picks}.issubset({id(p) for p in pts})


# --- select_baselines: strategy dispatch -------------------------------------


def test_select_all_returns_every_entry():
    e0 = _entry(100, 1000)
    e1 = _entry(200, 500)
    e2 = _entry(150, 700)
    lib = _make_lib([e0, e1, e2])
    picks = select_baselines(lib, k=1, strategy="all")
    assert set(map(id, picks)) == {id(e0), id(e1), id(e2)}


def test_select_min_cycles_sorts_and_caps():
    e0 = _entry(100, 1000)
    e1 = _entry(50, 2000)
    e2 = _entry(200, 500)
    lib = _make_lib([e0, e1, e2])
    picks = select_baselines(lib, k=2, strategy="min_cycles")
    assert [e.cycles for e in picks] == [50, 100]


def test_select_min_on_chip_sorts_and_caps():
    e0 = _entry(100, 1000)
    e1 = _entry(50, 2000)
    e2 = _entry(200, 500)
    lib = _make_lib([e0, e1, e2])
    picks = select_baselines(lib, k=2, strategy="min_on_chip")
    assert [e.on_chip for e in picks] == [500, 1000]


def test_select_pareto_diverse_returns_front_when_small():
    a = _entry(100, 1000)
    b = _entry(50, 2000)
    c = _entry(200, 500)
    lib = _make_lib([a, b, c])
    picks = select_baselines(lib, k=10, strategy="pareto_diverse")
    assert set(map(id, picks)) == {id(a), id(b), id(c)}


def test_select_pareto_diverse_prunes_dominated_then_caps():
    a = _entry(100, 1000)
    b = _entry(50, 2000)
    c = _entry(200, 500)
    dominated = _entry(300, 3000)
    lib = _make_lib([a, b, c, dominated])
    picks = select_baselines(lib, k=2, strategy="pareto_diverse")
    assert id(dominated) not in {id(e) for e in picks}
    assert len(picks) == 2


def test_select_excludes_by_identity():
    e0 = _entry(100, 1000)
    e1 = _entry(50, 2000)
    e2 = _entry(200, 500)
    lib = _make_lib([e0, e1, e2])
    picks = select_baselines(lib, k=5, strategy="min_cycles", exclude=[e1])
    assert id(e1) not in {id(e) for e in picks}
    assert [e.cycles for e in picks] == [100, 200]


def test_select_excludes_only_by_identity_not_value():
    # Two entries with identical (cycles, on_chip) — excluding one shouldn't drop both.
    e0 = _entry(100, 1000)
    e1 = _entry(100, 1000)  # same values, different object
    lib = _make_lib([e0, e1])
    picks = select_baselines(lib, k=5, strategy="all", exclude=[e0])
    assert {id(e) for e in picks} == {id(e1)}


def test_select_k_zero_returns_empty():
    e0 = _entry(100, 1000)
    lib = _make_lib([e0])
    assert select_baselines(lib, k=0, strategy="min_cycles") == []
    # "all" with k=0 still returns everything (k is ignored).
    assert select_baselines(lib, k=0, strategy="all") == [e0]


def test_select_empty_lib_returns_empty():
    assert select_baselines({}, k=5, strategy="pareto_diverse") == []
    assert select_baselines({}, k=5, strategy="min_cycles") == []
    assert select_baselines({}, k=5, strategy="all") == []


def test_select_multi_cell_lib_flattens():
    a0 = _entry(100, 1000)
    b0 = _entry(50, 2000)
    lib = _make_multi_cell_lib([a0], [b0])
    picks = select_baselines(lib, k=10, strategy="min_cycles")
    assert {id(e) for e in picks} == {id(a0), id(b0)}


def test_select_rejects_unknown_strategy():
    with pytest.raises(AssertionError, match="strategy 'bogus' not in"):
        select_baselines({}, k=1, strategy="bogus")


def test_select_rejects_negative_k():
    with pytest.raises(AssertionError, match="k must be >= 0"):
        select_baselines({}, k=-1, strategy="min_cycles")


# --- STRATEGIES advertised list matches dispatch -----------------------------


def test_strategies_constant_matches_dispatch_table():
    # Every advertised strategy must be accepted; nothing else.
    e = _entry(100, 1000)
    lib = _make_lib([e])
    for s in STRATEGIES:
        picks = select_baselines(lib, k=1, strategy=s)
        assert isinstance(picks, list)
