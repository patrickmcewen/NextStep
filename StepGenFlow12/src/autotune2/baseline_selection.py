"""Per-pass baseline selection.

Multi-pass autotuning lets each pass branch off entries from the prior
pass's library. The number of branches scales the LLM-call cost linearly,
so each pass picks at most ``k`` entries per node via a configurable
strategy. The caller always re-seeds the pass-1 baseline separately, so
this helper is the rest of the branch list.

Strategies:

  - ``"all"``: every entry in the library, ignoring ``k``. Use for tiny
    libraries where exploring everything is cheap.

  - ``"min_cycles"``: top-``k`` by lowest ``cycles``. Favors variants
    near the cycle-optimal end of the Pareto front.

  - ``"min_on_chip"``: top-``k`` by lowest ``on_chip``. Favors the
    memory-optimal end.

  - ``"pareto_diverse"``: prune to Pareto-non-dominated entries, then
    spread ``k`` picks across the front via greedy farthest-point
    sampling in normalized ``(cycles, on_chip)`` space. Default.

All strategies return at most ``k`` entries (except ``"all"``). The
order of the returned list is meaningful for ``"min_*"`` (ascending by
the relevant axis) and arbitrary for ``"all"`` / ``"pareto_diverse"``.
"""

from __future__ import annotations

from typing import Iterable

from src.autotune2.contracts import DesignEntry, NodeLibrary


STRATEGIES = ("all", "min_cycles", "min_on_chip", "pareto_diverse")


def _flatten(lib: NodeLibrary) -> list[DesignEntry]:
    """Every DesignEntry across every (in, out) cell of ``lib``."""
    return [
        entry
        for in_cell in lib.values()
        for out_cell in in_cell.values()
        for entry in out_cell
    ]


def _pareto_front(entries: list[DesignEntry]) -> list[DesignEntry]:
    """Return the subset of ``entries`` not strictly dominated on
    ``(cycles, on_chip)``. A dominates B iff A.cycles <= B.cycles AND
    A.on_chip <= B.on_chip AND at least one inequality is strict.

    Entries with identical (cycles, on_chip) are both kept (neither
    strictly dominates the other); the caller deduplicates by identity
    if needed.
    """
    front: list[DesignEntry] = []
    for e in entries:
        dominated = False
        for f in entries:
            if f is e:
                continue
            if (
                f.cycles <= e.cycles
                and f.on_chip <= e.on_chip
                and (f.cycles < e.cycles or f.on_chip < e.on_chip)
            ):
                dominated = True
                break
        if not dominated:
            front.append(e)
    return front


def _farthest_point_sample(
    front: list[DesignEntry], k: int
) -> list[DesignEntry]:
    """Pick ``k`` entries from ``front`` to maximize spread in normalized
    ``(cycles, on_chip)`` space. Assumes ``len(front) > k > 0``.

    Seed with the lowest-cycles entry, then iteratively pick the entry
    maximizing min L1-distance to the already-picked set.
    """
    cycles_vals = [e.cycles for e in front]
    on_chip_vals = [e.on_chip for e in front]
    c_min, c_max = min(cycles_vals), max(cycles_vals)
    m_min, m_max = min(on_chip_vals), max(on_chip_vals)
    c_span = max(c_max - c_min, 1)
    m_span = max(m_max - m_min, 1)
    norm = [
        ((e.cycles - c_min) / c_span, (e.on_chip - m_min) / m_span)
        for e in front
    ]

    picked_idx = [min(range(len(front)), key=lambda i: front[i].cycles)]
    while len(picked_idx) < k:
        best_i, best_dist = -1, -1.0
        for i in range(len(front)):
            if i in picked_idx:
                continue
            d = min(
                abs(norm[i][0] - norm[j][0]) + abs(norm[i][1] - norm[j][1])
                for j in picked_idx
            )
            if d > best_dist:
                best_dist, best_i = d, i
        assert best_i >= 0, "farthest-point loop failed to find an unpicked entry"
        picked_idx.append(best_i)
    return [front[i] for i in picked_idx]


def select_baselines(
    lib: NodeLibrary,
    *,
    k: int,
    strategy: str = "pareto_diverse",
    exclude: Iterable[DesignEntry] = (),
) -> list[DesignEntry]:
    """Pick up to ``k`` baseline entries from a node's library.

    ``exclude`` is compared by object identity (``is``), letting callers
    skip e.g. the pass-1 baseline that they re-seed explicitly without
    requiring ``DesignEntry`` to be hashable.
    """
    assert strategy in STRATEGIES, (
        f"select_baselines: strategy {strategy!r} not in {STRATEGIES}"
    )
    assert k >= 0, f"select_baselines: k must be >= 0, got {k}"
    exclude_ids = {id(e) for e in exclude}
    pool = [e for e in _flatten(lib) if id(e) not in exclude_ids]

    if not pool or k == 0:
        return [] if strategy != "all" else pool

    if strategy == "all":
        return pool
    if strategy == "min_cycles":
        return sorted(pool, key=lambda e: e.cycles)[:k]
    if strategy == "min_on_chip":
        return sorted(pool, key=lambda e: e.on_chip)[:k]
    # pareto_diverse
    front = _pareto_front(pool)
    if len(front) <= k:
        return front
    return _farthest_point_sample(front, k)
