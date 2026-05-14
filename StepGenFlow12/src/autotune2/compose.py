"""Library composition for autotune2.

Bottom-up DP step: after the parent LLM commits a parent DSL plus
``child_picks``, this module enumerates the discrete Cartesian product
of each picked child's Pareto front, scores each composition with the
analytical timing model, and inserts non-dominated entries into the
parent's library cell.

The scoring function is **dependency-injected** via ``score_fn`` so
unit tests can swap in deterministic fake scorers (no STeP / sympy /
torch graph build required). The default analytical scorer is built
by ``make_analytical_scorer`` and pulls in ``translate`` +
``_exec_build_graph`` + ``analyze_timing`` lazily, so importing this
module from a light test environment is cheap.

Scope boundary (deferred to Phase 5 — the search driver):
  - Building a synthetic ``tiled_reference`` wrapper around a non-root
    node so its graph can be analyzed in isolation. Phase 3 takes a
    composed source as-is and scores it.
  - Collecting transitive descendant DSLs from each child's chosen
    library entry. ``compose_source`` here just concatenates whatever
    list of pre-ordered DSL strings the caller hands in.
  - Resolving the kernel-level ``tensors`` dict to per-arg keys
    matching a node's signature.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterator

from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    TensorContract,
    library_cell,
)
from src.autotune2.pareto import insert_pareto


# Global compute-BW budget. Per HANDOFF.md the regime is memory-bound at
# ~100k so per-op BW distribution is vacuous; we just enforce the total
# at the boundary.
LEVEL_TOTAL_BW = 100_000


# A scorer takes a composed-source string and returns (cycles, on_chip_bytes).
ScoreFn = Callable[[str], tuple[int, int]]


# ---------------------------------------------------------------------------
# Source composition (pure string ops)
# ---------------------------------------------------------------------------


def compose_source(
    *,
    parent_dsl: str,
    descendant_dsls_postorder: list[str],
) -> str:
    """Join descendants (post-order, deepest first) followed by the parent.

    Mirrors ``_pass2_compose_namespace`` from the orchestrator: at exec
    time the parent's natural-name call (e.g. ``attention_block(...)``)
    resolves to whichever descendant DSL was exec'd last under that
    name. The caller is responsible for ensuring exactly one
    descendant DSL exists per function name in the parent's call
    graph; this function does not deduplicate.
    """
    return "\n\n".join(descendant_dsls_postorder + [parent_dsl])


# ---------------------------------------------------------------------------
# Cartesian-product compose + Pareto admission
# ---------------------------------------------------------------------------


def cartesian_compose(
    *,
    parent_dsl: str,
    children_fronts: dict[str, list[DesignEntry]],
    children_order: list[str],
    score_fn: ScoreFn,
) -> Iterator[tuple[dict[str, DesignEntry], int, int]]:
    """Yield ``(chosen_per_child, cycles, on_chip)`` for every combination.

    ``children_order`` is a post-order child-path list; the yielded
    ``chosen`` dict and the concatenation order are both driven by it.
    """
    assert set(children_fronts.keys()) == set(children_order), (
        f"cartesian_compose: children_order keys {set(children_order)!r} "
        f"must equal children_fronts keys {set(children_fronts.keys())!r}"
    )
    for path in children_order:
        assert children_fronts[path], (
            f"cartesian_compose: child {path!r} has an empty Pareto front; "
            f"the autotuner must seed every node's library with at least "
            f"the pass-1 baseline before composing parents"
        )

    fronts = [children_fronts[p] for p in children_order]
    for combo in itertools.product(*fronts):
        chosen = dict(zip(children_order, combo))
        descendants_in_order = [chosen[p].dsl for p in children_order]
        composed = compose_source(
            parent_dsl=parent_dsl,
            descendant_dsls_postorder=descendants_in_order,
        )
        cycles, on_chip = score_fn(composed)
        yield chosen, cycles, on_chip


def compose_into_library(
    *,
    parent_lib: NodeLibrary,
    parent_input_contracts: dict[str, TensorContract],
    parent_output_contracts: dict[str, TensorContract],
    parent_dsl: str,
    children_fronts: dict[str, list[DesignEntry]],
    children_order: list[str],
    score_fn: ScoreFn,
    provenance: str,
) -> int:
    """Score every Cartesian combination; admit non-dominated into the cell.

    Returns the count of admitted entries. Provenance per entry encodes
    the child-by-index pick (``"<provenance>;<child_path>@<idx>,...">``)
    so a library dump can be back-traced to the exact composition that
    produced it within a single autotune2 session.
    """
    cell = library_cell(parent_lib, parent_input_contracts, parent_output_contracts)
    admitted = 0
    # Build child-entry -> index lookups once so provenance is O(1) per pick.
    front_index: dict[str, dict[int, int]] = {
        path: {id(entry): idx for idx, entry in enumerate(children_fronts[path])}
        for path in children_order
    }
    for chosen, cycles, on_chip in cartesian_compose(
        parent_dsl=parent_dsl,
        children_fronts=children_fronts,
        children_order=children_order,
        score_fn=score_fn,
    ):
        picks_tag = ",".join(
            f"{p}@{front_index[p][id(chosen[p])]}" for p in children_order
        )
        entry = DesignEntry(
            dsl=parent_dsl,
            input_contracts=dict(parent_input_contracts),
            output_contracts=dict(parent_output_contracts),
            cycles=cycles,
            on_chip=on_chip,
            provenance=f"{provenance};picks={picks_tag}",
        )
        if insert_pareto(cell, entry):
            admitted += 1
    return admitted


# ---------------------------------------------------------------------------
# Analytical scorer factory (production)
# ---------------------------------------------------------------------------


def make_analytical_scorer(
    *,
    dims: dict,
    tensors: dict,
    hw_config: dict,
    max_total_compute_bw: int = LEVEL_TOTAL_BW,
) -> ScoreFn:
    """Build a scorer that runs the STeP analytical timing model.

    Closes over kernel-level ``dims`` / ``tensors`` / ``hw_config`` so
    the returned ``score(composed_source) -> (cycles, on_chip)`` is a
    one-arg callable consumable by ``cartesian_compose``.

    Lazy-imports the STeP machinery (``translate``, ``analyze_timing``,
    ``_exec_build_graph``) so importing this module costs nothing in
    light test environments. The helpers re-implemented here mirror
    ``_normalize_compute_bw`` / ``_compute_memory_totals`` /
    ``_sym_to_int`` from ``src/autotune.py`` rather than importing
    those private symbols, keeping autotune2 self-contained (the
    legacy autotuner is the immutable comparison baseline per
    HANDOFF.md).
    """
    assert max_total_compute_bw >= 1, (
        f"max_total_compute_bw must be >= 1, got {max_total_compute_bw}"
    )

    def score(composed_source: str) -> tuple[int, int]:
        import sympy

        from src.dsl_to_step import translate
        from src.tools import _exec_build_graph
        from timing_and_emulator.timing import analyze_timing

        translated = translate(composed_source)
        graph, _ = _exec_build_graph(translated, dims, tensors)
        _rescale_compute_bw(graph, max_total_compute_bw)
        result = analyze_timing(graph, hw_config=hw_config)
        total_cycles = _sym_to_int(result["total_cycles"])
        on_chip_bytes = _sum_on_chip_bytes(result)
        return total_cycles, on_chip_bytes

    return score


def _sym_to_int(expr) -> int:
    """Collapse a sympy expression to int (free symbols substituted with 1)."""
    import sympy
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        expr = expr.xreplace({s: 1 for s in expr.free_symbols})
    return int(sympy.N(expr))


def _rescale_compute_bw(graph, max_total_compute_bw: int) -> None:
    """Rescale every compute op's ``compute_bw`` so the sum equals the budget.

    Floor at 1 (matches the operator-level minimum in ``ops.py``); the
    sum may exceed the budget slightly when many tiny shares hit the
    floor, acceptable for a budget enforcement pass.
    """
    compute_nodes = [n for n in graph.nodes if hasattr(n, "compute_bw")]
    if not compute_nodes:
        return
    total = sum(n.compute_bw for n in compute_nodes)
    assert total >= 1, (
        "_rescale_compute_bw: sum of compute_bw across compute ops is zero — "
        "invalid graph"
    )
    scale = max_total_compute_bw / total
    for n in compute_nodes:
        n.compute_bw = max(1, int(round(n.compute_bw * scale)))


def _sum_on_chip_bytes(result: dict) -> int:
    """Sum ``on_chip_requirement(count_fifos=False)`` across all nodes."""
    info = result["per_node"]
    sym_subs = result.get("sym_subs", {}) or {}

    def _sub(expr):
        if sym_subs and hasattr(expr, "free_symbols") and expr.free_symbols:
            return expr.xreplace(sym_subs)
        return expr

    total = 0
    for nid, i in info.items():
        total += _sym_to_int(_sub(i["node"].on_chip_requirement(count_fifos=False)))
    return total
