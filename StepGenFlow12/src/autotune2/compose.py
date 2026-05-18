"""DSL source composition + analytical scorer factory for autotune2.

Two pieces:
  - ``compose_source`` — flat string-join of descendant DSLs followed by
    the parent DSL. Lets ``exec`` resolve the parent's natural-name
    child calls (e.g. ``attention_block(...)``) against the last-bound
    descendant DSL of the same name. Caller is responsible for passing
    exactly one descendant DSL per natural name in the parent's call
    graph.
  - ``make_analytical_scorer`` — closure over kernel-level ``dims`` /
    ``tensors`` / ``hw_config`` returning ``score(composed_source) ->
    (cycles, on_chip)``. Lazy-imports the STeP machinery so importing
    this module from a light test environment is cheap.

The Cartesian-product compose path that previously lived here is gone:
``search_parent`` resolves each child pick to a single ``DesignEntry``
(per-entry variant indices), so there's exactly one composed source per
parent turn — no sweep.
"""

from __future__ import annotations

from typing import Callable


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
    one-arg callable.

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

    # Per-source memoization shared between ``score`` and ``breakdown`` so the
    # OVER_BUDGET / baseline paths can fetch the per-node report without
    # re-running ``analyze_timing`` on the same composed source.
    cache: dict[str, tuple[int, int, str]] = {}

    def _evaluate(composed_source: str) -> tuple[int, int, str]:
        if composed_source in cache:
            return cache[composed_source]
        from src.dsl_to_step import translate
        from src.tools import _exec_build_graph
        from timing_and_emulator.timing import analyze_timing

        translated = translate(composed_source)
        graph, _ = _exec_build_graph(translated, dims, tensors)
        _rescale_compute_bw(graph, max_total_compute_bw)
        result = analyze_timing(graph, hw_config=hw_config)
        total_cycles = _sym_to_int(result["total_cycles"])
        per_node = _per_node_on_chip_bytes(result)
        on_chip_bytes = sum(b for _, _, b in per_node)
        report = _format_per_node_memory_report(per_node, on_chip_bytes)
        out = (total_cycles, on_chip_bytes, report)
        cache[composed_source] = out
        return out

    def score(composed_source: str) -> tuple[int, int]:
        cycles, on_chip, _ = _evaluate(composed_source)
        return cycles, on_chip

    def breakdown(composed_source: str) -> str:
        _, _, report = _evaluate(composed_source)
        return report

    # Side-channel attribute: callers that want the per-node memory report
    # opt in by reading ``score_fn.breakdown``. Keeping the ``ScoreFn`` return
    # type as ``(cycles, on_chip)`` avoids touching every test scorer stub.
    score.breakdown = breakdown
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


def _per_node_on_chip_bytes(result: dict) -> list[tuple[int, str, int]]:
    """``[(instance_id, op_label, on_chip_bytes), ...]`` for every node.

    Computed once per ``analyze_timing`` run and shared between the
    ``(cycles, on_chip)`` summary and the per-node memory report so we
    never walk the per-node info twice for the same composed source.
    """
    info = result["per_node"]
    sym_subs = result.get("sym_subs", {}) or {}

    def _sub(expr):
        if sym_subs and hasattr(expr, "free_symbols") and expr.free_symbols:
            return expr.xreplace(sym_subs)
        return expr

    out: list[tuple[int, str, int]] = []
    for nid, i in info.items():
        n = i["node"]
        b = _sym_to_int(_sub(n.on_chip_requirement(count_fifos=False)))
        out.append((nid, _node_label(n), b))
    return out


def _node_label(n) -> str:
    """``"[id] OpType"`` or ``"[id] OpType<FnName>"`` for compute ops with ``fn``.

    Mirrors ``src/autotune.py:_node_label`` so the per-node lines in the
    prompt feedback match the labels in the verbose timing report a user
    might inspect side-by-side.
    """
    label = f"[{n.instance_id}] {n.__class__.__name__}"
    fn = getattr(n, "fn", None)
    if fn is not None:
        label += f"<{fn.__class__.__name__}>"
    return label


def _format_per_node_memory_report(
    per_node: list[tuple[int, str, int]], total: int,
) -> str:
    """Render the per-node on-chip-memory contributors as a markdown stanza.

    Sorted descending by bytes; every node with a non-zero footprint
    appears on its own line. Earlier versions capped at top-K with a
    ``... N more nodes`` rollup, which hid the long tail right when the
    LLM most needs it (e.g. MoE graphs where the per-expert loads are
    individually mid-sized but dominate the total in aggregate).
    Returns the empty string when total is zero — no on-chip footprint
    means nothing useful to surface.
    """
    if total <= 0:
        return ""
    nonzero = [t for t in per_node if t[2] > 0]
    nonzero.sort(key=lambda t: t[2], reverse=True)

    lines = [
        f"Per-node on-chip memory breakdown (total {total} B across "
        f"{len(nonzero)} contributing nodes):",
    ]
    for _nid, label, b in nonzero:
        pct = 100.0 * b / total
        lines.append(f"  {label:<44s} {b:>12d} B  ({pct:5.1f}%)")
    return "\n".join(lines)
