"""Unit tests for autotune2 Phase 6 runtime: top-K, rust promotion, summary."""

import asyncio
import json
from pathlib import Path

import pytest

from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    TensorContract,
    library_cell,
    vanilla_contract_for,
)
from src.autotune2.runtime import (
    RustPromotionResult,
    _contract_conformance_smoke_test,
    _extract_node_def_block,
    build_real_verifier_factory_fn,
    pick_top_k_pareto_entries,
    promote_top_k,
    write_autotune2_summary,
)
from src.autotune2.search import AutotuneResult


# --- pick_top_k_pareto_entries -----------------------------------------------


def _root_lib_with_entries(entries: list[tuple[int, int, str]]) -> NodeLibrary:
    """Build a root library with the given (cycles, on_chip, prov) entries
    all in a single cell."""
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    for cycles, on_chip, prov in entries:
        cell.append(DesignEntry(
            dsl=f"# dsl prov={prov}\n",
            cycles=cycles, on_chip=on_chip, provenance=prov,
        ))
    return lib


def test_pick_top_k_drops_dominated_and_limits_size():
    lib = _root_lib_with_entries([
        (100, 100, "A"),   # dominated by B
        (50, 50, "B"),     # Pareto
        (80, 30, "C"),     # Pareto
        (200, 10, "D"),    # Pareto
        (150, 150, "E"),   # dominated by B
    ])
    picks = pick_top_k_pareto_entries(lib, k=3)
    provs = [e.provenance for e in picks]
    # Non-dominated: B, C, D (3 total)
    assert set(provs) == {"B", "C", "D"}
    # Sorted by (cycles, on_chip)
    assert provs == ["B", "C", "D"]


def test_pick_top_k_truncates_when_more_than_k_non_dominated():
    lib = _root_lib_with_entries([
        (10, 100, "alpha"),
        (20, 80, "beta"),
        (30, 60, "gamma"),
        (40, 40, "delta"),
    ])
    picks = pick_top_k_pareto_entries(lib, k=2)
    provs = [e.provenance for e in picks]
    # All 4 are non-dominated; we keep the lowest-cycle 2
    assert provs == ["alpha", "beta"]


def test_pick_top_k_returns_fewer_when_pareto_smaller_than_k():
    lib = _root_lib_with_entries([(50, 50, "X")])
    picks = pick_top_k_pareto_entries(lib, k=5)
    assert len(picks) == 1


def test_pick_top_k_empty_library_asserts():
    with pytest.raises(AssertionError, match="has no entries"):
        pick_top_k_pareto_entries({}, k=1)


def test_pick_top_k_rejects_k_zero():
    with pytest.raises(AssertionError, match=">= 1"):
        pick_top_k_pareto_entries({}, k=0)


# --- promote_top_k -----------------------------------------------------------


def test_promote_top_k_runs_rust_per_pick_and_sorts_by_rust_cycles():
    lib = _root_lib_with_entries([
        (50, 100, "A"),
        (80, 30, "B"),
    ])
    # The two are non-dominated. Mock rust returns inverted ordering: B beats A.
    rust_calls = []
    def rust(src):
        rust_calls.append(src)
        # First call (A) returns 999, second (B) returns 100.
        return (999 if "A" in src else 100, 12.5)

    results = promote_top_k(root_library=lib, k=2, rust_evaluate_fn=rust)
    assert len(results) == 2
    assert results[0].entry.provenance == "B"
    assert results[0].rust_cycles == 100
    assert results[1].entry.provenance == "A"
    assert results[1].rust_cycles == 999
    assert rust_calls, "rust evaluator must be called per pick"


def test_promote_top_k_includes_descendants_in_composed_source():
    """Root entry with a children_picks reference should produce a composed
    source that includes the descendant DSL before the root DSL."""
    leaf_entry = DesignEntry(
        dsl="def leaf_fn():\n    return None\n",
        cycles=10, on_chip=10, provenance="leaf_baseline",
    )
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    cell.append(DesignEntry(
        dsl=("def tiled_reference(dims, tensors):\n"
             "    return offchip_store(leaf_fn())\n"),
        cycles=20, on_chip=20, provenance="root_baseline",
        children_picks={"root/leaf": leaf_entry},
    ))

    captured = []
    def rust(src):
        captured.append(src)
        return (1, 1.0)

    promote_top_k(root_library=lib, k=1, rust_evaluate_fn=rust)
    assert len(captured) == 1
    src = captured[0]
    # leaf DSL appears before root's tiled_reference
    assert src.index("def leaf_fn") < src.index("def tiled_reference")


# --- write_autotune2_summary -------------------------------------------------


def _trivial_autotune_result() -> AutotuneResult:
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    cell.append(DesignEntry(
        dsl="# root\n", cycles=50, on_chip=100, provenance="pass1_baseline",
    ))
    cell.append(DesignEntry(
        dsl="# root_v1\n", cycles=40, on_chip=120, provenance="llm_turn_0",
    ))
    return AutotuneResult(libraries={"root": lib}, root_path="root")


def test_write_summary_includes_pareto_and_winner(tmp_path):
    res = _trivial_autotune_result()
    promotions = [
        RustPromotionResult(
            entry=next(iter(next(iter(res.libraries["root"].values())).values()))[0],
            rust_cycles=55, rust_dur_ms=20.0, composed_source="<<src>>",
        ),
    ]
    out = tmp_path / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=promotions, out_path=out,
    )
    payload = json.loads(out.read_text())
    assert payload["root_path"] == "root"
    assert payload["library_sizes"]["root"] == 1
    # root_pareto sorted by cycles
    assert payload["root_pareto"][0]["cycles"] == 40
    assert payload["root_pareto"][1]["cycles"] == 50
    assert payload["best_rust_entry"]["rust_cycles"] == 55
    # sources NOT included by default
    assert "best_composed_source" not in payload


def test_write_summary_include_sources_writes_source(tmp_path):
    res = _trivial_autotune_result()
    promotions = [
        RustPromotionResult(
            entry=next(iter(next(iter(res.libraries["root"].values())).values()))[0],
            rust_cycles=1, rust_dur_ms=0.0, composed_source="MARKER_SOURCE",
        ),
    ]
    out = tmp_path / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=promotions,
        out_path=out, include_sources=True,
    )
    payload = json.loads(out.read_text())
    assert payload["best_composed_source"] == "MARKER_SOURCE"


def test_write_summary_no_promotions_still_valid(tmp_path):
    res = _trivial_autotune_result()
    out = tmp_path / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=[], out_path=out,
    )
    payload = json.loads(out.read_text())
    assert payload["best_rust_entry"] is None
    assert payload["rust_winners"] == []


# --- _extract_node_def_block --------------------------------------------------


def test_extract_node_def_block_picks_named_top_level_def():
    src = (
        "def tiled_reference(dims, tensors):\n"
        "    _x_in = offchip_load(tensors['x'], stride=(1,),\n"
        "                          out_shape_tiled=(2,), tile_row=1, tile_col=8)\n"
        "    return offchip_store(my_leaf(_x_in, out_shapes=((1, 2, 8),)))\n"
        "\n"
        "def my_leaf(x, *, out_shapes):\n"
        "    return unary_add_imm(x, 1.0)\n"
    )
    block = _extract_node_def_block(src, "my_leaf")
    # Only the leaf def comes back; the wrapper's offchip_store doesn't.
    assert block.lstrip().startswith("def my_leaf(")
    assert "offchip_store" not in block
    assert "unary_add_imm" in block


def test_extract_node_def_block_raises_when_missing():
    src = "def other_fn():\n    pass\n"
    with pytest.raises(AssertionError, match="def my_leaf"):
        _extract_node_def_block(src, "my_leaf")


# --- build_real_verifier_factory_fn dispatch ---------------------------------


def test_verifier_factory_root_uses_compute_gold_kernel_name():
    """At root (parent_contract is None) the factory should produce a
    verifier whose gate_correctness uses ``root_kernel`` directly,
    *not* a synthetic kernel name."""
    from src import gold_cache
    import torch

    # Pre-stash a gold tensor under the root kernel name so the verifier
    # picks it up via _get_gold. Use a name that won't collide with a
    # real kernel — _get_gold will read the cache hit and skip
    # run_reference.
    gold_cache._inject_gold("__test_root_kernel__", {"d": 1}, torch.zeros(3))

    make_verifier = build_real_verifier_factory_fn(
        root_kernel="__test_root_kernel__",
        dims={"d": 1},
        check_order="correctness-first",
    )

    class _StubNode:
        path = "root"
        name = "ignored"

    # Build a root verifier (parent_contract=None) — the closure should
    # resolve to "__test_root_kernel__" as kernel_name. Smoke-test: we
    # invoke it on trivially-wrong code so gate_correctness fires and
    # returns a feedback string; we only check the kernel-name routing
    # is wired (verify the verifier runs without KeyError or
    # synth-kernel-name miscaching).
    verifier = make_verifier(_StubNode(), None, {})
    # Trivially-wrong DSL that defines tiled_reference but returns None.
    src = "def tiled_reference(dims, tensors):\n    return None\n"
    result = asyncio.run(verifier(src))
    # Either passed (matches gold by chance) or failed (with feedback) —
    # what matters is no exception bubbled up from a missing kernel.
    assert isinstance(result.passed, bool)


def test_verifier_factory_non_root_passes_on_clean_translate_and_build(
    tmp_path, monkeypatch,
):
    """A non-root variant whose composed source translates + builds
    without error should pass — the verifier no longer checks gold,
    only that the subgraph is structurally well-formed."""
    import torch
    from src.contract import Contract
    from src.node_signature import TensorArg

    c = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 4),),
        tiled_shapes=((1, 4, 4),),
        tiled_values=(torch.zeros(1, 4, 4),),
        out_shapes=((1, 4, 4),),
        tiled_outputs=(torch.zeros(1, 4, 4),),
        out_is_tuple=False,
        arg_specs=(TensorArg(shape=(4, 4)),),
        arg_is_raw=(False,),
    )

    # Stub translate + _exec_build_graph at the module-import level so
    # we don't need a real STeP toolchain in unit tests.
    from src.autotune2 import runtime as rt_mod
    monkeypatch.setattr(rt_mod, "_build_non_root_verifier",
                        rt_mod._build_non_root_verifier)  # no-op; just guard against drift

    captured = {}
    def fake_translate(src):
        captured["translate_src"] = src
        return "# translated\n"
    def fake_build(translated, dims, tensors):
        captured["build_args"] = (translated, dims, tensors)
        return (object(), object())
    def fake_dsl_exec(code, dims, tensors, **kwargs):
        captured["dsl_exec_src"] = code
        return torch.zeros(1, 4, 4)

    import src.dsl_to_step as dsl_to_step_mod
    import src.tools as tools_mod
    monkeypatch.setattr(dsl_to_step_mod, "translate", fake_translate)
    monkeypatch.setattr(tools_mod, "_exec_build_graph", fake_build)
    monkeypatch.setattr(tools_mod, "_exec_dsl_ref", fake_dsl_exec)

    make_verifier = build_real_verifier_factory_fn(
        root_kernel="__test_root_kernel_clean__",
        dims={"d": 1},
        check_order="correctness-first",
    )

    class _StubNode:
        path = "root/my_leaf"
        name = "my_leaf"

    verifier = make_verifier(_StubNode(), c, {"x": torch.zeros(4, 4)})
    composed = (
        "def tiled_reference(dims, tensors):\n"
        "    return offchip_store(promote_outer(\n"
        "        my_leaf(tensors['x'], out_shapes=((1, 4, 4),))))\n"
        "\n"
        "def my_leaf(x, *, out_shapes):\n"
        "    return unary_add_imm(x, 1.0)\n"
    )
    result = asyncio.run(verifier(composed))
    assert result.passed, (
        f"variant whose subgraph translates + builds cleanly should be "
        f"admitted; got feedback={result.feedback!r}"
    )
    # Make sure both smoke tests (DSL exec + graph build) actually ran.
    assert captured["dsl_exec_src"] == composed
    assert captured["translate_src"] == composed
    assert captured["build_args"][0] == "# translated\n"


def test_verifier_factory_non_root_surfaces_graph_build_failure_as_feedback(
    monkeypatch,
):
    """If the LLM's proposed contracts produce a stream shape the leaf
    can't consume internally, ``_exec_build_graph`` raises a STeP
    frontend assertion. The verifier should catch it and return it as
    feedback so the LLM can try again — not crash the search loop."""
    import torch
    from src.contract import Contract
    from src.node_signature import TensorArg

    c = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 4),),
        tiled_shapes=((1, 4, 4),),
        tiled_values=(torch.zeros(1, 4, 4),),
        out_shapes=((1, 4, 4),),
        tiled_outputs=(torch.zeros(1, 4, 4),),
        out_is_tuple=False,
        arg_specs=(TensorArg(shape=(4, 4)),),
        arg_is_raw=(False,),
    )

    import src.dsl_to_step as dsl_to_step_mod
    import src.tools as tools_mod
    monkeypatch.setattr(dsl_to_step_mod, "translate", lambda src: "# translated\n")
    monkeypatch.setattr(
        tools_mod, "_exec_dsl_ref",
        lambda code, dims, tensors, **kw: torch.zeros(1, 4, 4),
    )
    def boom(translated, dims, tensors):
        raise AssertionError(
            "stride (4, 1, 16) x out_shape_tiled (4, 4, 64) exceeds "
            "buffer grid (16,) (max_idx=1023, n_tiles=16)."
        )
    monkeypatch.setattr(tools_mod, "_exec_build_graph", boom)

    make_verifier = build_real_verifier_factory_fn(
        root_kernel="__test_root_kernel_buildfail__",
        dims={"d": 1},
        check_order="correctness-first",
    )

    class _StubNode:
        path = "root/my_leaf"
        name = "my_leaf"

    verifier = make_verifier(_StubNode(), c, {"x": torch.zeros(4, 4)})
    composed = (
        "def tiled_reference(dims, tensors):\n"
        "    return offchip_store(promote_outer(\n"
        "        my_leaf(tensors['x'], out_shapes=((1, 4, 4),))))\n"
        "\n"
        "def my_leaf(x, *, out_shapes):\n"
        "    return unary_add_imm(x, 1.0)\n"
    )
    result = asyncio.run(verifier(composed))
    assert not result.passed
    assert "STeP graph build failed" in result.feedback
    assert "stride" in result.feedback and "buffer grid" in result.feedback


def test_verifier_factory_non_root_surfaces_dsl_exec_failure_as_feedback(
    monkeypatch,
):
    """If the LLM's DSL raises during eager execution — the regime that
    used to mean ``score_fn`` would crash inside ``analyze_timing`` —
    the verifier should now catch it BEFORE the score path and feed
    the traceback back to the LLM, matching pass1's
    ``_gate_correctness`` behavior. Locks in that DSL-exec errors are
    surfaced as ``DSL eager-exec smoke test failed`` feedback (not
    graph-build or score errors)."""
    import torch
    from src.contract import Contract
    from src.node_signature import TensorArg

    c = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 4),),
        tiled_shapes=((1, 4, 4),),
        tiled_values=(torch.zeros(1, 4, 4),),
        out_shapes=((1, 4, 4),),
        tiled_outputs=(torch.zeros(1, 4, 4),),
        out_is_tuple=False,
        arg_specs=(TensorArg(shape=(4, 4)),),
        arg_is_raw=(False,),
    )

    import src.dsl_to_step as dsl_to_step_mod
    import src.tools as tools_mod

    # Mimic the bug we just fixed: DSL eager exec raises with the
    # exact crash signature the user originally hit. Graph build path
    # is also stubbed so we can prove DSL exec short-circuits before
    # we ever reach it.
    graph_build_invoked = []
    def fake_dsl_exec(code, dims, tensors, **kwargs):
        raise RuntimeError(
            "stack expects each tensor to be equal size, but got "
            "[1, 23, 1, 512] at entry 0 and [1, 19, 1, 512] at entry 1"
        )
    def fake_translate(src):
        graph_build_invoked.append("translate")
        return "# translated\n"
    def fake_build(translated, dims, tensors):
        graph_build_invoked.append("build")
        return (object(), object())

    monkeypatch.setattr(tools_mod, "_exec_dsl_ref", fake_dsl_exec)
    monkeypatch.setattr(dsl_to_step_mod, "translate", fake_translate)
    monkeypatch.setattr(tools_mod, "_exec_build_graph", fake_build)

    make_verifier = build_real_verifier_factory_fn(
        root_kernel="__test_root_kernel_dsl_exec_fail__",
        dims={"d": 1},
        check_order="correctness-first",
    )

    class _StubNode:
        path = "root/my_leaf"
        name = "my_leaf"

    verifier = make_verifier(_StubNode(), c, {"x": torch.zeros(4, 4)})
    composed = (
        "def tiled_reference(dims, tensors):\n"
        "    return offchip_store(promote_outer(\n"
        "        my_leaf(tensors['x'], out_shapes=((1, 4, 4),))))\n"
        "\n"
        "def my_leaf(x, *, out_shapes):\n"
        "    return unary_add_imm(x, 1.0)\n"
    )
    result = asyncio.run(verifier(composed))
    assert not result.passed
    assert "DSL eager-exec smoke test failed" in result.feedback
    assert "stack expects each tensor to be equal size" in result.feedback
    # In correctness-first mode, DSL-exec failure short-circuits the
    # rest of the correctness gate — graph-build never gets a chance.
    assert graph_build_invoked == [], (
        f"graph-build should be short-circuited after DSL-exec failure; "
        f"got invocations: {graph_build_invoked}"
    )


# --- _contract_conformance_smoke_test ----------------------------------------


def _build_fake_graph(output_layouts):
    """Build a real STeP graph emulating the wrapper output for tests.

    ``output_layouts`` is a list of ``(stream_shape, tile_shape)`` tuples;
    each becomes one wrapper output. We use ``LinearOffChipLoad`` to
    materialize a stream with the requested decomposition (the loader
    *prepends* a leading singleton so the requested stream is
    ``stream_shape[1:]`` if it starts with ``1``, else the loader's
    leading-singleton convention matches what the wrapper sees from
    leaves whose own first op is ``offchip_load``). Each output is wired
    through ``PromoteOuter`` into a separate ``OffChipStore``, mirroring
    ``build_synthetic_wrapper_for_node`` output handling.
    """
    import torch
    # src.tools sets up the step_tl/src path; importing it first makes
    # step_py importable in the test process.
    import src.tools  # noqa: F401
    from networkx import MultiDiGraph
    from step_py.ops import LinearOffChipLoad, OffChipStore, PromoteOuter

    g = MultiDiGraph()
    for stream_shape, tile_shape in output_layouts:
        # LinearOffChipLoad prepends a leading (1,) to the requested
        # out_shape_tiled, so request stream_shape[1:] (and assert that
        # the requested layout starts with the loader's implicit (1,)).
        assert stream_shape[0] == 1, (
            f"_build_fake_graph: stream_shape must start with 1 (LinearOffChipLoad "
            f"convention); got {stream_shape!r}"
        )
        # Underlying tensor must be sized so the loader's tile slicing works:
        # underlying.shape[-2:] = (tile_row * 1, tile_col * 1) and
        # underlying.shape[:-2] gets folded into out_shape_tiled.
        tile_row, tile_col = tile_shape
        out_shape_tiled = tuple(int(d) for d in stream_shape[1:])
        # Construct an underlying torch tensor whose tile-grid product
        # matches out_shape_tiled; LinearOffChipLoad uses
        # ``underlying.shape[-2:] // tile`` for grid counts.
        if not out_shape_tiled:
            grid_dims = (1,)
        else:
            grid_dims = out_shape_tiled
        underlying = torch.zeros(*grid_dims, tile_row, tile_col)
        load = LinearOffChipLoad(
            underlying, stride=(1,) * len(out_shape_tiled),
            out_shape_tiled=out_shape_tiled,
            tile_row=tile_row, tile_col=tile_col, par_dispatch=1,
        )
        g.add_node(load)
        pr = PromoteOuter(g, load)
        OffChipStore(g, pr, par_dispatch=1)
    return g


def _conformance_call(output_layouts, output_contracts, tmp_path):
    """Helper: build a fake graph + run the conformance smoke test."""
    import traceback as tb
    graph = _build_fake_graph(output_layouts)
    fake_translate = lambda src: "# translated\n"
    fake_build = lambda translated, dims, tensors: (graph, None)
    return _contract_conformance_smoke_test(
        "# composed\n", dims={}, tensors={},
        output_contracts=output_contracts,
        translate_fn=fake_translate,
        exec_build_graph=fake_build,
        traceback_mod=tb,
        scratch=tmp_path,
        log=lambda _msg: None,
    )


def test_conformance_passes_when_stream_plus_tile_matches_reshape(tmp_path):
    # Output stream=(1, 64), tile=(16, 32) → combined=(1, 64, 16, 32);
    # contract that truthfully declares that layout passes.
    layouts = [((1, 64), (16, 32))]
    contracts = {
        "out_0": TensorContract(
            reshape=(1, 64, 16, 32), permutation=(0, 1, 2, 3),
        ),
    }
    res = _conformance_call(layouts, contracts, tmp_path)
    assert res.feedback is None, f"unexpected failure: {res.feedback!r}"


def test_conformance_fails_when_total_rank_mismatch(tmp_path):
    # This mirrors the bug from the checkpoint: actual layout has rank 4
    # (`(1, 64, 16, 32)`), declared contract has rank 3 (`(64, 16, 32)`).
    # The leading singleton in the produced layout is what eventually
    # trips ``Parallelize`` downstream — catching it here rejects the
    # variant before it can poison the parent composition.
    layouts = [((1, 64), (16, 32))]
    contracts = {
        "out_0": TensorContract(
            reshape=(64, 16, 32), permutation=(0, 1, 2),
        ),
    }
    res = _conformance_call(layouts, contracts, tmp_path)
    assert res.feedback is not None
    assert "Contract conformance check failed" in res.feedback
    assert "out_0" in res.feedback
    assert "(64, 16, 32)" in res.feedback   # what the LLM declared
    assert "(1, 64, 16, 32)" in res.feedback  # what it actually produced


def test_conformance_fails_when_split_differs_at_same_rank(tmp_path):
    # Both layouts have rank 3 but different stream-vs-tile splits:
    # actual ``stream=(1, 64), tile=(16, 32)`` totals (1, 64, 16, 32);
    # declared ``reshape=(1024, 1, 32)`` is the wrong rank-3 split.
    layouts = [((1, 64), (16, 32))]
    contracts = {
        "out_0": TensorContract(
            reshape=(1024, 1, 32), permutation=(0, 1, 2),
        ),
    }
    res = _conformance_call(layouts, contracts, tmp_path)
    assert res.feedback is not None
    assert "Contract conformance check failed" in res.feedback


def test_conformance_respects_permutation(tmp_path):
    # Same underlying layout, but the contract claims a permutation that
    # reorders the dims. The check must compare after applying it.
    layouts = [((1, 64), (16, 32))]
    # post_permute_shape of reshape=(1, 64, 16, 32) with permutation
    # (1, 0, 2, 3) is (64, 1, 16, 32) — does NOT match actual (1, 64, 16, 32).
    contracts_mismatch = {
        "out_0": TensorContract(
            reshape=(1, 64, 16, 32), permutation=(1, 0, 2, 3),
        ),
    }
    res = _conformance_call(layouts, contracts_mismatch, tmp_path)
    assert res.feedback is not None
    # Identity permutation does match.
    contracts_match = {
        "out_0": TensorContract(
            reshape=(1, 64, 16, 32), permutation=(0, 1, 2, 3),
        ),
    }
    res2 = _conformance_call(layouts, contracts_match, tmp_path)
    assert res2.feedback is None


def test_conformance_multi_output_reports_only_mismatches(tmp_path):
    # Three outputs: out_0 matches, out_1 mismatches, out_2 matches.
    # Feedback must name out_1 only.
    layouts = [
        ((1, 64), (16, 32)),
        ((1, 64), (4, 32)),
        ((1, 64), (4, 32)),
    ]
    contracts = {
        "out_0": TensorContract(reshape=(1, 64, 16, 32), permutation=(0, 1, 2, 3)),
        "out_1": TensorContract(reshape=(64, 4, 32), permutation=(0, 1, 2)),  # rank mismatch
        "out_2": TensorContract(reshape=(1, 64, 4, 32), permutation=(0, 1, 2, 3)),
    }
    res = _conformance_call(layouts, contracts, tmp_path)
    assert res.feedback is not None
    assert "out_1" in res.feedback
    # out_0 / out_2 should NOT appear in the mismatch bullet list, even though
    # their names appear in surrounding boilerplate. Use a specific marker.
    assert "- `out_1`:" in res.feedback
    assert "- `out_0`:" not in res.feedback
    assert "- `out_2`:" not in res.feedback


def test_verifier_passes_conformance_when_layout_matches_contract(monkeypatch):
    """End-to-end: verifier admits a variant whose declared output
    contract correctly describes its produced stream+tile layout."""
    import torch
    from src.contract import Contract
    from src.node_signature import TensorArg

    c = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 4),),
        tiled_shapes=((1, 4, 4),),
        tiled_values=(torch.zeros(1, 4, 4),),
        out_shapes=((1, 4, 4),),
        tiled_outputs=(torch.zeros(1, 4, 4),),
        out_is_tuple=False,
        arg_specs=(TensorArg(shape=(4, 4)),),
        arg_is_raw=(False,),
    )

    graph_for_build = _build_fake_graph([((1, 4), (1, 4))])

    import src.dsl_to_step as dsl_to_step_mod
    import src.tools as tools_mod
    monkeypatch.setattr(dsl_to_step_mod, "translate", lambda src: "# translated\n")
    monkeypatch.setattr(
        tools_mod, "_exec_build_graph",
        lambda translated, dims, tensors: (graph_for_build, None),
    )
    monkeypatch.setattr(
        tools_mod, "_exec_dsl_ref",
        lambda code, dims, tensors, **kw: torch.zeros(1, 4, 4),
    )

    make_verifier = build_real_verifier_factory_fn(
        root_kernel="__test_root_kernel_conformance_ok__",
        dims={"d": 1},
        check_order="correctness-first",
    )

    class _StubNode:
        path = "root/my_leaf"
        name = "my_leaf"

    verifier = make_verifier(_StubNode(), c, {"x": torch.zeros(4, 4)})
    composed = (
        "def tiled_reference(dims, tensors):\n"
        "    return offchip_store(promote_outer(\n"
        "        my_leaf(tensors['x'], out_shapes=((1, 4, 4),))))\n"
        "\n"
        "def my_leaf(x, *, out_shapes):\n"
        "    return unary_add_imm(x, 1.0)\n"
    )
    output_contracts = {
        "out_0": TensorContract(reshape=(1, 4, 1, 4), permutation=(0, 1, 2, 3)),
    }
    result = asyncio.run(verifier(composed, output_contracts))
    assert result.passed, (
        f"variant whose declared contract matches actual layout must be "
        f"admitted; got feedback={result.feedback!r}"
    )


def test_verifier_surfaces_conformance_failure_as_feedback(monkeypatch):
    """End-to-end: verifier rejects a variant whose declared output
    contract lies about the stream/tile split — the exact regime from
    the pre_attn_norm_and_proj LLM-variant bug."""
    import torch
    from src.contract import Contract
    from src.node_signature import TensorArg

    c = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 4),),
        tiled_shapes=((1, 4, 4),),
        tiled_values=(torch.zeros(1, 4, 4),),
        out_shapes=((1, 4, 4),),
        tiled_outputs=(torch.zeros(1, 4, 4),),
        out_is_tuple=False,
        arg_specs=(TensorArg(shape=(4, 4)),),
        arg_is_raw=(False,),
    )

    # Actual produced layout: stream=(1, 4), tile=(1, 4) → (1, 4, 1, 4).
    # Declared (lying) contract: reshape=(4, 4) — rank 2, missing the
    # leading singleton + extra tile-row dim.
    graph_for_build = _build_fake_graph([((1, 4), (1, 4))])

    import src.dsl_to_step as dsl_to_step_mod
    import src.tools as tools_mod
    monkeypatch.setattr(dsl_to_step_mod, "translate", lambda src: "# translated\n")
    monkeypatch.setattr(
        tools_mod, "_exec_build_graph",
        lambda translated, dims, tensors: (graph_for_build, None),
    )
    monkeypatch.setattr(
        tools_mod, "_exec_dsl_ref",
        lambda code, dims, tensors, **kw: torch.zeros(1, 4, 4),
    )

    make_verifier = build_real_verifier_factory_fn(
        root_kernel="__test_root_kernel_conformance_fail__",
        dims={"d": 1},
        check_order="correctness-first",
    )

    class _StubNode:
        path = "root/my_leaf"
        name = "my_leaf"

    verifier = make_verifier(_StubNode(), c, {"x": torch.zeros(4, 4)})
    composed = (
        "def tiled_reference(dims, tensors):\n"
        "    return offchip_store(promote_outer(\n"
        "        my_leaf(tensors['x'], out_shapes=((1, 4, 4),))))\n"
        "\n"
        "def my_leaf(x, *, out_shapes):\n"
        "    return unary_add_imm(x, 1.0)\n"
    )
    output_contracts = {
        "out_0": TensorContract(reshape=(4, 4), permutation=(0, 1)),
    }
    result = asyncio.run(verifier(composed, output_contracts))
    assert not result.passed
    assert "Contract conformance check failed" in result.feedback
    assert "out_0" in result.feedback
