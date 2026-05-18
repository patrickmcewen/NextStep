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
    _derive_output_contracts_from_graph,
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


def test_promote_top_k_runs_picks_concurrently():
    """The rust evaluator should be invoked from multiple threads so wall-
    clock cost scales with the slowest pick, not the sum. We lock all calls
    inside the evaluator and observe that more than one thread is in-flight
    simultaneously — proves promote_top_k isn't running them serially."""
    import threading
    import time

    # All 4 are mutually non-dominated (each lower in one dim, higher in
    # the other) so pick_top_k_pareto_entries returns all of them.
    lib = _root_lib_with_entries([
        (10, 400, "a"), (20, 300, "b"), (30, 200, "c"), (40, 100, "d"),
    ])
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()
    start_barrier = threading.Barrier(4)

    def rust(src):
        nonlocal in_flight, max_in_flight
        # Wait for all 4 calls to have entered before any returns; if the
        # implementation is serial, this will deadlock and the test fails
        # with a Barrier timeout rather than a wrong value.
        start_barrier.wait(timeout=2.0)
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.01)
        with lock:
            in_flight -= 1
        return (1, 1.0)

    results = promote_top_k(root_library=lib, k=4, rust_evaluate_fn=rust)
    assert len(results) == 4
    assert max_in_flight == 4, (
        f"expected 4 concurrent rust calls, observed peak {max_in_flight}"
    )


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

    # Derivation walks ``graph.nodes`` for OffChipStore nodes, so the
    # fake build must return a graph that contains exactly one — matches
    # the synthetic wrapper's one-store-per-output contract.
    fake_graph = _build_fake_graph([((1, 4), (1, 4))])

    captured = {}
    def fake_translate(src):
        captured["translate_src"] = src
        return "# translated\n"
    def fake_build(translated, dims, tensors):
        captured["build_args"] = (translated, dims, tensors)
        return (fake_graph, object())
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


# --- _derive_output_contracts_from_graph -------------------------------------


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


def _derive_call(output_layouts):
    """Helper: build a fake graph + run the derivation helper."""
    graph = _build_fake_graph(output_layouts)
    fake_translate = lambda src: "# translated\n"
    fake_build = lambda translated, dims, tensors: (graph, None)
    return _derive_output_contracts_from_graph(
        "# composed\n", dims={}, tensors={},
        translate_fn=fake_translate,
        exec_build_graph=fake_build,
        log=lambda _msg: None,
    )


def test_derive_returns_identity_contract_with_stream_plus_tile_reshape():
    # Output stream=(1, 64), tile=(16, 32) → derived contract carries
    # reshape=(1, 64, 16, 32) with identity permutation, exactly the
    # concatenation of the produced stream + tile shapes.
    layouts = [((1, 64), (16, 32))]
    derived = _derive_call(layouts)
    assert derived == {
        "out_0": TensorContract(
            reshape=(1, 64, 16, 32), permutation=(0, 1, 2, 3),
        ),
    }


def test_derive_keys_by_offchipstore_order():
    # Multiple outputs are keyed ``out_0``, ``out_1``, ``out_2`` in
    # OffChipStore declaration order (which matches the order the
    # synthetic wrapper emits them in).
    layouts = [
        ((1, 64), (16, 32)),
        ((1, 8, 4), (1, 32)),
        ((1, 64), (4, 32)),
    ]
    derived = _derive_call(layouts)
    assert set(derived.keys()) == {"out_0", "out_1", "out_2"}
    assert derived["out_0"].reshape == (1, 64, 16, 32)
    assert derived["out_1"].reshape == (1, 8, 4, 1, 32)
    assert derived["out_2"].reshape == (1, 64, 4, 32)
    for c in derived.values():
        assert c.permutation == tuple(range(len(c.reshape)))


def test_verifier_surfaces_derived_output_contracts(monkeypatch):
    """End-to-end: a passing non-root verifier exposes the derived output
    contracts (reshape = produced stream + tile, identity permutation)
    via ``VerifyResult.derived_output_contracts`` — they are no longer
    declared by the LLM, so the search loop must read them off the
    verifier's return."""
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
        root_kernel="__test_root_kernel_derive__",
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
    assert result.passed, f"variant must be admitted; feedback={result.feedback!r}"
    assert result.derived_output_contracts == {
        "out_0": TensorContract(reshape=(1, 4, 1, 4), permutation=(0, 1, 2, 3)),
    }
