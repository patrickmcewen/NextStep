"""Unit tests for autotune2 search driver (Phase 5).

All external dependencies (LLM agent, correctness verifier, analytical
scorer) are mocked out. These tests validate the search driver's
control flow, library state evolution, descendant gathering, and
synthetic-wrapper generation — Phase 6 wires the real Anthropic /
4-gate / STeP components.
"""

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    TensorContract,
    library_cell,
    vanilla_contract_for,
)
from src.autotune2.search import (
    AutotuneResult,
    NodePromptInputs,
    SearchConfig,
    VerifyResult,
    autotune,
    build_node_tensors_dict,
    build_synthetic_wrapper_for_node,
    build_variant_callables,
    entry_for_variant,
    find_pass1_baseline_entry,
    gather_descendants_postorder,
    library_to_variant_registry,
    render_library_as_variant_summaries,
    search_leaf,
    search_parent,
    write_library_snapshot,
)
from src.autotune2.ace_context import AceContextConfig, AceContextManager
from src.autotune2.sim_manager import AnalyticalOnly


def _stub_prompt_inputs(node_name: str = "node") -> NodePromptInputs:
    """Minimal NodePromptInputs for tests (content doesn't affect search flow)."""
    return NodePromptInputs(
        function_signature=f"def {node_name}(x, *, out_shapes):",
        pytorch_reference="# ref",
        dims_block="```json\n{}\n```",
        tensors_block="",
    )
from src.contract import Contract
from src.node_signature import ListOfIntArg, ListOfTensorArg, TensorArg
from src.planner import PlanNode, Tree


def run(coro):
    return asyncio.run(coro)


class _TickingClock:
    """Deterministic clock: each read advances one second."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        now = self._now
        self._now += 1.0
        return now


def _search_config(
    turns: int,
    attempt_budgets_bytes: list[int | None],
    *,
    branches: int = 1,
):
    return SearchConfig(
        time_limit_seconds=float(turns * len(attempt_budgets_bytes) * branches),
        attempt_budgets_bytes=attempt_budgets_bytes,
        clock=_TickingClock(),
    )


# --- Contract / Tree fixtures ------------------------------------------------


def _raw_contract(
    arg_specs: dict[str, tuple[int, ...]],
    out_shapes: tuple[tuple[int, ...], ...],
) -> Contract:
    """Build a Contract where every TensorArg is RAW."""
    arg_names = tuple(arg_specs.keys())
    vanilla = tuple(arg_specs.values())
    specs = tuple(TensorArg(shape=s) for s in vanilla)
    # Use the vanilla shape as tile-stream shape too (irrelevant for tests).
    tiled = vanilla
    tiled_values = tuple(torch.zeros(s) for s in vanilla)
    return Contract(
        arg_names=arg_names,
        vanilla_shapes=vanilla,
        tiled_shapes=tiled,
        tiled_values=tiled_values,
        out_shapes=out_shapes,
        arg_specs=specs,
        arg_is_raw=tuple(True for _ in arg_names),
    )


def _leaf(name: str) -> PlanNode:
    return PlanNode(
        name=name, path=f"root/{name}",
        reference_code="# noop\n", refactored_code=None,
        is_leaf=True, children=(),
    )


def _parent(name: str, children: tuple[PlanNode, ...]) -> PlanNode:
    return PlanNode(
        name=name, path=f"root/{name}",
        reference_code="# parent\n", refactored_code="# refactored\n",
        is_leaf=False, children=children,
    )


# --- build_synthetic_wrapper_for_node ----------------------------------------


def test_wrapper_raw_only_inputs():
    c = _raw_contract({"x": (64, 512)}, out_shapes=((4, 16, 512),))
    src = build_synthetic_wrapper_for_node(
        node_name="my_leaf", parent_contract=c, input_contracts={},
    )
    assert "def tiled_reference(dims, tensors):" in src
    # RAW TensorArgs pass straight through; no offchip_load is synthesized for
    # them — the leaf body is responsible for that.
    assert 'my_leaf(tensors["x"]' in src
    assert "offchip_load(" not in src
    assert "out_shapes=((4, 16, 512),)" in src
    # Output flows through promote_outer to satisfy OffChipStore's stream
    # rank >= 1 startup constraint.
    assert "offchip_store(promote_outer(result))" in src


def test_wrapper_emits_offchip_load_for_on_chip_arg_identity_contract():
    """Identity contract on an on-chip TensorArg lowers to a vanilla-strided
    offchip_load followed by a leading-singleton flatten. The leaf consumes
    the flattened on-chip stream as its positional arg."""
    c = _raw_contract({"x": (64, 16, 32)}, out_shapes=((64, 16, 32),))
    c = replace(c, arg_is_raw=(False,))
    src = build_synthetic_wrapper_for_node(
        node_name="leaf", parent_contract=c,
        input_contracts={"x": vanilla_contract_for((64, 16, 32))},
    )
    # offchip_load strides + tile match the vanilla shape; leading-singleton
    # absorbed via flatten(min_rank=0, max_rank=1) so the leaf sees a rank-1
    # stream.
    assert (
        '_x_in = offchip_load(tensors["x"], stride=(1,), '
        "out_shape_tiled=(64,), tile_row=16, tile_col=32)"
    ) in src
    assert "_x_in = flatten(_x_in, min_rank=0, max_rank=1)" in src
    assert "leaf(_x_in" in src


def test_wrapper_on_chip_arg_permuted_leading_axes():
    """Permutations of the leading reshape axes are realizable as a strided
    walk over physical tiles — the wrapper picks strides matching the
    permuted batch order. The last two reshape axes must remain in place
    (tile = vanilla[-2:])."""
    c = _raw_contract({"x": (64, 16, 32)}, out_shapes=((64, 16, 32),))
    c = replace(c, arg_is_raw=(False,))
    # reshape = (8, 8, 16, 32), permutation = (1, 0, 2, 3) -> applied
    # = (8, 8, 16, 32) with the leading batch axes swapped.
    contract = TensorContract(
        reshape=(8, 8, 16, 32), permutation=(1, 0, 2, 3),
    )
    src = build_synthetic_wrapper_for_node(
        node_name="leaf", parent_contract=c, input_contracts={"x": contract},
    )
    # Stream axis 0 walks reshape axis 1 (stride 1); axis 1 walks reshape
    # axis 0 (stride 8). out_shape_tiled mirrors the permuted leading dims.
    assert (
        '_x_in = offchip_load(tensors["x"], stride=(1, 8), '
        "out_shape_tiled=(8, 8), tile_row=16, tile_col=32)"
    ) in src
    # Two stream dims so flatten absorbs index 0 (outermost = leading 1) and
    # index 1 (first applied stream dim).
    assert "_x_in = flatten(_x_in, min_rank=1, max_rank=2)" in src


def test_wrapper_sub_tiled_within_vanilla_tail():
    """Sub-tiling vanilla[-2:] (tile_row < vanilla[-2]) — splitting the row
    dim into row_grid * tile_row produces a strided load that walks the
    row-grid as an outer stream axis."""
    c = _raw_contract({"x": (64, 16, 32)}, out_shapes=((64, 16, 32),))
    c = replace(c, arg_is_raw=(False,))
    # Splits the row dim into (4, 4) — the contract's tile shrinks to (4, 32).
    contract = TensorContract(
        reshape=(64, 4, 4, 32), permutation=(0, 1, 2, 3),
    )
    src = build_synthetic_wrapper_for_node(
        node_name="leaf", parent_contract=c, input_contracts={"x": contract},
    )
    # Stride: outer reshape axis 0 (=64) walks one tile-row-block per step
    # (4 row-tiles per "vanilla batch") and the second axis (=4) walks one
    # row-tile per step.
    assert (
        '_x_in = offchip_load(tensors["x"], stride=(4, 1), '
        "out_shape_tiled=(64, 4), tile_row=4, tile_col=32)"
    ) in src


def test_wrapper_fully_streamed_tile_one_by_one():
    """Pass-1's "fully streamed" representation factors a (1, 1) tile onto
    the end of the reshape so the leaf sees the full vanilla shape as a
    stream. The wrapper computes per-axis strides that walk every vanilla
    element as its own physical tile."""
    c = _raw_contract({"x": (4, 4, 64, 32)}, out_shapes=((4, 4, 64, 32),))
    c = replace(c, arg_is_raw=(False,))
    contract = TensorContract(
        reshape=(4, 4, 64, 32, 1, 1), permutation=(0, 1, 2, 3, 4, 5),
    )
    src = build_synthetic_wrapper_for_node(
        node_name="leaf", parent_contract=c, input_contracts={"x": contract},
    )
    # Each tile = 1 element. Stride is just the row-major element-major
    # stride of the reshape's leading axes: (4*64*32, 64*32, 32, 1).
    assert (
        '_x_in = offchip_load(tensors["x"], stride=(8192, 2048, 32, 1), '
        "out_shape_tiled=(4, 4, 64, 32), tile_row=1, tile_col=1)"
    ) in src
    # m == 4 stream dims; flatten absorbs the outermost two (the leading 1
    # from offchip_load and the first applied stream dim).
    assert "_x_in = flatten(_x_in, min_rank=3, max_rank=4)" in src


def test_wrapper_rejects_missing_input_contract():
    c = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))
    c = replace(c, arg_is_raw=(False,))
    with pytest.raises(AssertionError, match="missing entries"):
        build_synthetic_wrapper_for_node(
            node_name="bad", parent_contract=c, input_contracts={},
        )


def test_wrapper_multi_output_destructures_and_stores_each():
    """A node with multiple outputs (e.g. pre_attention returning Q, K, V)
    cannot be passed straight to a single ``offchip_store`` — the synthetic
    wrapper must destructure the return tuple and emit one store per output
    so the translated graph satisfies OffChipStore's single-Stream input
    contract.
    """
    c = _raw_contract(
        {"x": (64, 512)},
        out_shapes=((64, 16, 32), (64, 4, 32), (64, 4, 32)),
    )
    src = build_synthetic_wrapper_for_node(
        node_name="pre_attention", parent_contract=c, input_contracts={},
    )
    # Destructured tuple assignment
    assert "_out_0, _out_1, _out_2 = pre_attention(" in src
    # One offchip_store per output (two bare + one returned = three total)
    assert src.count("offchip_store(") == 3
    # Each output goes through promote_outer to satisfy OffChipStore's
    # stream rank >= 1 startup constraint.
    assert "offchip_store(promote_outer(_out_0))" in src
    assert "offchip_store(promote_outer(_out_1))" in src
    assert "return offchip_store(promote_outer(_out_2))" in src


def test_wrapper_list_args_pass_through():
    c = Contract(
        arg_names=("x", "weights", "lens"),
        vanilla_shapes=((4, 8), (), ()),
        tiled_shapes=((4, 8), (), ()),
        tiled_values=(torch.zeros(4, 8), [torch.zeros(2, 2)], [3, 5]),
        out_shapes=((1, 4, 8),),
        arg_specs=(
            TensorArg(shape=(4, 8)),
            ListOfTensorArg(length=1, elem_shape=(2, 2)),
            ListOfIntArg(length=2),
        ),
        arg_is_raw=(True, True, True),
    )
    src = build_synthetic_wrapper_for_node(
        node_name="ln", parent_contract=c, input_contracts={},
    )
    # List args are referenced exactly like tensor args; the DSL function
    # itself handles their list-ness internally.
    assert 'tensors["weights"]' in src
    assert 'tensors["lens"]' in src


# --- build_node_tensors_dict --------------------------------------------------


def test_node_tensors_dict_shapes():
    c = _raw_contract({"x": (4, 8), "y": (16,)}, out_shapes=((1, 4, 8),))
    d = build_node_tensors_dict(c)
    assert set(d.keys()) == {"x", "y"}
    assert tuple(d["x"].shape) == (4, 8)
    assert tuple(d["y"].shape) == (16,)


def test_node_tensors_dict_lists_pass_through():
    c = Contract(
        arg_names=("w", "lens"),
        vanilla_shapes=((), ()),
        tiled_shapes=((), ()),
        tiled_values=([torch.ones(2)], [1, 2]),
        out_shapes=((1, 2, 2),),
        arg_specs=(ListOfTensorArg(length=1, elem_shape=(2,)), ListOfIntArg(length=2)),
        arg_is_raw=(True, True),
    )
    d = build_node_tensors_dict(c)
    assert isinstance(d["w"], list) and len(d["w"]) == 1
    assert d["lens"] == [1, 2]


def test_node_tensors_dict_preserves_real_values_for_dispatch_tensors():
    """RAW dispatch tensors (e.g. expert_onehot) must keep their real
    values, not be replaced with zeros — the functional executor in
    timing reads them to compute per-expert active-token counts.
    """
    onehot = torch.zeros(8, 2, 4)
    onehot[0, 0, 1] = 1
    onehot[1, 0, 3] = 1
    onehot[5, 1, 2] = 1
    expected_sum = float(onehot.abs().sum())
    c = Contract(
        arg_names=("expert_onehot",),
        vanilla_shapes=((8, 2, 4),),
        tiled_shapes=((8, 2, 4),),
        tiled_values=(onehot,),
        out_shapes=((1, 8, 4),),
        arg_specs=(TensorArg(shape=(8, 2, 4)),),
        arg_is_raw=(True,),
    )
    d = build_node_tensors_dict(c)
    assert tuple(d["expert_onehot"].shape) == (8, 2, 4)
    assert float(d["expert_onehot"].abs().sum()) == expected_sum


def test_node_tensors_dict_reshapes_tiled_value_to_vanilla():
    """For on-chip TensorArgs the tile-stream shape factors differently
    from vanilla; build_node_tensors_dict must lay the captured values
    back out at vanilla shape so the wrapper's offchip_load (which
    reads element-major from vanilla memory) sees the same bytes the
    leaf was authored against.
    """
    vanilla = (64, 512)
    tiled = (64, 1, 512)
    val = torch.arange(64 * 512, dtype=torch.float32).reshape(tiled)
    c = Contract(
        arg_names=("normed_2",),
        vanilla_shapes=(vanilla,),
        tiled_shapes=(tiled,),
        tiled_values=(val,),
        out_shapes=((1, 64, 512),),
        arg_specs=(TensorArg(shape=vanilla),),
        arg_is_raw=(False,),
    )
    d = build_node_tensors_dict(c)
    assert tuple(d["normed_2"].shape) == vanilla
    # Element ordering must match: tiled[i, 0, j] -> vanilla[i, j].
    assert torch.equal(d["normed_2"], val.reshape(vanilla))


# --- gather_descendants_postorder --------------------------------------------


def test_gather_descendants_empty_for_leaf():
    e = DesignEntry(dsl="# leaf\n")
    assert gather_descendants_postorder(e) == []


def test_gather_descendants_post_order_two_levels():
    grandchild = DesignEntry(dsl="# grandchild\n")
    childA = DesignEntry(dsl="# childA\n", children_picks={"root/a/gc": grandchild})
    childB = DesignEntry(dsl="# childB\n")
    parent = DesignEntry(
        dsl="# parent\n",
        children_picks={"root/a": childA, "root/b": childB},
    )
    out = gather_descendants_postorder(parent)
    # grandchild before childA; childA before childB; parent NOT included.
    assert out == ["# grandchild\n", "# childA\n", "# childB\n"]


# --- find_pass1_baseline_entry -----------------------------------------------


def test_find_pass1_baseline_entry_returns_the_pass1_entry_even_when_llm_is_faster():
    """Even if an LLM variant lands in the identity-contract cell with
    fewer cycles, the baseline picker must return the pass-1 entry."""
    lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    cell = library_cell(lib, {"x": c_id}, {"out_0": c_id})
    pass1 = DesignEntry(
        dsl="# pass1\n", input_contracts={"x": c_id},
        output_contracts={"out_0": c_id},
        cycles=100, on_chip=200, provenance="pass1_baseline",
    )
    llm = DesignEntry(
        dsl="# llm faster\n", input_contracts={"x": c_id},
        output_contracts={"out_0": c_id},
        cycles=50, on_chip=200, provenance="llm_attempt_0_turn_0",
    )
    cell.append(pass1)
    cell.append(llm)
    got = find_pass1_baseline_entry(lib)
    assert got is pass1


def test_find_pass1_baseline_entry_searches_across_cells():
    """The pass-1 baseline may not always live in the first cell after
    library mutation; the helper must scan every cell."""
    lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    c_other = TensorContract(reshape=(4, 8), permutation=(1, 0))
    other = library_cell(lib, {"x": c_other}, {"out_0": c_id})
    other.append(DesignEntry(
        dsl="# llm\n", provenance="llm_attempt_0_turn_0", cycles=10, on_chip=10,
    ))
    pass1_cell = library_cell(lib, {"x": c_id}, {"out_0": c_id})
    pass1_entry = DesignEntry(
        dsl="# pass1\n", provenance="pass1_baseline", cycles=100, on_chip=100,
    )
    pass1_cell.append(pass1_entry)
    assert find_pass1_baseline_entry(lib) is pass1_entry


def test_find_pass1_baseline_entry_asserts_missing():
    lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    cell = library_cell(lib, {"x": c_id}, {"out_0": c_id})
    cell.append(DesignEntry(dsl="# llm\n", provenance="llm_attempt_0_turn_0"))
    with pytest.raises(AssertionError, match="expected exactly one"):
        find_pass1_baseline_entry(lib)


def test_find_pass1_baseline_entry_asserts_duplicate():
    lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    cell = library_cell(lib, {"x": c_id}, {"out_0": c_id})
    cell.append(DesignEntry(dsl="# a\n", provenance="pass1_baseline"))
    cell.append(DesignEntry(dsl="# b\n", provenance="pass1_baseline"))
    with pytest.raises(AssertionError, match="expected exactly one"):
        find_pass1_baseline_entry(lib)


# --- library_to_variant_registry / entry_for_variant -------------------------


def _sample_lib() -> NodeLibrary:
    """Two cells, three entries total — the second cell has two Pareto
    entries so the per-entry flatten exposes three variant indices."""
    lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    c_perm = TensorContract(reshape=(4, 8), permutation=(1, 0))
    cell0 = library_cell(lib, {"x": c_id}, {"out_0": c_id})
    cell0.append(DesignEntry(dsl="# v0\n", cycles=100, on_chip=200))
    cell1 = library_cell(lib, {"x": c_perm}, {"out_0": c_id})
    cell1.append(DesignEntry(dsl="# v1a\n", cycles=80, on_chip=300))
    cell1.append(DesignEntry(dsl="# v1b\n", cycles=120, on_chip=150))
    return lib


def test_library_to_variant_registry_per_entry():
    lib = _sample_lib()
    reg = library_to_variant_registry(lib)
    # 1 entry in cell0 + 2 entries in cell1 = 3 indices total
    assert set(reg.keys()) == {0, 1, 2}
    assert reg[0]["input_contracts"]["x"] == vanilla_contract_for((4, 8))
    # idx 1 and 2 share boundary contracts (cell1) but are distinct entries
    perm = TensorContract(reshape=(4, 8), permutation=(1, 0))
    assert reg[1]["input_contracts"]["x"] == perm
    assert reg[2]["input_contracts"]["x"] == perm


def test_entry_for_variant_resolves_back():
    lib = _sample_lib()
    e0 = entry_for_variant(lib, 0)
    e1 = entry_for_variant(lib, 1)
    e2 = entry_for_variant(lib, 2)
    assert e0.dsl == "# v0\n"
    assert e1.dsl == "# v1a\n"
    assert e2.dsl == "# v1b\n"


def test_entry_for_variant_out_of_range():
    with pytest.raises(AssertionError, match="out of range"):
        entry_for_variant(_sample_lib(), 99)


# --- render_library_as_variant_summaries -------------------------------------


def test_render_summaries_one_row_per_entry():
    lib = _sample_lib()
    summaries = render_library_as_variant_summaries(lib)
    # three entries → three summaries, indices match registry order
    assert [s.variant_index for s in summaries] == [0, 1, 2]
    assert (summaries[0].cycles, summaries[0].on_chip) == (100, 200)
    assert (summaries[1].cycles, summaries[1].on_chip) == (80, 300)
    assert (summaries[2].cycles, summaries[2].on_chip) == (120, 150)


def test_render_summaries_filters_dominated_entries():
    """The library accumulates every admitted variant (admit_to_cell
    doesn't evict), but the parent LLM should only see Pareto-non-
    dominated entries. Indices are preserved from the full iteration
    so they still round-trip through library_to_variant_registry /
    entry_for_variant."""
    lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    cell = library_cell(lib, {"x": c_id}, {"out_0": c_id})
    cell.append(DesignEntry(dsl="# pareto_best\n", cycles=80, on_chip=200))
    cell.append(DesignEntry(dsl="# dominated\n", cycles=120, on_chip=300))
    cell.append(DesignEntry(dsl="# other_pareto\n", cycles=200, on_chip=100))

    summaries = render_library_as_variant_summaries(lib)
    # Only the two non-dominated entries surface to the parent agent.
    rendered = [(s.variant_index, s.cycles, s.on_chip) for s in summaries]
    assert (0, 80, 200) in rendered
    assert (2, 200, 100) in rendered
    assert all(s.cycles != 120 for s in summaries)
    # Indices still round-trip into the full library (dominated entry at
    # idx 1 remains addressable for variant binding).
    assert entry_for_variant(lib, 1).dsl == "# dominated\n"
    assert len(library_to_variant_registry(lib)) == 3


# --- build_variant_callables -------------------------------------------------


def test_build_variant_callables_creates_named_stubs():
    import torch.nn as nn
    variants = {
        0: {
            "input_contracts": {},
            "output_contracts": {"out_0": vanilla_contract_for((4, 8))},
        },
        1: {
            "input_contracts": {},
            "output_contracts": {"out_0": vanilla_contract_for((4, 8))},
        },
    }
    out = build_variant_callables(
        child_name="foo",
        arg_names=("x",),
        arg_specs=(TensorArg(shape=(4, 8)),),
        output_names=("out_0",),
        ref_module=nn.Identity(),
        variants=variants,
    )
    assert set(out.keys()) == {"foo_0", "foo_1"}
    # The returned callables share the variant-stub signature: positional
    # args + out_shapes keyword.
    from src.step_dsl import StepTensor
    res = out["foo_0"](torch.zeros(4, 8), out_shapes=((1, 4, 8),))
    assert isinstance(res, StepTensor)


# --- search_leaf -------------------------------------------------------------


def _make_leaf_response(reshape, perm, body: str = "    return None"):
    # These tests use RAW leaf inputs, so the emitted input-contract map
    # must be empty. Keep reshape/perm as a DSL comment marker so repeated
    # responses are structurally distinct for dedup tests.
    return (
        "```yaml\n"
        "parent_input_contracts: {}\n"
        "```\n\n"
        "```python\n"
        f"def my_leaf(x, *, out_shapes):\n{body}\n"
        f"    # marker reshape={list(reshape)} perm={list(perm)}\n"
        "```\n"
    )


def test_search_leaf_seeds_baseline_and_admits_llm_proposals(tmp_path):
    """Two LLM turns: one valid alternative, one with garbage YAML.
    Library ends with baseline + 1 alternative."""
    responses = [
        _make_leaf_response(reshape=(4, 4, 8), perm=(1, 0, 2)),
        "no fenced blocks at all, this should be discarded",
    ]
    call_idx = [0]

    async def agent(_conversation):
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    # Score: baseline=100, llm proposal=50 — Pareto-distinct cells, so both
    # admitted (different output_contracts keys).
    score_call = [0]
    def score(_src):
        score_call[0] += 1
        return (100 if score_call[0] == 1 else 50, 999)

    node = _leaf("my_leaf")
    contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))

    lib = run(search_leaf(
        node=node,
        parent_contract=contract,
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(score),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(1, [None]*2),
    ))
    # variants.py was emitted
    assert (tmp_path / "leaf" / "variants.py").exists()
    # library has at least baseline + one llm variant cell
    reg = library_to_variant_registry(lib)
    assert len(reg) >= 2


def test_search_leaf_writes_pass1_baseline_and_per_turn_score_artifacts(tmp_path):
    """The pass-1 baseline (cycles, on_chip) must land in the node-level
    ckpt_dir as ``pass1_baseline_score.json`` and each ACCEPTED turn must
    drop a ``score.json`` listing the admitted entries' scores. This lets
    a reader inspect the attempt directory directly instead of having to
    chase the numbers through the next attempt's user prompt."""
    responses = [_make_leaf_response(reshape=(4, 4, 8), perm=(1, 0, 2))]
    call_idx = [0]

    async def agent(_conversation):
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    score_call = [0]
    def score(_src):
        score_call[0] += 1
        # baseline = (111, 222); LLM proposal = (333, 444)
        return (111, 222) if score_call[0] == 1 else (333, 444)

    run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(score),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(1, [None]*1),
    ))

    baseline_path = tmp_path / "leaf" / "pass1_baseline_score.json"
    assert baseline_path.exists(), "pass1 baseline score not persisted"
    baseline = json.loads(baseline_path.read_text())
    assert baseline == {"cycles": 111, "on_chip": 222, "provenance": "pass1_baseline"}

    score_path = (
        tmp_path / "leaf" / "baseline_0_attempt_0_binf" / "turn_0" / "score.json"
    )
    assert score_path.exists(), "accepted-turn score not persisted"
    payload = json.loads(score_path.read_text())
    assert payload == {
        "entries": [
            {"cycles": 333, "on_chip": 444,
             "provenance": "llm_baseline_0_attempt_0_binf_turn_0"},
        ],
    }


def test_search_leaf_appends_feedback_on_parse_fail_then_recovers(tmp_path):
    """Within a single attempt, a bad first turn should append feedback to
    the conversation and the next turn should see it. We verify by capturing
    the conversation each call sees and asserting the second turn's
    conversation includes a user-role feedback message."""
    captured_convos: list[list[dict]] = []
    responses = [
        "garbage — no fenced blocks",
        # Non-identity output contract → lands in a different Pareto cell
        _make_leaf_response(reshape=(4, 1, 8), perm=(1, 0, 2)),
    ]
    call_idx = [0]

    async def agent(conversation: list[dict]):
        # Snapshot conversation at call time
        captured_convos.append([dict(m) for m in conversation])
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    lib = run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (50, 100)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(2, [None]*1),
    ))
    # Two agent calls within one attempt
    assert call_idx[0] == 2
    # First call had only the initial user prompt
    assert len(captured_convos[0]) == 1
    assert captured_convos[0][0]["role"] == "user"
    # Second call has user prompt + assistant garbage + feedback user msg
    assert len(captured_convos[1]) == 3
    assert captured_convos[1][1]["role"] == "assistant"
    assert captured_convos[1][2]["role"] == "user"
    assert "could not be parsed" in captured_convos[1][2]["content"]
    # The successful second turn admitted an entry beyond the baseline
    # (different output contracts cell)
    reg = library_to_variant_registry(lib)
    assert len(reg) >= 2


def test_search_leaf_stops_attempt_when_pass_time_limit_expires(tmp_path):
    """The proposal loop should consult a pass deadline, not a turn cap."""
    captured_convos: list[list[dict]] = []
    clock = _TickingClock()

    async def agent(conversation: list[dict]):
        captured_convos.append([dict(m) for m in conversation])
        return "garbage — no fenced blocks"

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (50, 100)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=SearchConfig(
            time_limit_seconds=2.0,
            attempt_budgets_bytes=[None],
            clock=clock,
        ),
    ))

    assert len(captured_convos) == 2


def test_search_leaf_ace_refresh_starts_new_logged_session(tmp_path):
    """ACE refresh windows restart the lane conversation under unique
    session directories and inject refreshed context into the next prompt."""
    captured_prompts: list[str] = []

    async def refresh_fn(
        *,
        playbook: str,
        events: list[dict],
        metadata: dict,
        attempt_dir: Path,
        completed_session_index: int,
        next_session_index: int,
    ) -> str:
        assert playbook == "initial guidance"
        assert len(events) == 1
        assert metadata["node_path"] == "root/my_leaf"
        assert completed_session_index == 0
        assert next_session_index == 1
        assert attempt_dir.name == "baseline_0_attempt_0_binf"
        return "refreshed guidance"

    ace_context = AceContextManager(
        AceContextConfig(
            enabled=True,
            refresh_interval_turns=1,
            initial_playbook="initial guidance",
        ),
        refresh_fn=refresh_fn,
    )

    async def agent(conversation: list[dict]):
        captured_prompts.append(conversation[0]["content"])
        return "garbage — no fenced blocks"

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (50, 100)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=SearchConfig(
            time_limit_seconds=2.0,
            attempt_budgets_bytes=[None],
            clock=_TickingClock(),
            ace_context=ace_context,
        ),
    ))

    assert "initial guidance" in captured_prompts[0]
    assert "refreshed guidance" in captured_prompts[1]
    assert (tmp_path / "leaf" / "baseline_0_attempt_0_binf"
            / "session_0" / "turn_0" / "status.txt").exists()
    assert (tmp_path / "leaf" / "baseline_0_attempt_0_binf"
            / "session_1" / "turn_0" / "status.txt").exists()
    session_log = (
        tmp_path / "leaf" / "baseline_0_attempt_0_binf" / "ace_sessions.jsonl"
    ).read_text()
    assert '"session_index": 0' in session_log
    assert '"session_index": 1' in session_log
    event_log = (
        tmp_path / "leaf" / "baseline_0_attempt_0_binf" / "ace_events.jsonl"
    ).read_text()
    assert '"global_turn": 0' in event_log
    assert '"global_turn": 1' in event_log
    assert '"playbook_version": 1' in event_log


def test_search_leaf_fresh_attempt_includes_accepted_summary(tmp_path):
    """After a successful first attempt, the second attempt's opening user
    prompt should include the rendered Pareto-front summary so the LLM can
    avoid duplicating already-accepted variants.

    Parallel attempts share the same baseline-only accepted_summary at
    start time (they can't see each other's admissions), so the relevant
    invariant is that *budgeted* attempts get distinct prompts via the
    budget block. We verify two attempts with different budgets produce
    two distinct opening prompts, both of which include the baseline in
    the accepted-summary section."""
    captured_first_user_per_attempt: list[str] = []
    responses = [
        _make_leaf_response(reshape=(4, 1, 8), perm=(1, 0, 2)),
        _make_leaf_response(reshape=(2, 2, 8), perm=(0, 1, 2)),
    ]
    call_idx = [0]

    async def agent(conversation: list[dict]):
        # Each attempt opens with a 1-message conversation (just the user
        # prompt). With parallel attempts the prompt is captured per
        # coroutine — record one entry per "opener" call.
        if len(conversation) == 1:
            captured_first_user_per_attempt.append(conversation[0]["content"])
        r = responses[call_idx[0] % len(responses)]
        call_idx[0] += 1
        return r

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (50, 100)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=SearchConfig(
            time_limit_seconds=2.0,
            clock=_TickingClock(),
            # Two attempts at distinct budgets — unlimited and 1024 bytes.
            attempt_budgets_bytes=[None, 1024],
        ),
    ))
    assert len(captured_first_user_per_attempt) == 2
    # Both attempts render the baseline in the accepted-summary section.
    for prompt in captured_first_user_per_attempt:
        assert "Already-accepted variants" in prompt
    # Exactly one of the two attempts is budgeted → exactly one prompt
    # carries the on-chip budget block.
    budget_mentions = sum(
        "On-chip memory budget" in p for p in captured_first_user_per_attempt
    )
    assert budget_mentions == 1
    # The two prompts must differ (budget block presence / wording).
    assert captured_first_user_per_attempt[0] != captured_first_user_per_attempt[1]


def test_search_leaf_default_initial_baselines_matches_current_behavior(tmp_path):
    """``initial_baselines=None`` (default) ⇒ exactly N attempt coros for
    N budgets — one branch (the always-seeded pass-1 baseline). Verifies
    that the multi-pass plumbing didn't accidentally fan out in the
    single-pass call path."""
    agent_calls = [0]

    async def agent(_conversation):
        agent_calls[0] += 1
        # Garbage response — drops on parse; the count is what matters.
        return "no fenced blocks"

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (10, 20)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(1, [None, 1024]),
    ))
    # 2 budgets × 1 baseline (pass-1 only) = 2 agent calls.
    assert agent_calls[0] == 2
    # Attempt dirs use the baseline_0_ prefix even in single-pass mode.
    assert (tmp_path / "leaf" / "baseline_0_attempt_0_binf").exists()
    assert (tmp_path / "leaf" / "baseline_0_attempt_1_b1024").exists()


def test_search_leaf_extra_baselines_spawn_additional_attempts(tmp_path):
    """``initial_baselines=[A, B]`` + 1 budget ⇒ 3 attempt coros
    (pass-1 + A + B). Each conversation opens with its branch's
    ``baseline_dsl`` rendered into the user prompt — verified by
    capturing the opening user prompts and checking the DSL text
    appears in the right one."""
    pass1_dsl = "def my_leaf(x, *, out_shapes):\n    return None\n# pass1\n"
    extra_a = DesignEntry(
        dsl=(
            "def my_leaf(x, *, out_shapes):\n    return None\n# branch_A\n"
        ),
        input_contracts={"x": vanilla_contract_for((4, 8))},
        output_contracts={"out_0": vanilla_contract_for((1, 4, 8))},
        cycles=42, on_chip=84, provenance="prior_pass_A", breakdown="",
    )
    extra_b = DesignEntry(
        dsl=(
            "def my_leaf(x, *, out_shapes):\n    return None\n# branch_B\n"
        ),
        input_contracts={"x": vanilla_contract_for((4, 8))},
        output_contracts={"out_0": vanilla_contract_for((1, 4, 8))},
        cycles=21, on_chip=168, provenance="prior_pass_B", breakdown="",
    )

    opener_prompts: list[str] = []

    async def agent(conversation: list[dict]):
        if len(conversation) == 1:
            opener_prompts.append(conversation[0]["content"])
        return "no fenced blocks"

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl=pass1_dsl,
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (10, 20)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(1, [None], branches=3),
        initial_baselines=[extra_a, extra_b],
    ))

    # 3 branches × 1 budget = 3 attempt openers.
    assert len(opener_prompts) == 3
    # Each opener carries its branch's DSL.
    pass1_hit = sum("# pass1" in p for p in opener_prompts)
    a_hit = sum("# branch_A" in p for p in opener_prompts)
    b_hit = sum("# branch_B" in p for p in opener_prompts)
    assert pass1_hit == 1, f"pass1 baseline DSL appears in {pass1_hit} prompts, expected 1"
    assert a_hit == 1, f"branch_A DSL appears in {a_hit} prompts, expected 1"
    assert b_hit == 1, f"branch_B DSL appears in {b_hit} prompts, expected 1"


def test_search_leaf_per_branch_attempt_dir_naming(tmp_path):
    """Multi-baseline + multi-budget fan-out produces attempt dirs named
    ``baseline_{b_idx}_attempt_{a_idx}_b{label}``. Verifies the naming
    convention end-to-end so artifacts are discoverable per branch."""
    extra = DesignEntry(
        dsl="def my_leaf(x, *, out_shapes):\n    return None\n# branch_X\n",
        input_contracts={"x": vanilla_contract_for((4, 8))},
        output_contracts={"out_0": vanilla_contract_for((1, 4, 8))},
        cycles=1, on_chip=1, provenance="prior_pass", breakdown="",
    )

    async def agent(_conversation):
        return "no fenced blocks"

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (10, 20)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(1, [None, 2048], branches=2),
        initial_baselines=[extra],
    ))

    # 2 baselines × 2 budgets = 4 distinct attempt dirs.
    for b_idx in (0, 1):
        for a_idx, blabel in enumerate(("binf", "b2048")):
            d = tmp_path / "leaf" / f"baseline_{b_idx}_attempt_{a_idx}_{blabel}"
            assert d.exists(), f"missing attempt dir {d}"


def test_search_leaf_skips_failed_verification(tmp_path):
    async def agent(_conversation):
        return _make_leaf_response(reshape=(1, 4, 8), perm=(0, 1, 2))

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=False, feedback="bad")

    lib = run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(lambda _src: (1, 1)),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(1, [None]*3),
    ))
    # Only the baseline cell survives.
    assert len(library_to_variant_registry(lib)) == 1


def test_search_leaf_score_fn_raise_becomes_user_feedback(tmp_path):
    """If ``score_fn`` raises (e.g. analytical timing model crashes deep
    inside ``execute_values``), the search loop must convert the
    exception into LLM feedback and continue searching — not propagate
    the exception out and tear down the whole autotune run."""
    captured_convos: list[list[dict]] = []
    responses = [
        # First variant: score will raise.
        _make_leaf_response(reshape=(1, 4, 8), perm=(0, 1, 2)),
        # Second variant: score returns cleanly so the library gets a non-
        # baseline cell.
        _make_leaf_response(reshape=(4, 1, 8), perm=(1, 0, 2)),
    ]
    call_idx = [0]

    async def agent(conversation: list[dict]):
        captured_convos.append([dict(m) for m in conversation])
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    score_calls = [0]
    def score(_src):
        score_calls[0] += 1
        if score_calls[0] <= 2:
            # 1st = baseline seed, 2nd = first LLM variant. Raise on the
            # variant only — letting the baseline through.
            if score_calls[0] == 2:
                raise RuntimeError(
                    "stack expects each tensor to be equal size, but got "
                    "[1, 23, 1, 512] at entry 0 and [1, 19, 1, 512] at entry 1"
                )
            return (10, 10)
        return (50, 50)

    lib = run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        sim_manager=AnalyticalOnly(score),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=_search_config(2, [None]*1),
    ))
    # Both LLM responses fired in the same attempt before the fake deadline.
    assert call_idx[0] == 2
    # The second agent call's conversation must include the score-fail
    # feedback the search loop appended from the first turn's exception.
    assert len(captured_convos[1]) >= 3
    assert captured_convos[1][-1]["role"] == "user"
    feedback = captured_convos[1][-1]["content"]
    assert "Analytical scorer failed" in feedback
    assert "stack expects each tensor to be equal size" in feedback
    # Final library: baseline + second variant accepted (different cell).
    reg = library_to_variant_registry(lib)
    assert len(reg) >= 2


# --- search_parent ------------------------------------------------------------


def _make_parent_response(child_name: str, variant_idx: int):
    return (
        "```yaml\n"
        "child_picks:\n"
        f"  {child_name}: {variant_idx}\n"
        "parent_input_contracts: {}\n"
        "```\n\n"
        "```python\n"
        "def my_parent(x, *, out_shapes):\n"
        "    return child_under(x, out_shapes=out_shapes)\n"
        "```\n"
    )


def test_search_parent_picks_one_child_entry_and_admits(tmp_path):
    """search_parent resolves child_picks to one concrete DesignEntry per
    child (no Cartesian sweep) and admits the resulting composition."""
    child_node = _leaf("child_under")
    child_lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    cell0 = library_cell(child_lib, {"x": c_id}, {"out_0": c_id})
    cell0.append(DesignEntry(
        dsl="def child_under(x, *, out_shapes):\n    return None\n",
        input_contracts={"x": c_id}, output_contracts={"out_0": c_id},
        cycles=100, on_chip=200, provenance="pass1_baseline",
    ))
    c_perm = TensorContract(reshape=(4, 8), permutation=(1, 0))
    cell1 = library_cell(child_lib, {"x": c_perm}, {"out_0": c_id})
    cell1.append(DesignEntry(
        dsl="def child_under(x, *, out_shapes):\n    return None  # v1\n",
        input_contracts={"x": c_perm}, output_contracts={"out_0": c_id},
        cycles=80, on_chip=300, provenance="llm_turn_0",
    ))

    parent_node = _parent("my_parent", children=(child_node,))
    parent_contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))

    async def agent(_conversation):
        # variant_idx=1 → cell1's single entry (the second per-entry index).
        return _make_parent_response("child_under", variant_idx=1)

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    score_calls = []
    def score(src):
        score_calls.append(src)
        return (50, 100)

    lib = run(search_parent(
        node=parent_node,
        parent_contract=parent_contract,
        pass1_dsl="def my_parent(x, *, out_shapes):\n    return None\n",
        children_libraries={child_node.path: child_lib},
        children_picks_baseline={child_node.path: cell0[0]},
        ckpt_dir=tmp_path / "parent",
        sim_manager=AnalyticalOnly(score),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_parent"),
        config=_search_config(1, [None]*1),
    ))

    # Two scoring calls expected: baseline + exactly one LLM composition
    # (no Cartesian sweep — one DesignEntry per child).
    assert len(score_calls) == 2
    reg = library_to_variant_registry(lib)
    assert len(reg) >= 1


def test_search_parent_admits_only_baseline_when_llm_verify_fails(tmp_path):
    """When every LLM-proposed turn fails verification, the library should
    contain only the pass-1 baseline. ``render_library_as_variant_summaries``
    must not see an empty cell."""
    child_node = _leaf("child_under")
    child_lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    cell0 = library_cell(child_lib, {"x": c_id}, {"out_0": c_id})
    cell0.append(DesignEntry(
        dsl="def child_under(x, *, out_shapes):\n    return None\n",
        input_contracts={"x": c_id}, output_contracts={"out_0": c_id},
        cycles=100, on_chip=200, provenance="pass1_baseline",
    ))

    parent_node = _parent("my_parent", children=(child_node,))
    parent_contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))

    async def agent(_conversation):
        return _make_parent_response("child_under", variant_idx=0)

    verify_calls = [0]
    async def verifier(_src, *_a, **_kw):
        verify_calls[0] += 1
        # Pass for the baseline (first call), fail for every LLM composition.
        if verify_calls[0] == 1:
            return VerifyResult(passed=True)
        return VerifyResult(passed=False, feedback="synthetic fail")

    def score(_src):
        return (50, 100)

    lib = run(search_parent(
        node=parent_node,
        parent_contract=parent_contract,
        pass1_dsl="def my_parent(x, *, out_shapes):\n    return None\n",
        children_libraries={child_node.path: child_lib},
        children_picks_baseline={child_node.path: cell0[0]},
        ckpt_dir=tmp_path / "parent",
        sim_manager=AnalyticalOnly(score),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_parent"),
        config=_search_config(1, [None]*1),
    ))

    for by_out in lib.values():
        for cell in by_out.values():
            assert cell, "search_parent left an empty cell after failed turn"
    render_library_as_variant_summaries(lib)


# --- autotune driver ---------------------------------------------------------


def test_autotune_post_order_walk_populates_all_libraries(tmp_path):
    """Two-node tree: root (parent) + one leaf. Verify both libraries are
    populated and the post-order traversal hits the leaf before the root."""
    leaf = _leaf("inner")
    root = _parent("outer", children=(leaf,))
    root = replace(root, path="root", name="outer")
    leaf = replace(leaf, path="root/inner", name="inner")
    root = replace(root, children=(leaf,))
    tree = Tree(root=root)

    leaf_contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))
    # Root: empty arg list, but Contract requires at least one arg... use
    # parent_contract=None at root (autotune passes None for root).
    pass1_dsls = {
        leaf.path: "def inner(x, *, out_shapes):\n    return None\n",
        root.path: ("def tiled_reference(dims, tensors):\n"
                    "    return inner(tensors['x'], out_shapes=((1, 4, 8),))\n"),
    }
    # Root passes None as its parent_contract; the leaf takes leaf_contract.
    contracts = {leaf.path: leaf_contract, root.path: leaf_contract}

    visited = []
    async def agent(conversation):
        # Record the opening user message's first 30 chars for visit
        # ordering checks. Pass-1 baseline runs ahead of any agent call.
        first_user = next(
            (m["content"] for m in conversation if m["role"] == "user"), "",
        )
        visited.append(first_user[:30])
        # Always return a malformed response so library only has baseline.
        return "garbage"

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    def agent_factory(_system_prompt: str):
        return agent

    prompt_inputs = {
        leaf.path: _stub_prompt_inputs("inner"),
        root.path: _stub_prompt_inputs("outer"),
    }
    sys_prompts = {leaf.path: "sys-leaf", root.path: "sys-root"}

    result = run(autotune(
        plan_tree=tree,
        pass1_dsls=pass1_dsls,
        pass1_contracts=contracts,
        ckpt_dir=tmp_path / "tune",
        make_sim_manager=lambda _tensors: AnalyticalOnly(lambda _src: (10, 10)),
        root_tensors={},
        agent_factory=agent_factory,
        make_verifier=lambda _node, _pc, _t: verifier,
        prompt_inputs=prompt_inputs,
        system_prompts=sys_prompts,
        config=_search_config(1, [None]*1, branches=10),
    ))
    assert set(result.libraries.keys()) == {leaf.path, root.path}
    assert result.root_path == root.path
    # Leaf invoked before root (post-order). Under a shared pass deadline,
    # leaf feedback turns may exhaust the pass before the parent gets an
    # LLM turn; the parent library still receives its baseline.
    assert "inner" in visited[0]
    # variants.py written for each
    assert (tmp_path / "tune" / "autotune2" / leaf.path / "variants.py").exists()
    assert (tmp_path / "tune" / "autotune2" / root.path / "variants.py").exists()


def test_autotune_runs_sibling_leaves_in_parallel(tmp_path):
    """Two leaves under one parent: their agent calls must interleave
    (i.e. both leaves start before either finishes) because the bottom-up
    driver schedules every node as its own task. The parent must still
    wait for both leaves before running.
    """
    leaf_a = _leaf("leaf_a")
    leaf_b = _leaf("leaf_b")
    leaf_a = replace(leaf_a, path="root/leaf_a", name="leaf_a")
    leaf_b = replace(leaf_b, path="root/leaf_b", name="leaf_b")
    root = _parent("outer", children=(leaf_a, leaf_b))
    root = replace(root, path="root", name="outer", children=(leaf_a, leaf_b))
    tree = Tree(root=root)

    leaf_contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))
    pass1_dsls = {
        leaf_a.path: "def leaf_a(x, *, out_shapes):\n    return None\n",
        leaf_b.path: "def leaf_b(x, *, out_shapes):\n    return None\n",
        root.path: ("def tiled_reference(dims, tensors):\n"
                    "    return leaf_a(tensors['x'], out_shapes=((1, 4, 8),))\n"),
    }
    contracts = {
        leaf_a.path: leaf_contract,
        leaf_b.path: leaf_contract,
        root.path: leaf_contract,
    }

    # Order tracker: each agent appends "<node>:start" on entry, awaits a
    # short sleep to yield the event loop, then "<node>:end". If the two
    # leaves run in parallel both starts appear before either end.
    order: list[str] = []
    barrier = asyncio.Event()
    started: set[str] = set()

    def _node_name_from_convo(conversation: list[dict]) -> str:
        # Parent prompts embed child variant tables that contain the
        # child names as substrings, so match against the user prompt's
        # explicit "## Node: <name>" header instead of bare substring.
        first_user = next(
            (m["content"] for m in conversation if m["role"] == "user"), "",
        )
        for cand in ("leaf_a", "leaf_b", "outer"):
            if f"## Node: {cand}" in first_user:
                return cand
        return "?"

    async def agent(conversation):
        nm = _node_name_from_convo(conversation)
        order.append(f"{nm}:start")
        # Release the barrier only once both leaves have entered. The
        # parent (``outer``) doesn't gate the barrier — it only runs
        # after the leaves resolve, and asserting "outer:start" comes
        # after both leaf ends is the actual parent-after-children
        # invariant.
        if nm in ("leaf_a", "leaf_b"):
            started.add(nm)
            if {"leaf_a", "leaf_b"}.issubset(started):
                barrier.set()
            await barrier.wait()
        order.append(f"{nm}:end")
        return "garbage"

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    def agent_factory(_system_prompt: str):
        return agent

    prompt_inputs = {
        leaf_a.path: _stub_prompt_inputs("leaf_a"),
        leaf_b.path: _stub_prompt_inputs("leaf_b"),
        root.path: _stub_prompt_inputs("outer"),
    }
    sys_prompts = {
        leaf_a.path: "sys-leaf-a",
        leaf_b.path: "sys-leaf-b",
        root.path: "sys-root",
    }

    result = run(autotune(
        plan_tree=tree,
        pass1_dsls=pass1_dsls,
        pass1_contracts=contracts,
        ckpt_dir=tmp_path / "tune",
        make_sim_manager=lambda _tensors: AnalyticalOnly(lambda _src: (10, 10)),
        root_tensors={},
        agent_factory=agent_factory,
        make_verifier=lambda _node, _pc, _t: verifier,
        prompt_inputs=prompt_inputs,
        system_prompts=sys_prompts,
        config=_search_config(1, [None]*1, branches=10),
    ))

    # All three libraries populated
    assert set(result.libraries.keys()) == {leaf_a.path, leaf_b.path, root.path}

    # Both leaves started before either finished → interleaved execution.
    first_leaf_starts = [
        order.index("leaf_a:start"),
        order.index("leaf_b:start"),
    ]
    first_leaf_ends = [
        order.index("leaf_a:end"),
        order.index("leaf_b:end"),
    ]
    assert max(first_leaf_starts) < min(first_leaf_ends), (
        f"sibling leaves did not interleave: {order!r}"
    )

    # Parent library exists even when the shared deadline is exhausted by
    # leaves before the parent gets an LLM turn.
    assert root.path in result.libraries


def test_autotune_root_as_leaf_single_node_tree(tmp_path):
    """A plan tree with a single node (root == leaf) — happens when pass-1
    produces no sub-functions. The root has no parent_contract; the search
    must skip the synthetic wrapper and use empty identity contracts."""
    only = _leaf("only")
    only = replace(only, path="root", name="only", is_leaf=True, children=())
    tree = Tree(root=only)

    pass1_dsls = {
        only.path: (
            "def tiled_reference(dims, tensors):\n"
            "    return None\n"
        ),
    }
    # pass1_contracts has no entry for the root — _load_pass1_state skips it.
    contracts: dict = {}

    async def agent(_conversation):
        return "garbage"  # baseline-only library
    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)
    def agent_factory(_sys):
        return agent

    prompt_inputs = {only.path: _stub_prompt_inputs("only")}
    sys_prompts = {only.path: "sys-only"}

    result = run(autotune(
        plan_tree=tree,
        pass1_dsls=pass1_dsls,
        pass1_contracts=contracts,
        ckpt_dir=tmp_path / "tune",
        make_sim_manager=lambda _tensors: AnalyticalOnly(lambda _src: (10, 10)),
        root_tensors={},
        agent_factory=agent_factory,
        make_verifier=lambda _node, _pc, _t: verifier,
        prompt_inputs=prompt_inputs,
        system_prompts=sys_prompts,
        config=_search_config(1, [None]*1),
    ))
    assert set(result.libraries.keys()) == {only.path}
    assert result.root_path == only.path
    # The root-as-leaf library is keyed on empty contracts (root convention).
    reg = library_to_variant_registry(result.libraries[only.path])
    assert len(reg) == 1


# --- write_library_snapshot --------------------------------------------------


def test_write_library_snapshot_emits_json(tmp_path):
    lib = _sample_lib()
    out = tmp_path / "lib.json"
    write_library_snapshot(lib, out)
    payload = json.loads(out.read_text())
    assert isinstance(payload, list) and len(payload) == 2
    for cell_repr in payload:
        assert "input_contracts" in cell_repr
        assert "output_contracts" in cell_repr
        assert "entries" in cell_repr
