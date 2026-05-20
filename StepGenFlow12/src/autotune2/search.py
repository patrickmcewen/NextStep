"""Bottom-up search driver for autotune2.

Phase 5 of the autotuner. Three entry points:

  - ``search_leaf(node, ...)``   — populate one leaf node's library.
  - ``search_parent(node, ...)`` — populate one parent node's library by
    composing its DSL with each immediate child's already-populated
    library and scoring the Cartesian product.
  - ``autotune(plan_tree, ...)`` — bottom-up driver that walks the plan
    tree post-order and calls the appropriate search routine per node.
    Returns the full ``{node_path: NodeLibrary}`` map.

All external dependencies — the LLM agent, the correctness verifier,
and the analytical scorer — are **injected** as callables. Phase 5
unit tests exercise this module against deterministic mocks; Phase 6
wires the real Anthropic agent + 4-gate verification chain +
``make_analytical_scorer``.

Locked Phase-5 design decisions
-------------------------------
1. ``DesignEntry.dsl`` stores the **standalone** function for that
   node only. The descendant chain is reconstructed at compose time by
   walking ``DesignEntry.children_picks`` (added in this phase) — a
   ``{child_path: chosen_DesignEntry}`` map populated when the entry
   was scored. Direct object references mean Pareto culling at a child
   level cannot orphan a parent's reproducibility chain.

2. **Synthetic ``tiled_reference`` wrapper** for non-root nodes is
   constructed by ``build_synthetic_wrapper_for_node``. For each
   non-RAW ``TensorArg``, it emits an ``offchip_load`` whose stream +
   tile dims exactly match the variant's ``input_contracts[name]``
   (computed by ``_offchip_load_args_for_contract``), then
   ``flatten``-s away ``offchip_load``'s leading singleton so the
   stream rank matches ``applied_shape[:-2]``. RAW ``TensorArg``s
   pass through as ``tensors[name]`` — the leaf is responsible for
   loading those itself. Because strides are contract-dependent the
   wrapper is rebuilt per variant; the search loops pass identity
   contracts for the baseline and ``parsed.input_contracts`` for
   each LLM response. The wrapper terminates with ``offchip_store``
   so the translator's root-return-wrapping logic has a stream to
   wrap.

3. **Per-node ``tensors`` dict** is built by
   ``build_node_tensors_dict``: ``torch.zeros(spec.shape)`` for each
   TensorArg, the parent contract's recorded tile values for list
   args. The timing model only inspects shapes, so zero-filled tensors
   are sufficient.

4. **Variant binding**: ``build_variant_callables`` reads a child's
   ``variants.py`` registry, calls ``make_variant_stub`` per entry,
   and returns ``{f"{child_name}_{variant_index}": callable}`` — the
   shape the search driver injects into the parent's exec namespace
   alongside the existing blackbox stubs.

5. **Variant indices** are assigned sequentially per node, one per
   ``DesignEntry``, when the library is serialized to ``variants.py``.
   Each entry — even two entries that share a Pareto cell's boundary
   contracts — gets its own index, so the parent LLM picks one concrete
   child implementation per child. No Cartesian sweep; ``search_parent``
   composes exactly one descendant chain per turn.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import torch

from src.autotune2.compose import (
    ScoreFn,
    compose_source,
)
from src.autotune2.contracts import (
    ContractsKey,
    DesignEntry,
    NodeLibrary,
    TensorContract,
    freeze_contracts,
    library_cell,
    vanilla_contract_for,
)
from src.autotune2.baseline_selection import select_baselines
from src.autotune2.pareto import insert_pareto
from src.autotune2.persistence import (
    SNAPSHOT_FILENAME,
    save_library_snapshot,
    try_load_library_snapshot,
)
from src.autotune2.prompts import (
    VariantSummary,
    build_autotune2_user_prompt,
    parse_autotune2_response,
    render_accepted_summary,
    render_variant_block,
)
from src.autotune2.stubs import (
    emit_variants_module,
    make_variant_stub,
)
from src.contract import Contract
from src.node_signature import ListOfIntArg, ListOfTensorArg, TensorArg
from src.planner import PlanNode, Tree


# ---------------------------------------------------------------------------
# Injected interfaces
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    passed: bool
    feedback: str = ""
    derived_output_contracts: dict[str, "TensorContract"] = field(default_factory=dict)
    """Output contracts derived from the built graph (one per OffChipStore
    node in declaration order, keyed ``out_0``, ``out_1``, ...). Populated
    on ``passed=True`` for non-root nodes; empty for root (the root verifier
    checks against ``compute_gold`` directly and has no contract to derive)."""


@dataclass
class AgentResponse:
    """Richer agent return value carrying reasoning + token usage.

    An ``AgentFn`` may return either a bare ``str`` (legacy / test
    fixtures) or an ``AgentResponse``. The search loop normalises via
    ``_coerce_agent_response`` so callers don't have to care which form
    came back. When set, ``reasoning`` is logged to ``reasoning.txt`` and
    ``usage`` is fed into ``write_turn_tokens`` to produce
    ``tokens.json`` — both mirror the pass-1 turn-dir layout.
    """

    text: str
    reasoning: str = ""
    usage: object | None = None


# conversation (OpenAI-style list of {role, content}) -> next assistant message
# Returns either the bare assistant text or an ``AgentResponse`` for richer
# per-turn artifacts (reasoning, token usage). Production callers built via
# ``runtime.build_real_agent_fn`` return ``AgentResponse``; test stubs return
# plain ``str``.
AgentFn = Callable[[list[dict]], Awaitable["str | AgentResponse"]]
# composed_source -> VerifyResult. The non-root verifier derives
# ``output_contracts`` from the built graph and surfaces them via
# ``VerifyResult.derived_output_contracts``; the root verifier ignores
# contracts (it checks against ``compute_gold``).
VerifierFn = Callable[[str], Awaitable[VerifyResult]]


def _coerce_agent_response(result) -> AgentResponse:
    """Normalize either a raw str or an AgentResponse into an AgentResponse."""
    if isinstance(result, AgentResponse):
        return result
    assert isinstance(result, str), (
        f"_coerce_agent_response: AgentFn must return str or AgentResponse, "
        f"got {type(result).__name__}"
    )
    return AgentResponse(text=result)


@dataclass
class SearchConfig:
    max_turns_per_attempt: int = 3
    """Max LLM turns within a single attempt before giving up and starting fresh.

    Each turn inside an attempt accumulates assistant responses + gate
    feedback in the same conversation, mirroring pass-1's refactor_final
    loop. Exhausting this budget on an attempt aborts that attempt.
    """

    attempt_budgets_bytes: list[int | None] = field(
        default_factory=lambda: [None]
    )
    """Per-attempt on-chip-memory budgets (bytes). One attempt is fanned
    out in parallel per list element. ``None`` ⇒ unlimited (no prompt
    mention, no over-budget reject). A positive int ⇒ admit a variant
    only if its scored ``on_chip <= budget``; over-budget hits become
    ``OVER_BUDGET`` turn feedback so the LLM can revise.

    The length of this list replaces the legacy ``max_attempts`` knob —
    each entry is one fresh-conversation attempt. Default ``[None]``
    matches the prior single-unlimited-attempt behavior.
    """

    check_order: str = "correctness-first"
    """Gate cascade order, mirrors pass-1's ``check_order`` knob.

    One of ``"correctness-first"``, ``"compliance-first"``, or
    ``"always-both"``. Wired through to the 4-gate cascade
    (correctness, compliance regex, LLM judge, post-validator) inside
    ``build_real_verifier_factory_fn``.
    """

    fewshot: str = "tile_shrink"
    """Which per-agent user-prompt template to use — matches the
    fewshot agent baked into the system prompt by
    ``build_autotune2_system_prompt``. One of ``"tile_shrink"`` or
    ``"parallel"``.
    """


@dataclass(frozen=True)
class NodePromptInputs:
    """All per-node inputs needed to build autotune2 user prompts.

    The autotune2 *system* prompt is per-node but constant across
    attempts; it is baked into the ``AgentFn`` (via ``Agent.instructions``)
    at construction time. The user prompt is rebuilt per attempt so that
    fresh-attempt prompts can include a Pareto-front summary of
    already-accepted variants — see ``render_accepted_summary``.

    Fields:
      function_signature: e.g. ``"def attention_block(Q, K, V, *, out_shapes):"``
      pytorch_reference: gold PyTorch source for this node's semantics
      dims_block: pre-formatted dims description (markdown)
      tensors_block: pre-formatted tensors description (markdown)
    """

    function_signature: str
    pytorch_reference: str
    dims_block: str
    tensors_block: str


# ---------------------------------------------------------------------------
# Synthetic wrapper for non-root nodes
# ---------------------------------------------------------------------------


def _offchip_load_args_for_contract(
    arg_name: str,
    vanilla_shape: tuple[int, ...],
    contract: TensorContract,
) -> tuple[tuple[int, ...], int, int, tuple[int, ...]]:
    """Compute ``(out_shape_tiled, tile_row, tile_col, stride)`` for an
    ``offchip_load`` whose result — once its leading singleton stream dim
    is flattened away — has stream + tile dims exactly equal to the
    contract's ``applied_shape``.

    The off-chip tensor lives row-major in ``vanilla_shape``.
    ``applied_shape = vanilla.reshape(contract.reshape)
    .permute(contract.permutation)`` is the layout the on-chip stream must
    present to the leaf. ``offchip_load`` tiles the underlying tensor by
    ``(tile_row, tile_col)`` (which must divide ``vanilla[-2:]``), walks
    the resulting tile-grid in (batch, r_grid, c_grid) order, and indexes
    that walk by ``stride`` against ``out_shape_tiled``.

    For the on-chip layout to match ``applied_shape`` we require:

      - ``permutation[-2:] == (n-2, n-1)`` — the contract's two innermost
        axes are *its own* two innermost reshape axes, so the tile dims
        come from the tail of ``reshape``. This is what fixes
        ``tile_row = reshape[-2]`` and ``tile_col = reshape[-1]``.
      - Each leading reshape axis's element-major stride decomposes
        cleanly as ``batch_step * R*C + r_step * tile_row*C + c_step *
        tile_col`` with zero remainder — i.e., one unit of that axis
        lands the load on the top-left of a physical tile, never on a
        sub-tile element offset.

    Under those constraints we can compute the tile-flat stride per axis
    in closed form, then permute / select to match the contract's
    permutation. This supports both "tile equals vanilla[-2:]" and
    sub-tiled contracts (e.g. ``reshape = vanilla + (1, 1)`` with a (1,1)
    tile, which is how pass-1 represents fully-streamed leaves).
    """
    assert len(vanilla_shape) >= 2, (
        f"_offchip_load_args_for_contract({arg_name!r}): vanilla_shape "
        f"{vanilla_shape!r} must have rank >= 2"
    )
    R_tuple = contract.reshape
    P = contract.permutation
    n = len(R_tuple)
    assert n >= 2, (
        f"_offchip_load_args_for_contract({arg_name!r}): contract.reshape "
        f"{R_tuple!r} must have rank >= 2 (the tile occupies the last two "
        f"axes)"
    )
    assert tuple(P[-2:]) == (n - 2, n - 1), (
        f"_offchip_load_args_for_contract({arg_name!r}): contract.permutation "
        f"must leave the last two reshape axes in place; got permutation={P!r}. "
        f"Permutations that move the contract's tile dims into the stream "
        f"(or pull stream dims into the tile) aren't realizable as a single "
        f"strided offchip_load — they'd need follow-up DSL ops "
        f"(retile / streamify) to express, which isn't mechanically "
        f"derivable from contract metadata alone."
    )
    R = int(vanilla_shape[-2])
    C = int(vanilla_shape[-1])
    tile_row = int(R_tuple[-2])
    tile_col = int(R_tuple[-1])
    assert R % tile_row == 0, (
        f"_offchip_load_args_for_contract({arg_name!r}): vanilla R={R} not "
        f"divisible by contract tile_row={tile_row}; reshape={R_tuple!r}, "
        f"vanilla_shape={vanilla_shape!r}"
    )
    assert C % tile_col == 0, (
        f"_offchip_load_args_for_contract({arg_name!r}): vanilla C={C} not "
        f"divisible by contract tile_col={tile_col}; reshape={R_tuple!r}, "
        f"vanilla_shape={vanilla_shape!r}"
    )
    Rg = R // tile_row
    Cg = C // tile_col

    m = n - 2
    # offchip_load asserts ``len(out_shape_tiled) >= 1`` — rank-2 applied
    # shapes (m == 0) would need a different load idiom. Surface this here
    # so the failure mode is a parse-time assertion rather than a deep
    # offchip_load shape mismatch later.
    assert m >= 1, (
        f"_offchip_load_args_for_contract({arg_name!r}): contract has no "
        f"leading stream dims (rank-2 post_permute_shape="
        f"{contract.post_permute_shape()!r}); the isolation wrapper currently "
        f"requires at least one stream dim. Reshape the contract to factor "
        f"a leading 1 (e.g. reshape=(1,)+...) if you want a single-tile load."
    )

    # For each leading reshape axis k (0..m-1), decompose its element-major
    # stride E_k into (batch_step, r_step, c_step) over the offchip_load
    # tile grid. The tile-flat stride contributed by axis k is then
    # ``batch_step * Rg * Cg + r_step * Cg + c_step``.
    leading_tile_flat_strides = [0] * m
    for k in range(m):
        E_k = 1
        for j in range(k + 1, n):
            E_k *= int(R_tuple[j])
        batch_step, rem = divmod(E_k, R * C)
        r_step, rem = divmod(rem, tile_row * C)
        c_step, rem = divmod(rem, tile_col)
        assert rem == 0, (
            f"_offchip_load_args_for_contract({arg_name!r}): leading reshape "
            f"axis {k} has element-major stride {E_k} that does not align to "
            f"a tile boundary in vanilla layout (tile=({tile_row}, {tile_col}), "
            f"vanilla[-2:]=({R}, {C})); reshape={R_tuple!r}, vanilla_shape="
            f"{vanilla_shape!r}. The contract's stream axis would land mid-tile, "
            f"which isn't realizable as a single strided offchip_load."
        )
        leading_tile_flat_strides[k] = batch_step * Rg * Cg + r_step * Cg + c_step

    # Apply the permutation: applied stream axis j (0..m-1) corresponds to
    # reshape leading axis P[j].
    stride = tuple(leading_tile_flat_strides[P[j]] for j in range(m))
    out_shape_tiled = tuple(int(R_tuple[P[j]]) for j in range(m))
    return out_shape_tiled, tile_row, tile_col, stride


def build_synthetic_wrapper_for_node(
    *,
    node_name: str,
    parent_contract: Contract,
    input_contracts: dict[str, TensorContract],
) -> str:
    """Build a ``def tiled_reference(dims, tensors): ...`` wrapper that
    calls ``<node_name>`` with kernel-level tensor refs.

    For each non-RAW ``TensorArg``, the wrapper emits an ``offchip_load``
    whose stream + tile dims exactly match ``input_contracts[name]``'s
    ``applied_shape`` (then ``flatten``-s away ``offchip_load``'s leading
    singleton). RAW ``TensorArg``s are passed through as ``tensors[name]``
    — the leaf is responsible for ``offchip_load``-ing those itself
    (``tensors`` arrives wrapped as ``StepRawTensor`` per
    ``_wrap_input_tensors`` in tools.py, so the bare reference satisfies
    every DSL source op's ``_assert_raw`` gate).

    Because the load strides are contract-dependent, this wrapper is
    rebuilt per variant — once with identity contracts to seed the
    baseline, once per parsed LLM response. Pass ``input_contracts =
    {name: vanilla_contract_for(spec.shape) for name, spec in ... if
    on-chip}`` to reproduce the identity-contract baseline behavior.
    """
    arg_names = parent_contract.arg_names
    arg_specs = parent_contract.arg_specs
    arg_is_raw = parent_contract.arg_is_raw
    out_shapes = parent_contract.out_shapes

    on_chip_arg_names = {
        name for name, spec, raw in zip(arg_names, arg_specs, arg_is_raw)
        if isinstance(spec, TensorArg) and not raw
    }
    missing = on_chip_arg_names - set(input_contracts.keys())
    assert not missing, (
        f"build_synthetic_wrapper_for_node({node_name!r}): input_contracts "
        f"is missing entries for on-chip TensorArg(s) {sorted(missing)!r}; "
        f"got contracts for {sorted(input_contracts.keys())!r}"
    )
    extra = set(input_contracts.keys()) - on_chip_arg_names
    assert not extra, (
        f"build_synthetic_wrapper_for_node({node_name!r}): input_contracts "
        f"has entries for non-on-chip args {sorted(extra)!r}; contracts "
        f"only apply to on-chip TensorArg inputs"
    )

    preamble_lines: list[str] = []
    call_arg_exprs: list[str] = []
    for name, spec, raw in zip(arg_names, arg_specs, arg_is_raw):
        if isinstance(spec, TensorArg) and not raw:
            ost, tr, tc, st = _offchip_load_args_for_contract(
                name, tuple(spec.shape), input_contracts[name],
            )
            local = f"_{name}_in"
            preamble_lines.append(
                f'    {local} = offchip_load(tensors["{name}"], '
                f"stride={st!r}, out_shape_tiled={ost!r}, "
                f"tile_row={tr}, tile_col={tc})"
            )
            # offchip_load returns a stream of rank len(ost)+1 (it prepends
            # a leading singleton); flatten the outermost two stream dims
            # so the stream rank matches contract.applied_shape[:-2].
            m = len(ost)
            preamble_lines.append(
                f"    {local} = flatten({local}, "
                f"min_rank={m - 1}, max_rank={m})"
            )
            call_arg_exprs.append(local)
        else:
            call_arg_exprs.append(f'tensors["{name}"]')

    lookups = ", ".join(call_arg_exprs)
    out_shapes_repr = repr(tuple(tuple(s) for s in out_shapes))
    n_outputs = len(out_shapes)
    # Multi-output nodes return a tuple; OffChipStore requires a single
    # Stream input, so we must destructure and emit one store per output.
    # Compliance's "exactly one offchip_store" rule doesn't apply here —
    # this wrapper exists only for analytical scoring + correctness
    # checking of the node in isolation, not as a production root.
    #
    # Each output goes through ``promote_outer`` before ``offchip_store``
    # so OffChipStore's "stream rank >= 1" startup constraint is satisfied
    # (see step-perf/src/memory/offchip_store.rs:99). Some node outputs
    # come back from a fully-collapsing accum with stream rank 0; without
    # the promote the Rust simulator panics on startup.
    if n_outputs == 1:
        body_lines = list(preamble_lines)
        body_lines.append(
            f"    result = {node_name}({lookups}, out_shapes={out_shapes_repr})"
        )
        body_lines.append("    return offchip_store(promote_outer(result))")
    else:
        out_names = [f"_out_{i}" for i in range(n_outputs)]
        destruct = ", ".join(out_names)
        body_lines = list(preamble_lines)
        body_lines.append(
            f"    {destruct} = {node_name}({lookups}, "
            f"out_shapes={out_shapes_repr})"
        )
        # Store every output but the last; return the final store so the
        # tiled_reference has a well-defined return value (mirrors single-
        # output case).
        for nm in out_names[:-1]:
            body_lines.append(f"    offchip_store(promote_outer({nm}))")
        body_lines.append(
            f"    return offchip_store(promote_outer({out_names[-1]}))"
        )
    body = "\n".join(body_lines) + "\n"
    return f"def tiled_reference(dims, tensors):\n{body}"


def _identity_input_contracts(
    parent_contract: Contract,
) -> dict[str, TensorContract]:
    """Pass-1-matching baseline contract per on-chip TensorArg.

    The applied shape is ``parent_contract.tiled_shapes[i]`` — the actual
    on-chip layout the leaf was authored against in pass-1, which may
    differ from ``arg_specs[i].shape`` (the vanilla PyTorch input shape).
    Specifically, pass-1 leaves that operate element-wise on a vanilla
    tensor factor a trailing ``(1, 1)`` tile onto the reshape, producing
    e.g. ``tiled_shape = (4, 4, 64, 32, 1, 1)`` for ``vanilla = (4, 4,
    64, 32)`` so the leaf consumes a per-element stream. ``permutation``
    is identity over the tiled rank, so the wrapper's ``offchip_load``
    walks vanilla element-major into a stream whose layout exactly
    matches what the leaf saw at pass-1 invocation time.
    """
    return {
        name: vanilla_contract_for(tuple(tshape))
        for name, spec, raw, tshape in zip(
            parent_contract.arg_names,
            parent_contract.arg_specs,
            parent_contract.arg_is_raw,
            parent_contract.tiled_shapes,
        )
        if isinstance(spec, TensorArg) and not raw
    }


def build_node_tensors_dict(parent_contract: Contract) -> dict:
    """Build a ``tensors`` dict for scoring this node in isolation.

    For each TensorArg: the recorded ``tiled_values[i]`` reshaped to the
    vanilla ``spec.shape``. Using the real pass-1 value (instead of
    ``torch.zeros``) is required for leaves whose functional executor
    inspects values, not just shapes — e.g. moe_dispatch reads
    ``expert_onehot`` to compute per-expert active-token counts and
    reshapes the resulting ragged buffer accordingly; an all-zero
    one-hot collapses every expert's bucket to size 0 and
    ``FlatReassemble`` panics on the invalid reshape. For RAW tensor
    args ``tiled_value.shape == spec.shape`` (the parent passed through
    the root's vanilla tensor unchanged); for on-chip TensorArgs the
    tiled shape factors differently but numel matches, so the reshape
    is a pure layout-recovery step (mirrors ``_vanillify`` in
    blackbox_stub.py).

    For each list arg: the recorded ``tiled_values`` entry (already a
    ``list[Tensor]`` or ``list[int]``) passes through unchanged.
    """
    out: dict = {}
    for name, spec, val in zip(
        parent_contract.arg_names,
        parent_contract.arg_specs,
        parent_contract.tiled_values,
    ):
        if isinstance(spec, TensorArg):
            assert isinstance(val, torch.Tensor), (
                f"build_node_tensors_dict: arg {name!r} is a TensorArg but "
                f"contract.tiled_values entry is "
                f"{type(val).__name__}"
            )
            out[name] = val.reshape(spec.shape)
        else:
            assert isinstance(spec, (ListOfTensorArg, ListOfIntArg)), (
                f"build_node_tensors_dict: arg {name!r} has unsupported spec "
                f"type {type(spec).__name__}"
            )
            out[name] = val
    return out


# ---------------------------------------------------------------------------
# Descendant gathering
# ---------------------------------------------------------------------------


def gather_descendants_postorder(entry: DesignEntry) -> list[str]:
    """Walk ``entry.children_picks`` recursively; return descendant DSLs
    in post-order (deepest first). Includes the children's own DSL but
    NOT ``entry.dsl`` itself.
    """
    out: list[str] = []
    for child_entry in entry.children_picks.values():
        out.extend(gather_descendants_postorder(child_entry))
        out.append(child_entry.dsl)
    return out


# ---------------------------------------------------------------------------
# Variant binding (registry -> callables for namespace injection)
# ---------------------------------------------------------------------------


def build_variant_callables(
    *,
    child_name: str,
    arg_names: tuple[str, ...],
    arg_specs: tuple,
    output_names: tuple[str, ...],
    ref_module,
    variants: dict[int, dict],
) -> dict[str, Callable]:
    """Build ``{f"{child_name}_{idx}": stub}`` from a variants registry."""
    out: dict[str, Callable] = {}
    for idx, entry in variants.items():
        out[f"{child_name}_{idx}"] = make_variant_stub(
            ref_module=ref_module,
            arg_names=arg_names,
            arg_specs=arg_specs,
            output_names=output_names,
            input_contracts=entry["input_contracts"],
            output_contracts=entry["output_contracts"],
            variant_name=f"{child_name}_{idx}",
        )
    return out


# ---------------------------------------------------------------------------
# Library <-> variant_registry conversion (variant indices)
# ---------------------------------------------------------------------------


def library_to_variant_registry(lib: NodeLibrary) -> dict[int, dict]:
    """Assign sequential variant indices, one per DesignEntry.

    Every Pareto-front entry across every cell gets its own index. Two
    entries sharing boundary contracts (same cell) still get distinct
    indices — the parent LLM picks one concrete child implementation,
    not a cell, so the compose step has a single DSL per child.
    Deterministic in lib iteration order.
    """
    reg: dict[int, dict] = {}
    idx = 0
    for in_key, by_out in lib.items():
        in_contracts = dict(in_key)
        for out_key, cell in by_out.items():
            out_contracts = dict(out_key)
            for _ in cell:
                reg[idx] = {
                    "input_contracts": in_contracts,
                    "output_contracts": out_contracts,
                }
                idx += 1
    return reg


def entry_for_variant(lib: NodeLibrary, variant_index: int) -> DesignEntry:
    """Resolve a variant index to its DesignEntry. Iteration order matches
    ``library_to_variant_registry`` so a registry-derived index round-trips."""
    idx = 0
    for in_key, by_out in lib.items():
        for out_key, cell in by_out.items():
            for entry in cell:
                if idx == variant_index:
                    return entry
                idx += 1
    raise AssertionError(
        f"entry_for_variant: variant_index {variant_index} out of range; "
        f"library has {idx} entries"
    )


def render_library_as_variant_summaries(
    lib: NodeLibrary,
) -> list[VariantSummary]:
    """Build the summary list shown to the LLM. One ``VariantSummary``
    per DesignEntry — every Pareto-front entry appears as its own row so
    the parent agent can pick a specific child implementation rather
    than a boundary-contract cell.
    """
    summaries: list[VariantSummary] = []
    idx = 0
    for in_key, by_out in lib.items():
        in_contracts = dict(in_key)
        for out_key, cell in by_out.items():
            assert cell, (
                "render_library_as_variant_summaries: empty cell at "
                f"({sorted(in_key)!r}, {sorted(out_key)!r}); library "
                "must not contain empty cells"
            )
            for entry in cell:
                summaries.append(VariantSummary(
                    variant_index=idx,
                    input_contracts=in_contracts,
                    output_contracts=dict(out_key),
                    cycles=entry.cycles,
                    on_chip=entry.on_chip,
                ))
                idx += 1
    return summaries


def find_pass1_baseline_entry(lib: NodeLibrary) -> DesignEntry:
    """Locate the unique pass-1 baseline entry in ``lib`` (provenance==
    ``"pass1_baseline"``).

    ``_seed_baseline`` / ``_seed_root_baseline`` insert exactly one such
    entry per library, into the identity-contracts cell, before any LLM
    variant is considered. The parent's baseline composition must use the
    *pass-1* descendant DSLs — not whichever LLM variant happens to be
    Pareto-best in the identity cell — because the pass-1 chain is the
    only one already known to graph-build end-to-end as a tree (pass-1
    proved it). LLM variants may legitimately reshape their stream/tile
    decomposition; reusing one for the parent's baseline can break the
    composition (see HANDOFF.md's contract-conformance section).
    """
    candidates = [
        entry
        for by_out in lib.values() for cell in by_out.values()
        for entry in cell
        if entry.provenance == "pass1_baseline"
    ]
    assert len(candidates) == 1, (
        f"find_pass1_baseline_entry: expected exactly one entry with "
        f"provenance='pass1_baseline' per library, got {len(candidates)}. "
        f"_seed_baseline/_seed_root_baseline must insert the entry before "
        f"any LLM-variant scoring runs."
    )
    return candidates[0]


# ---------------------------------------------------------------------------
# Search drivers
# ---------------------------------------------------------------------------


def _seed_baseline(
    *,
    lib: NodeLibrary,
    node_name: str,
    parent_contract: Contract,
    pass1_dsl: str,
    score_fn: ScoreFn,
    descendant_dsls: list[str],
    children_picks: dict[str, DesignEntry],
) -> tuple[DesignEntry, str]:
    """Insert the pass-1 baseline with identity contracts into the library.

    The synthetic wrapper is rebuilt here with the identity input contracts
    so the per-arg ``offchip_load`` strides match the baseline's vanilla
    layout; identical to what the search loop emits for any LLM variant
    that picks identity contracts on its on-chip inputs.

    Returns ``(entry, baseline_breakdown)``; the breakdown is the
    per-node on-chip memory report from the analytical scorer (or the
    empty string when ``score_fn`` is a test stub without
    ``.breakdown``). Callers thread the breakdown into the budgeted
    user prompt via ``_budget_block``.
    """
    identity_in = _identity_input_contracts(parent_contract)
    output_names = tuple(f"out_{i}" for i in range(len(parent_contract.out_shapes)))
    identity_out = {
        name: vanilla_contract_for(tuple(parent_contract.out_shapes[i]))
        for i, name in enumerate(output_names)
    }
    wrapper = build_synthetic_wrapper_for_node(
        node_name=node_name,
        parent_contract=parent_contract,
        input_contracts=identity_in,
    )
    composed = compose_source(
        parent_dsl=wrapper + "\n" + pass1_dsl,
        descendant_dsls_postorder=descendant_dsls,
    )
    cycles, on_chip = score_fn(composed)
    breakdown = _maybe_breakdown(score_fn, composed)
    entry = DesignEntry(
        dsl=pass1_dsl,
        input_contracts=identity_in,
        output_contracts=identity_out,
        cycles=cycles,
        on_chip=on_chip,
        provenance="pass1_baseline",
        breakdown=breakdown,
        children_picks=dict(children_picks),
    )
    cell = library_cell(lib, identity_in, identity_out)
    cell.append(entry)
    return entry, breakdown


def _on_chip_vanilla_shapes(parent_contract: Contract) -> dict[str, tuple[int, ...]]:
    """Map ``{on_chip_arg_name: vanilla_shape}`` from a parent contract.

    RAW args and non-tensor args are excluded — contracts don't apply to
    them, so they wouldn't appear in any ``input_contracts`` rendering.
    """
    return {
        name: tuple(spec.shape)
        for name, spec, raw in zip(
            parent_contract.arg_names, parent_contract.arg_specs,
            parent_contract.arg_is_raw)
        if isinstance(spec, TensorArg) and not raw
    }


def _output_vanilla_shapes(parent_contract: Contract) -> dict[str, tuple[int, ...]]:
    """Map ``{out_<i>: vanilla_shape}`` from a parent contract's out_shapes."""
    return {
        f"out_{i}": tuple(parent_contract.out_shapes[i])
        for i in range(len(parent_contract.out_shapes))
    }


def _append_turn_feedback(conversation: list[dict], feedback: str) -> None:
    """Append a user-role feedback message to the conversation in place."""
    conversation.append({"role": "user", "content": feedback})


def _admission_continuation_feedback(
    entry: DesignEntry, budget: int | None,
) -> str:
    """Feedback appended after a variant is admitted, asking the agent to
    keep proposing better designs within the same attempt.

    The conversation stays open after each admission so the model can
    build on what it just produced and iterate toward lower cycles
    without burning a fresh attempt. Successive admissions all land in
    the same per-node library via ``insert_pareto``.
    """
    msg = (
        f"Variant ACCEPTED at cycles={entry.cycles}, on_chip={entry.on_chip} "
        f"bytes. It has been admitted to this node's library.\n\n"
        "Now propose another DSL implementation that achieves **lower "
        "cycles** than the variant above"
    )
    if budget is not None:
        msg += f" while still keeping on_chip <= {int(budget)} bytes"
    msg += (
        ". The new variant must remain functionally equivalent to the "
        "PyTorch reference and must differ structurally from every "
        "variant already accepted in this conversation (and from the "
        "pass-1 baseline). Emit your YAML + python blocks per the "
        "output protocol."
    )
    if entry.breakdown:
        msg += (
            "\n\nPer-node on-chip memory of the just-accepted variant "
            "(largest contributors first) — use this to find where "
            "cycles might still be improved without busting the budget:\n\n"
            f"```\n{entry.breakdown}\n```"
        )
    return msg


def _safe_score(
    score_fn: ScoreFn, composed: str,
) -> tuple[int | None, int | None, str | None]:
    """Run ``score_fn(composed)`` and convert any exception to LLM feedback.

    The analytical scorer goes ``translate → _exec_build_graph →
    analyze_timing``; ``analyze_timing`` in turn invokes the timing
    model's functional executor (``execute_values``), which can raise
    on shape regimes the executor doesn't handle uniformly. The
    non-root verifier's DSL-exec + graph-build smoke tests catch most
    of these upstream (DSL eager-exec catches torch-level shape
    mismatches that match the timing model's executor; graph-build
    catches STeP frontend assertions). This helper is the last-resort
    net for anything that still slips through — timing-model internals
    or executor paths the DSL surface doesn't reach — so the search
    loop converts them into LLM next-turn feedback instead of crashing
    the autotune2 run.

    Returns ``(cycles, on_chip, None)`` on success or ``(None, None,
    feedback_str)`` on failure — call site picks one tuple shape and
    branches.
    """
    import traceback as _tb
    try:
        cycles, on_chip = score_fn(composed)
        return cycles, on_chip, None
    except Exception:
        err = _tb.format_exc()
        feedback = (
            "## Analytical scorer failed on this variant\n\n"
            "The DSL translated and the STeP graph built, but the timing "
            "model's analyzer raised while propagating concrete values "
            "through the graph. This is usually a shape regime the "
            "functional executor can't handle uniformly (e.g. an op like "
            "`flat_reassemble` / `flat_partition` whose per-token or "
            "per-expert buckets must be equal-sized for the executor's "
            "internal stack-and-reshape path). Error follows:\n\n"
            "```\n" + err + "```\n\n"
            "Consider an alternative implementation that avoids the "
            "failing op pattern, or adjust your contracts so the "
            "downstream stream shapes are uniform across the dynamic "
            "axes the failing op spans."
        )
        return None, None, feedback


def _write_pass1_baseline_score(ckpt_dir: Path, baseline: DesignEntry) -> None:
    """Persist the pass-1 baseline (cycles, on_chip) to the node's pass-2
    ckpt_dir so the reference point is visible alongside the attempts
    rather than only readable indirectly from the next attempt's prompt.

    For parent nodes the recorded score reflects the composition of this
    node's pass-1 DSL with each child's pass-1 baseline — matching what
    ``_seed_baseline`` actually scored — so the number is directly
    comparable to the cycles/on_chip recorded for each accepted variant
    that uses different child picks.
    """
    import json
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "pass1_baseline_score.json").write_text(json.dumps({
        "cycles": baseline.cycles,
        "on_chip": baseline.on_chip,
        "provenance": baseline.provenance,
    }, indent=2))


def _write_turn_artifacts(
    turn_dir: Path,
    *,
    user_prompt: str,
    agent_response: AgentResponse,
    status: str,
    extracted_code: str | None = None,
    composed_source: str | None = None,
    verify_result: VerifyResult | None = None,
    admitted_entries: list[DesignEntry] | None = None,
) -> None:
    """Per-turn checkpoint. Mirrors pass-1's turn_<N>/ layout so the same
    inspection tooling works across passes.

    Always written: ``user_prompt.txt`` (the latest user message before the
    LLM call), ``response.txt``, ``status.txt``. Conditionally written:

      - ``reasoning.txt`` when the agent returned reasoning summaries
        (only emitted by reasoning-capable models).
      - ``tokens.json`` when the agent returned a usage object — funneled
        through ``write_turn_tokens`` for consistency with pass-1.
      - ``extracted_code.py`` when YAML/python parsing succeeded.
      - ``composed_source.py`` when a composed source was scored or
        verified (full parent + descendant DSL the verifier saw).
      - ``verify_result.txt`` when the verifier was invoked
        (``"PASS"`` on success, the full gate feedback on failure).
      - ``score.json`` when one or more entries were admitted on this
        turn — records each admitted entry's (cycles, on_chip) so the
        turn's performance is visible without inspecting the next
        attempt's user prompt.
    """
    from src.token_accounting import write_turn_tokens

    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "user_prompt.txt").write_text(user_prompt)
    (turn_dir / "response.txt").write_text(agent_response.text)
    (turn_dir / "status.txt").write_text(status)
    if agent_response.reasoning:
        (turn_dir / "reasoning.txt").write_text(agent_response.reasoning)
    if agent_response.usage is not None:
        write_turn_tokens(turn_dir, agent_response.usage, kind="main")
    if extracted_code is not None:
        (turn_dir / "extracted_code.py").write_text(extracted_code)
    if composed_source is not None:
        (turn_dir / "composed_source.py").write_text(composed_source)
    if verify_result is not None:
        (turn_dir / "verify_result.txt").write_text(
            "PASS" if verify_result.passed else verify_result.feedback
        )
    if admitted_entries:
        import json
        (turn_dir / "score.json").write_text(json.dumps({
            "entries": [
                {
                    "cycles": e.cycles,
                    "on_chip": e.on_chip,
                    "provenance": e.provenance,
                }
                for e in admitted_entries
            ],
        }, indent=2))


def _budget_label(budget: int | None) -> str:
    """Filesystem/provenance-safe label for a per-attempt budget.

    ``None`` → ``"inf"``; positive ints → their decimal form. Used in
    ``attempt_<i>_b<label>/`` dir names and entry provenance strings so
    each parallel attempt's artifacts are unambiguous on disk.
    """
    return "inf" if budget is None else str(int(budget))


def _budget_block(budget: int | None, baseline_breakdown: str = "") -> str:
    """User-prompt block describing this attempt's on-chip budget.

    Empty string for unlimited so the prompt looks identical to a
    no-budget run; a budgeted attempt gets an explicit ``## On-chip
    memory budget`` section the LLM is expected to honor. When
    ``baseline_breakdown`` is non-empty (analytical scorer wired
    through), append the pass-1 baseline's per-node on-chip memory
    breakdown so the LLM can see where the biggest tiles live before
    proposing a variant.
    """
    if budget is None:
        return ""
    block = (
        "\n\n### On-chip memory budget\n\n"
        f"This attempt enforces an on-chip memory budget of "
        f"**{int(budget)} bytes**. Variants whose scored on_chip exceeds "
        "this budget are rejected. Your goal should be to minimize the latency of the program"
        "while staying within the memory budget. The equations used to calculate memory for each operation were given in the system prompt.\n"
    )
    if baseline_breakdown:
        block += (
            "\nPer-node on-chip memory of the pass-1 baseline (largest "
            "contributors first) — use this to target the biggest tiles "
            "when restructuring:\n\n"
            f"```\n{baseline_breakdown}\n```\n"
        )
    return block


def _maybe_breakdown(score_fn: ScoreFn, composed: str) -> str:
    """Per-node memory report for ``composed`` if ``score_fn`` provides one.

    ``make_analytical_scorer`` attaches ``score.breakdown``; test stubs
    (``lambda _src: (10, 10)``) don't. Callers that want the report
    funnel through here and degrade gracefully on stubs.
    """
    breakdown_fn = getattr(score_fn, "breakdown", None)
    if breakdown_fn is None:
        return ""
    return breakdown_fn(composed)


def _compliance_preflight_feedback(dsl: str, *, is_root: bool) -> str | None:
    """Run pass-1's refactor_final regex compliance over ``dsl``; return
    LLM-actionable feedback when violations are found, ``None`` otherwise.

    Mirrors what the verifier's ``_gate_compliance`` does, but runs
    before ``build_synthetic_wrapper_for_node`` / the smoke tests so the
    banned-op patterns (``.underlying_tensor``, ``.unsqueeze(``, ...)
    surface as concrete, named feedback even when a downstream check
    (graph build, wrapper build) would otherwise fail first with a
    cryptic error like ``'Flatten' object has no attribute
    'underlying_tensor'``. Short-circuits the loop so the LLM iterates
    on the actual cause rather than the symptom.
    """
    from src.orchestrator import _check_banned_ops

    violations = _check_banned_ops(dsl, "refactor_final", is_root=is_root)
    if not violations:
        return None
    return (
        "## Compliance check FAILED\n\n"
        "Your code uses disallowed operations:\n\n"
        + "\n".join(violations)
        + "\n\nReplace these with the corresponding DSL function calls "
        "listed in the instructions."
    )


async def _run_leaf_attempt(
    *,
    node: PlanNode,
    parent_contract: Contract | None,
    baseline_dsl: str,
    baseline_index: int,
    attempt_index: int,
    budget: int | None,
    attempt_dir: Path,
    score_fn: ScoreFn,
    agent: AgentFn,
    verifier: VerifierFn,
    prompt_inputs: NodePromptInputs,
    config: SearchConfig,
    baseline_accepted: list[DesignEntry],
    baseline_breakdown: str = "",
) -> list[DesignEntry]:
    """Run one leaf attempt (one fresh conversation, up to
    ``max_turns_per_attempt`` turns) under the given on-chip budget.

    Returns the list of DesignEntries admitted on this attempt (may be
    empty). The caller merges them into the shared per-node library via
    ``insert_pareto`` so the cross-attempt Pareto admission is
    consistent.

    ``baseline_dsl`` is the DSL shown to the LLM as its starting design.
    Single-pass autotune2 always passes the pass-1 baseline; multi-pass
    branching may supply any prior library entry's DSL.
    ``baseline_index`` identifies which branch this attempt belongs to —
    0 is the always-seeded pass-1 baseline; 1+ are caller-supplied
    branches. It is embedded in the admitted entries' ``provenance``
    string so the originating branch is recoverable from the library.

    ``budget`` is in bytes; ``None`` disables both the prompt budget
    block and the over-budget reject path (effectively the same control
    flow as the prior single-attempt behavior).
    """
    is_root = parent_contract is None
    arg_vanilla_shapes = (
        {} if is_root else _on_chip_vanilla_shapes(parent_contract)
    )
    output_vanilla_shapes = (
        {} if is_root else _output_vanilla_shapes(parent_contract)
    )
    accepted_summary = render_accepted_summary(
        baseline_accepted,
        arg_vanilla_shapes=arg_vanilla_shapes,
        output_vanilla_shapes=output_vanilla_shapes,
    )
    user_prompt = build_autotune2_user_prompt(
        is_leaf=True,
        node_name=node.name,
        function_signature=prompt_inputs.function_signature,
        pytorch_reference=prompt_inputs.pytorch_reference,
        baseline_dsl=baseline_dsl,
        dims_block=prompt_inputs.dims_block,
        tensors_block=prompt_inputs.tensors_block,
        fewshot=config.fewshot,
        accepted_summary=accepted_summary,
        budget_block=_budget_block(budget, baseline_breakdown),
    )
    conversation: list[dict] = [{"role": "user", "content": user_prompt}]
    blabel = _budget_label(budget)
    admitted: list[DesignEntry] = []

    for turn in range(config.max_turns_per_attempt):
        assert conversation[-1]["role"] == "user", (
            "search_leaf: expected last conversation message to be a user "
            "turn before invoking the agent"
        )
        turn_user_prompt = conversation[-1]["content"]
        agent_response = _coerce_agent_response(await agent(conversation))
        response = agent_response.text
        conversation.append({"role": "assistant", "content": response})
        turn_dir = attempt_dir / f"turn_{turn}"

        try:
            parsed = parse_autotune2_response(response, is_leaf=True)
        except AssertionError as e:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status=f"PARSE_FAIL: {e}",
            )
            _append_turn_feedback(conversation, (
                f"Your response could not be parsed: {e}\n\n"
                "Please re-emit the YAML and python blocks exactly per "
                "the output protocol described in the system prompt."
            ))
            continue

        compliance_feedback = _compliance_preflight_feedback(
            parsed.dsl, is_root=is_root,
        )
        if compliance_feedback is not None:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="COMPLIANCE_FAIL",
                extracted_code=parsed.dsl,
            )
            _append_turn_feedback(conversation, compliance_feedback)
            continue

        try:
            wrapper = "" if is_root else build_synthetic_wrapper_for_node(
                node_name=node.name,
                parent_contract=parent_contract,
                input_contracts=parsed.input_contracts,
            )
        except AssertionError as e:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status=f"WRAPPER_BUILD_FAIL: {e}",
                extracted_code=parsed.dsl,
            )
            _append_turn_feedback(conversation, (
                "Your declared input_contracts are not realizable as a "
                f"single strided offchip_load: {e}\n\n"
                "Pick contracts that keep the tile = vanilla[-2:] and "
                "leave the last two reshape axes in place; the leading "
                "stream axes may be factored/permuted freely."
            ))
            continue

        composed = compose_source(
            parent_dsl=wrapper + ("\n" if wrapper else "") + parsed.dsl,
            descendant_dsls_postorder=[],
        )
        verify = await verifier(composed)
        if not verify.passed:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="VERIFY_FAIL",
                extracted_code=parsed.dsl,
                composed_source=composed,
                verify_result=verify,
            )
            _append_turn_feedback(conversation, (
                "Your variant did not pass verification. Gate feedback "
                "follows; please emit a corrected DSL implementation "
                "that addresses the issues:\n\n"
                f"{verify.feedback}"
            ))
            continue

        cycles, on_chip, score_err = _safe_score(score_fn, composed)
        if score_err is not None:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="SCORE_FAIL",
                extracted_code=parsed.dsl,
                composed_source=composed,
                verify_result=VerifyResult(passed=False, feedback=score_err),
            )
            _append_turn_feedback(conversation, score_err)
            continue

        if budget is not None and on_chip > budget:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status=f"OVER_BUDGET (on_chip={on_chip} > {budget})",
                extracted_code=parsed.dsl,
                composed_source=composed,
                verify_result=verify,
            )
            over_budget_breakdown = _maybe_breakdown(score_fn, composed)
            feedback = (
                f"Your variant verified and scored at on_chip={on_chip} "
                f"bytes, which exceeds this attempt's on-chip budget of "
                f"{budget} bytes. Emit a corrected DSL implementation "
                "that reduces on-chip memory below the budget."
            )
            if over_budget_breakdown:
                feedback += (
                    "\n\nPer-node on-chip memory of the rejected variant "
                    "(largest contributors first):\n\n"
                    f"```\n{over_budget_breakdown}\n```"
                )
            _append_turn_feedback(conversation, feedback)
            continue

        entry = DesignEntry(
            dsl=parsed.dsl,
            input_contracts=parsed.input_contracts,
            output_contracts=verify.derived_output_contracts,
            cycles=cycles,
            on_chip=on_chip,
            provenance=(
                f"llm_baseline_{baseline_index}_attempt_{attempt_index}"
                f"_b{blabel}_turn_{turn}"
            ),
            breakdown=_maybe_breakdown(score_fn, composed),
        )
        admitted.append(entry)
        _write_turn_artifacts(
            turn_dir,
            user_prompt=turn_user_prompt,
            agent_response=agent_response,
            status="ACCEPTED",
            extracted_code=parsed.dsl,
            composed_source=composed,
            verify_result=verify,
            admitted_entries=[entry],
        )
        # Don't break — keep the conversation open so the agent can chase
        # further improvements within the same attempt. Every admitted
        # entry is later merged into the shared library via ``insert_pareto``.
        _append_turn_feedback(
            conversation,
            _admission_continuation_feedback(entry, budget),
        )

    return admitted


async def search_leaf(
    *,
    node: PlanNode,
    parent_contract: Contract | None,
    pass1_dsl: str,
    ckpt_dir: Path,
    score_fn: ScoreFn,
    agent: AgentFn,
    verifier: VerifierFn,
    prompt_inputs: NodePromptInputs,
    config: SearchConfig = SearchConfig(),
    system_prompt: str = "",
    initial_baselines: list[DesignEntry] | None = None,
) -> NodeLibrary:
    """Populate one leaf node's library.

    Seeds with the pass-1 baseline (identity contracts) — always
    baseline index 0 — and then runs one fresh-conversation attempt per
    ``(baseline, budget)`` pair via ``asyncio.gather``. The default
    ``initial_baselines=None`` matches single-pass autotune2: a single
    branch of attempts off the pass-1 baseline. Supplying additional
    branches (e.g. tiling variants from a prior pass) spawns one
    attempt-fanout per branch, each conversation seeded with the
    branch's own DSL and breakdown. Branches are isolated — each
    conversation only sees its own starting design.

    Each attempt budget is rendered into its own user prompt; variants
    whose scored ``on_chip`` exceeds the budget are rejected with
    ``OVER_BUDGET`` turn feedback. ``budget == None`` skips both the
    prompt mention and the reject filter.

    After all attempts complete, every admitted DesignEntry is merged
    into the shared library via ``insert_pareto`` so the cross-branch
    Pareto admissions remain consistent across the fan-out.

    ``parent_contract`` is ``None`` only for the root-as-leaf case
    (single-node plan tree): the pass-1 DSL is already
    ``tiled_reference(dims, tensors)`` so no synthetic wrapper is
    needed and the library uses empty identity contracts.
    """
    assert node.is_leaf, f"search_leaf called on non-leaf node {node.path!r}"
    assert config.attempt_budgets_bytes, (
        "search_leaf: SearchConfig.attempt_budgets_bytes must contain at "
        "least one entry (use [None] for a single unlimited attempt)"
    )
    lib: NodeLibrary = {}
    is_root = parent_contract is None

    if system_prompt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "system_prompt.txt").write_text(system_prompt)

    if is_root:
        baseline, baseline_breakdown = _seed_root_baseline(
            lib=lib, node_name=node.name, pass1_dsl=pass1_dsl,
            score_fn=score_fn, descendant_dsls=[], children_picks={},
        )
    else:
        baseline, baseline_breakdown = _seed_baseline(
            lib=lib, node_name=node.name, parent_contract=parent_contract,
            pass1_dsl=pass1_dsl, score_fn=score_fn,
            descendant_dsls=[],
            children_picks={},
        )
    _write_pass1_baseline_score(ckpt_dir, baseline)

    baselines: list[DesignEntry] = [baseline, *(initial_baselines or [])]
    attempt_coros = [
        _run_leaf_attempt(
            node=node,
            parent_contract=parent_contract,
            baseline_dsl=b.dsl,
            baseline_index=b_idx,
            attempt_index=a_idx,
            budget=budget,
            attempt_dir=(
                ckpt_dir
                / f"baseline_{b_idx}_attempt_{a_idx}_b{_budget_label(budget)}"
            ),
            score_fn=score_fn,
            agent=agent,
            verifier=verifier,
            prompt_inputs=prompt_inputs,
            config=config,
            baseline_accepted=[b],
            baseline_breakdown=b.breakdown,
        )
        for b_idx, b in enumerate(baselines)
        for a_idx, budget in enumerate(config.attempt_budgets_bytes)
    ]
    per_attempt = await asyncio.gather(*attempt_coros)
    for admitted in per_attempt:
        for entry in admitted:
            cell = library_cell(
                lib, entry.input_contracts, entry.output_contracts)
            insert_pareto(cell, entry)

    # Persist registry artifact for downstream variant binding.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emit_variants_module(
        out_path=ckpt_dir / "variants.py",
        child_name=node.name,
        variants=library_to_variant_registry(lib),
    )
    return lib


async def _run_parent_attempt(
    *,
    node: PlanNode,
    parent_contract: Contract | None,
    baseline_dsl: str,
    baseline_index: int,
    children_libraries: dict[str, NodeLibrary],
    child_blocks: dict[str, str],
    expected_child_names: tuple[str, ...],
    attempt_index: int,
    budget: int | None,
    attempt_dir: Path,
    score_fn: ScoreFn,
    agent: AgentFn,
    verifier: VerifierFn,
    prompt_inputs: NodePromptInputs,
    config: SearchConfig,
    baseline_accepted: list[DesignEntry],
    baseline_breakdown: str = "",
) -> list[DesignEntry]:
    """One parent attempt — fresh conversation, up to
    ``max_turns_per_attempt`` turns, scoped to a single on-chip-memory
    budget. Returns admitted DesignEntries (merged into the shared lib
    by the caller).

    Children's full Pareto fronts are exposed to the LLM via
    ``child_blocks``; the agent autonomously picks one variant_index
    per child (a single ``DesignEntry``), and the parent DSL is
    expected to add intermediate STeP ops if the picked children's
    contracts don't compose cleanly. No Cartesian sweep.

    ``baseline_dsl`` is the parent DSL shown to the LLM as its starting
    design. Single-pass autotune2 always passes the pass-1 baseline;
    multi-pass branching may supply any prior library entry's DSL.
    ``baseline_index`` identifies which branch this attempt belongs to —
    0 is the pass-1 baseline; 1+ are caller-supplied branches. Embedded
    in admitted entries' ``provenance`` for branch traceability.
    """
    is_root = parent_contract is None
    arg_vanilla_shapes = (
        {} if is_root else _on_chip_vanilla_shapes(parent_contract)
    )
    output_vanilla_shapes = (
        {} if is_root else _output_vanilla_shapes(parent_contract)
    )
    accepted_summary = render_accepted_summary(
        baseline_accepted,
        arg_vanilla_shapes=arg_vanilla_shapes,
        output_vanilla_shapes=output_vanilla_shapes,
    )
    user_prompt = build_autotune2_user_prompt(
        is_leaf=False,
        node_name=node.name,
        function_signature=prompt_inputs.function_signature,
        pytorch_reference=prompt_inputs.pytorch_reference,
        baseline_dsl=baseline_dsl,
        dims_block=prompt_inputs.dims_block,
        tensors_block=prompt_inputs.tensors_block,
        fewshot=config.fewshot,
        child_variant_blocks=child_blocks,
        accepted_summary=accepted_summary,
        budget_block=_budget_block(budget, baseline_breakdown),
    )
    conversation: list[dict] = [{"role": "user", "content": user_prompt}]
    blabel = _budget_label(budget)
    admitted: list[DesignEntry] = []

    for turn in range(config.max_turns_per_attempt):
        assert conversation[-1]["role"] == "user", (
            "search_parent: expected last conversation message to be a "
            "user turn before invoking the agent"
        )
        turn_user_prompt = conversation[-1]["content"]
        agent_response = _coerce_agent_response(await agent(conversation))
        response = agent_response.text
        conversation.append({"role": "assistant", "content": response})
        turn_dir = attempt_dir / f"turn_{turn}"

        try:
            parsed = parse_autotune2_response(
                response, is_leaf=False,
                expected_child_names=expected_child_names,
            )
        except AssertionError as e:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status=f"PARSE_FAIL: {e}",
            )
            _append_turn_feedback(conversation, (
                f"Your response could not be parsed: {e}\n\n"
                "Please re-emit the YAML and python blocks exactly per "
                "the output protocol described in the system prompt."
            ))
            continue

        try:
            children_picks: dict[str, DesignEntry] = {
                child.path: entry_for_variant(
                    children_libraries[child.path],
                    parsed.child_picks[child.name],
                )
                for child in node.children
            }
        except AssertionError as e:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status=f"BAD_CHILD_PICK: {e}",
                extracted_code=parsed.dsl,
            )
            _append_turn_feedback(conversation, (
                f"One of your child_picks indices is invalid: {e}\n\n"
                "Refer to the child variant tables in the user prompt "
                "and pick a valid variant_index per child."
            ))
            continue

        compliance_feedback = _compliance_preflight_feedback(
            parsed.dsl, is_root=is_root,
        )
        if compliance_feedback is not None:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="COMPLIANCE_FAIL",
                extracted_code=parsed.dsl,
            )
            _append_turn_feedback(conversation, compliance_feedback)
            continue

        try:
            wrapper = "" if is_root else build_synthetic_wrapper_for_node(
                node_name=node.name,
                parent_contract=parent_contract,
                input_contracts=parsed.input_contracts,
            )
        except AssertionError as e:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status=f"WRAPPER_BUILD_FAIL: {e}",
                extracted_code=parsed.dsl,
            )
            _append_turn_feedback(conversation, (
                "Your declared parent_input_contracts are not realizable "
                f"as a single strided offchip_load: {e}\n\n"
                "Pick contracts that keep the tile = vanilla[-2:] and "
                "leave the last two reshape axes in place; the leading "
                "stream axes may be factored/permuted freely."
            ))
            continue

        descendants: list[str] = []
        for child in node.children:
            child_entry = children_picks[child.path]
            descendants.extend(gather_descendants_postorder(child_entry))
            descendants.append(child_entry.dsl)
        composed = compose_source(
            parent_dsl=wrapper + ("\n" if wrapper else "") + parsed.dsl,
            descendant_dsls_postorder=descendants,
        )
        verify = await verifier(composed)
        if not verify.passed:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="VERIFY_FAIL",
                extracted_code=parsed.dsl,
                composed_source=composed,
                verify_result=verify,
            )
            _append_turn_feedback(conversation, (
                "Your variant did not pass verification. Gate feedback "
                "follows; please emit a corrected DSL implementation "
                "that addresses the issues:\n\n"
                f"{verify.feedback}"
            ))
            continue
        cycles, on_chip, score_err = _safe_score(score_fn, composed)
        if score_err is not None:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="SCORE_FAIL",
                extracted_code=parsed.dsl,
                composed_source=composed,
                verify_result=VerifyResult(passed=False, feedback=score_err),
            )
            _append_turn_feedback(conversation, score_err)
            continue

        if budget is not None and on_chip > budget:
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status=f"OVER_BUDGET (on_chip={on_chip} > {budget})",
                extracted_code=parsed.dsl,
                composed_source=composed,
                verify_result=verify,
            )
            over_budget_breakdown = _maybe_breakdown(score_fn, composed)
            feedback = (
                f"Your composed variant scored at on_chip={on_chip} "
                f"bytes, which exceeds this attempt's on-chip budget of "
                f"{budget} bytes. Pick different child variants and/or "
                "restructure the parent DSL to reduce on-chip memory "
                "below the budget."
            )
            if over_budget_breakdown:
                feedback += (
                    "\n\nPer-node on-chip memory of the rejected "
                    "composition (largest contributors first):\n\n"
                    f"```\n{over_budget_breakdown}\n```"
                )
            _append_turn_feedback(conversation, feedback)
            continue

        entry = DesignEntry(
            dsl=parsed.dsl,
            input_contracts=parsed.input_contracts,
            output_contracts=verify.derived_output_contracts,
            cycles=cycles,
            on_chip=on_chip,
            provenance=(
                f"llm_baseline_{baseline_index}_attempt_{attempt_index}"
                f"_b{blabel}_turn_{turn}"
            ),
            breakdown=_maybe_breakdown(score_fn, composed),
            children_picks=children_picks,
        )
        admitted.append(entry)
        _write_turn_artifacts(
            turn_dir,
            user_prompt=turn_user_prompt,
            agent_response=agent_response,
            status="ACCEPTED",
            extracted_code=parsed.dsl,
            composed_source=composed,
            verify_result=verify,
            admitted_entries=[entry],
        )
        # Don't break — keep the conversation open so the agent can chase
        # further improvements within the same attempt. Every admitted
        # entry is later merged into the shared library via ``insert_pareto``.
        _append_turn_feedback(
            conversation,
            _admission_continuation_feedback(entry, budget),
        )

    return admitted


async def search_parent(
    *,
    node: PlanNode,
    parent_contract: Contract | None,
    pass1_dsl: str,
    children_libraries: dict[str, NodeLibrary],
    children_picks_baseline: dict[str, DesignEntry],
    ckpt_dir: Path,
    score_fn: ScoreFn,
    agent: AgentFn,
    verifier: VerifierFn,
    prompt_inputs: NodePromptInputs,
    config: SearchConfig = SearchConfig(),
    system_prompt: str = "",
    initial_baselines: list[DesignEntry] | None = None,
) -> NodeLibrary:
    """Populate one parent node's library.

    ``children_libraries`` provides each immediate child's already-
    populated library (keyed by child_path). ``children_picks_baseline``
    provides the pass-1-baseline pick per child (used to score the
    parent's pass-1 baseline against the children's pass-1 baselines).

    Fans out one fresh-conversation attempt per ``(baseline, budget)``
    pair (all in parallel via ``asyncio.gather``). The default
    ``initial_baselines=None`` matches single-pass autotune2: attempts
    branch only off the pass-1 baseline. Supplying additional branches
    (e.g. tiling variants from a prior pass) spawns one attempt-fanout
    per branch; each branch's conversation only sees its own starting
    design. The branch's existing ``children_picks`` are reused as the
    composition for the prompt; ``children_picks_baseline`` is still
    used for the pass-1 (index-0) seed and for the parent's vanilla-
    shape variant tables, which are stable across branches.

    Each attempt picks one specific child variant per child and composes
    a single descendant chain — no Cartesian sweep. Variants whose
    scored composition exceeds the attempt's budget are rejected with
    ``OVER_BUDGET`` turn feedback; ``None`` skips both the prompt
    mention and the reject filter.
    """
    assert config.attempt_budgets_bytes, (
        "search_parent: SearchConfig.attempt_budgets_bytes must contain at "
        "least one entry (use [None] for a single unlimited attempt)"
    )
    lib: NodeLibrary = {}
    is_root = parent_contract is None

    if system_prompt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "system_prompt.txt").write_text(system_prompt)

    # Baseline: parent's pass-1 DSL composed with each child's pass-1 entry.
    baseline_descendants: list[str] = []
    for child_entry in children_picks_baseline.values():
        baseline_descendants.extend(gather_descendants_postorder(child_entry))
        baseline_descendants.append(child_entry.dsl)
    if is_root:
        baseline, baseline_breakdown = _seed_root_baseline(
            lib=lib, node_name=node.name, pass1_dsl=pass1_dsl,
            score_fn=score_fn, descendant_dsls=baseline_descendants,
            children_picks=children_picks_baseline,
        )
    else:
        baseline, baseline_breakdown = _seed_baseline(
            lib=lib, node_name=node.name, parent_contract=parent_contract,
            pass1_dsl=pass1_dsl, score_fn=score_fn,
            descendant_dsls=baseline_descendants,
            children_picks=children_picks_baseline,
        )
    _write_pass1_baseline_score(ckpt_dir, baseline)

    # Build per-child variant tables for the prompt (stable across attempts —
    # children's libraries don't change while this parent is searching).
    child_blocks: dict[str, str] = {}
    for child in node.children:
        child_lib = children_libraries[child.path]
        summaries = render_library_as_variant_summaries(child_lib)
        # vanilla shapes + RAW classification for the variant table header.
        # NOTE: pulled from the child's pass1 contract recorded against THIS
        # parent's call site. The driver passes that via children_picks_baseline.
        baseline_entry = children_picks_baseline[child.path]
        arg_van_shapes = {
            name: c.reshape
            for name, c in baseline_entry.input_contracts.items()
        }
        out_van_shapes = {
            name: c.reshape
            for name, c in baseline_entry.output_contracts.items()
        }
        arg_is_raw_map = {name: False for name in arg_van_shapes}
        child_blocks[child.name] = render_variant_block(
            child_name=child.name,
            arg_vanilla_shapes=arg_van_shapes,
            arg_is_raw=arg_is_raw_map,
            output_vanilla_shapes=out_van_shapes,
            variants=summaries,
        )

    expected_child_names = tuple(c.name for c in node.children)
    baselines: list[DesignEntry] = [baseline, *(initial_baselines or [])]

    attempt_coros = [
        _run_parent_attempt(
            node=node,
            parent_contract=parent_contract,
            baseline_dsl=b.dsl,
            baseline_index=b_idx,
            children_libraries=children_libraries,
            child_blocks=child_blocks,
            expected_child_names=expected_child_names,
            attempt_index=a_idx,
            budget=budget,
            attempt_dir=(
                ckpt_dir
                / f"baseline_{b_idx}_attempt_{a_idx}_b{_budget_label(budget)}"
            ),
            score_fn=score_fn,
            agent=agent,
            verifier=verifier,
            prompt_inputs=prompt_inputs,
            config=config,
            baseline_accepted=[b],
            baseline_breakdown=b.breakdown,
        )
        for b_idx, b in enumerate(baselines)
        for a_idx, budget in enumerate(config.attempt_budgets_bytes)
    ]
    per_attempt = await asyncio.gather(*attempt_coros)
    for admitted in per_attempt:
        for entry in admitted:
            cell = library_cell(
                lib, entry.input_contracts, entry.output_contracts)
            insert_pareto(cell, entry)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emit_variants_module(
        out_path=ckpt_dir / "variants.py",
        child_name=node.name,
        variants=library_to_variant_registry(lib),
    )
    return lib


def _seed_root_baseline(
    *,
    lib: NodeLibrary,
    node_name: str,
    pass1_dsl: str,
    score_fn: ScoreFn,
    descendant_dsls: list[str],
    children_picks: dict[str, DesignEntry],
) -> tuple[DesignEntry, str]:
    """Like ``_seed_baseline`` but for the root (no synthetic wrapper, no
    parent_contract). The root's library has a single cell keyed by
    empty input/output contracts.

    Returns ``(entry, baseline_breakdown)`` — see ``_seed_baseline``.
    """
    composed = compose_source(
        parent_dsl=pass1_dsl,
        descendant_dsls_postorder=descendant_dsls,
    )
    cycles, on_chip = score_fn(composed)
    breakdown = _maybe_breakdown(score_fn, composed)
    entry = DesignEntry(
        dsl=pass1_dsl,
        input_contracts={},
        output_contracts={},
        cycles=cycles,
        on_chip=on_chip,
        provenance="pass1_baseline",
        breakdown=breakdown,
        children_picks=dict(children_picks),
    )
    cell = library_cell(lib, {}, {})
    cell.append(entry)
    return entry, breakdown


# ---------------------------------------------------------------------------
# Bottom-up driver
# ---------------------------------------------------------------------------


@dataclass
class AutotuneResult:
    libraries: dict[str, NodeLibrary]
    """node_path -> library."""

    root_path: str

    def root_library(self) -> NodeLibrary:
        return self.libraries[self.root_path]


# Files + dir-prefixes that ``search_leaf`` / ``search_parent`` own and that
# need wiping when a node's snapshot stamp doesn't match — leaving any of
# these around would make a fresh run either collide with prior turn dirs
# or read stale ``variants.py`` from the previous attempt.
_NODE_OWNED_FILES = frozenset({
    SNAPSHOT_FILENAME, "variants.py", "system_prompt.txt",
    "pass1_baseline_score.json",
})
_NODE_OWNED_DIR_PREFIXES = ("baseline_",)


def _wipe_node_run_artifacts(node_ckpt: Path) -> None:
    """Delete the node's own run artifacts in-place, preserving child
    subdirectories (children live at ``node_ckpt / <child_name>/`` and may
    hold valid snapshots from already-completed search tasks)."""
    if not node_ckpt.exists():
        return
    for item in node_ckpt.iterdir():
        if item.name in _NODE_OWNED_FILES and item.is_file():
            item.unlink()
        elif item.is_dir() and any(
            item.name.startswith(p) for p in _NODE_OWNED_DIR_PREFIXES
        ):
            shutil.rmtree(item)


async def autotune(
    *,
    plan_tree: Tree,
    pass1_dsls: dict[str, str],
    pass1_contracts: dict[str, Contract],
    ckpt_dir: Path,
    make_score_fn: Callable[[dict], ScoreFn],
    root_tensors: dict,
    agent_factory: Callable[[str], AgentFn],
    make_verifier: Callable[["PlanNode", "Contract | None", dict], VerifierFn],
    prompt_inputs: dict[str, NodePromptInputs],
    system_prompts: dict[str, str],
    config: SearchConfig = SearchConfig(),
    node_stamps: dict[str, str] | None = None,
    initial_libraries: dict[str, NodeLibrary] | None = None,
    max_baselines_per_node: int = 4,
    baseline_selection: str = "pareto_diverse",
    pass_subdir: str | None = None,
) -> AutotuneResult:
    """Walk plan_tree bottom-up and search each node — in parallel where
    the tree shape allows.

    Each node becomes its own ``asyncio.Task`` that first ``await``s its
    children's tasks (yielding the event loop so siblings progress) and
    then runs ``search_leaf`` / ``search_parent``. All leaves start
    concurrently; a parent fires as soon as every node in its subtree
    has completed. This mirrors pass-1's top-down parallel walk in
    reverse — the only sequencing constraint here is "children before
    parent" (a parent's prompt embeds the children's Pareto libraries).

    Inputs (keyed by node path):
      - ``pass1_dsls``: verified pass-1 DSL function per node.
      - ``pass1_contracts``: recorded ``Contract`` per non-root node.
        Root's entry is ignored.
      - ``prompt_inputs``: per-node ``NodePromptInputs`` for building
        autotune2 user prompts (rebuilt per attempt with the current
        Pareto front rendered inline).
      - ``system_prompts``: per-node autotune2 system prompt string
        (built by ``build_autotune2_system_prompt``). Passed to
        ``agent_factory`` to construct a per-node ``AgentFn`` with the
        system prompt baked into the underlying SDK ``Agent``.
      - ``agent_factory``: ``Callable[[system_prompt], AgentFn]``. The
        driver calls it once per node, so each ``search_leaf`` /
        ``search_parent`` receives an agent whose system prompt and
        conversation history are local to that node.
      - ``make_score_fn``: ``Callable[[tensors_dict], ScoreFn]``. The
        driver invokes it once per node to construct a node-local
        analytical scorer whose closed-over ``tensors`` dict matches
        the per-node arg names referenced by that node's synthetic
        wrapper (e.g. ``tensors["Q"]``). For non-root nodes the dict
        is built via ``build_node_tensors_dict(parent_contract)``;
        for the root it's ``root_tensors``.
      - ``make_verifier``: ``Callable[[PlanNode, Contract | None,
        tensors_dict], VerifierFn]``. Mirrors ``make_score_fn`` but
        builds a per-node 4-gate verifier closed over the node's own
        gold / call-args / tensors. Root nodes get a verifier whose
        gold comes from the kernel's ``compute_gold`` (entry point
        ``tiled_reference(dims, tensors)``). Non-root nodes get one
        whose gold is the recorded ``parent_contract.tiled_outputs``
        and whose entry point is ``<node_name>(*tiled_values, *,
        out_shapes=...)`` — mirroring pass-1's non-root verification.
      - ``root_tensors``: kernel-level tensors dict (same shape as
        the dict passed to ``make_analytical_scorer`` in pass-1). Used
        only at the root, whose DSL is the kernel's own
        ``tiled_reference`` and resolves ``tensors[...]`` against the
        kernel inputs directly.
      - ``node_stamps``: optional ``{node_path: stamp}`` for per-node
        snapshot resume. When provided, each node checks for an existing
        ``library.json`` under its ckpt dir and reuses it if its stored
        stamp matches; otherwise the node's ckpt subtree is wiped and the
        search re-runs, saving a fresh snapshot at the end. ``None``
        disables persistence — no load, no save, no cleanup. Stamps must
        be transitive (a descendant DSL change must invalidate every
        ancestor's stamp); see ``persistence.compute_plan_stamps``.
      - ``initial_libraries``: optional ``{node_path: NodeLibrary}`` from
        a prior pass. When present, each node calls ``select_baselines``
        to pick ``max_baselines_per_node`` extra branches from the prior
        pass's library (excluding the prior library's own pass-1
        baseline by identity, since this pass re-seeds its own pass-1
        baseline at index 0). Branches are threaded into
        ``search_leaf``/``search_parent`` as ``initial_baselines``. Pass-0
        (single-pass mode) leaves this ``None``, preserving existing
        semantics.
      - ``max_baselines_per_node`` / ``baseline_selection``: knobs for
        ``select_baselines``. Only consulted when ``initial_libraries``
        is non-None.
      - ``pass_subdir``: optional dir-name suffix appended to each
        node's ckpt path so multi-pass artifacts land in
        ``<ckpt>/autotune2/<node>/<pass_subdir>/`` (node-major layout).
        ``None`` (default) keeps the single-pass layout
        ``<ckpt>/autotune2/<node>/``.

    Returns the full ``{node_path: NodeLibrary}`` map plus the root path.
    """
    root_path = plan_tree.root.path
    tasks: dict[str, asyncio.Task] = {}

    async def _search_node(node: PlanNode) -> NodeLibrary:
        # Wait for every child's task before doing any work on this node.
        # ``asyncio.gather`` yields the loop while children are pending,
        # letting unrelated subtrees (e.g. other leaves) run in parallel.
        if node.children:
            child_libs_list = await asyncio.gather(
                *[tasks[c.path] for c in node.children]
            )
            child_libs = {
                c.path: lib for c, lib in zip(node.children, child_libs_list)
            }
        else:
            child_libs = {}

        node_ckpt = ckpt_dir / "autotune2" / node.path
        if pass_subdir is not None:
            node_ckpt = node_ckpt / pass_subdir

        # Resume path: if a snapshot exists with a matching stamp, reuse
        # it and skip the search entirely. Mismatched / missing → wipe the
        # node's own run artifacts (attempt_*, variants.py, etc.) so the
        # upcoming search's writes don't collide with stale files. Children
        # live under ``node_ckpt`` (paths are hierarchical: ``root/`` contains
        # ``root/leaf_a/``), so a blanket rmtree would clobber an already-
        # completed child's snapshot — wipe selectively.
        node_stamp = node_stamps.get(node.path) if node_stamps else None
        if node_stamp is not None:
            loaded = try_load_library_snapshot(
                node_ckpt / SNAPSHOT_FILENAME,
                expected_stamp=node_stamp,
                children_libraries=child_libs,
            )
            if loaded is not None:
                return loaded
            _wipe_node_run_artifacts(node_ckpt)

        node_system_prompt = system_prompts[node.path]
        node_agent = agent_factory(node_system_prompt)
        is_root_node = node.path == root_path
        parent_contract = None if is_root_node else pass1_contracts[node.path]
        # Build the per-node tensors dict the wrapper's offchip_load /
        # ``tensors[arg_name]`` references resolve against. For the root the
        # kernel-level tensors are correct (its DSL is the kernel's own
        # ``tiled_reference``); for any other node we synthesize per-arg
        # zero tensors at the contract's vanilla shapes — the analytical
        # timing model only inspects shapes, not values.
        node_tensors = (
            root_tensors if is_root_node
            else build_node_tensors_dict(parent_contract)
        )
        node_score_fn = make_score_fn(node_tensors)
        node_verifier = make_verifier(node, parent_contract, node_tensors)

        # Multi-pass branching: pull additional starting designs from the
        # prior pass's library for this node. Exclude the prior library's
        # own pass-1 baseline by object identity — this pass re-seeds its
        # own pass-1 baseline at index 0 via ``_seed_baseline``, and
        # double-seeding wastes one branch slot on a duplicate.
        node_initial_baselines: list[DesignEntry] | None = None
        if initial_libraries is not None and node.path in initial_libraries:
            prior_lib = initial_libraries[node.path]
            prior_pass1 = find_pass1_baseline_entry(prior_lib)
            node_initial_baselines = select_baselines(
                prior_lib,
                k=max_baselines_per_node,
                strategy=baseline_selection,
                exclude=[prior_pass1],
            )

        if node.is_leaf:
            lib = await search_leaf(
                node=node,
                parent_contract=parent_contract,
                pass1_dsl=pass1_dsls[node.path],
                ckpt_dir=node_ckpt,
                score_fn=node_score_fn,
                agent=node_agent,
                verifier=node_verifier,
                prompt_inputs=prompt_inputs[node.path],
                config=config,
                system_prompt=node_system_prompt,
                initial_baselines=node_initial_baselines,
            )
        else:
            # Parent: gather each child's pass-1 baseline entry. We must use
            # the pass-1 DSL (not the child's Pareto-best LLM variant) because
            # the parent's baseline is the canonical pass-1 reference — pass-1
            # already proved that the parent's pass-1 DSL composes with each
            # child's pass-1 DSL. An LLM child variant may declare the same
            # output contract but a different stream/tile decomposition that
            # the parent's pass-1 DSL wasn't authored against.
            baseline_picks: dict[str, DesignEntry] = {
                child.path: find_pass1_baseline_entry(child_libs[child.path])
                for child in node.children
            }
            lib = await search_parent(
                node=node,
                parent_contract=parent_contract,
                pass1_dsl=pass1_dsls[node.path],
                children_libraries=child_libs,
                children_picks_baseline=baseline_picks,
                ckpt_dir=node_ckpt,
                score_fn=node_score_fn,
                agent=node_agent,
                verifier=node_verifier,
                prompt_inputs=prompt_inputs[node.path],
                config=config,
                system_prompt=node_system_prompt,
                initial_baselines=node_initial_baselines,
            )

        if node_stamp is not None:
            save_library_snapshot(
                lib,
                path=node_ckpt / SNAPSHOT_FILENAME,
                stamp=node_stamp,
                children_libraries=child_libs,
            )
        return lib

    # Schedule every node's task. ``iter_topological`` is post-order, so
    # child tasks are created before any parent task that references them.
    for node in plan_tree.iter_topological():
        tasks[node.path] = asyncio.create_task(
            _search_node(node), name=f"autotune2:{node.path}",
        )

    # ``gather`` lets the first failure cancel the rest — fail-fast is the
    # right default here because a child failure means the parent can't
    # be composed anyway.
    results = await asyncio.gather(*tasks.values())
    libraries = {path: lib for path, lib in zip(tasks.keys(), results)}
    return AutotuneResult(libraries=libraries, root_path=root_path)


# ---------------------------------------------------------------------------
# Library snapshot (debug / resume)
# ---------------------------------------------------------------------------


def write_library_snapshot(lib: NodeLibrary, out_path: Path) -> None:
    """Write a JSON debug snapshot of a node's library.

    Stores per-cell metadata (contract keys, entry counts, Pareto
    coordinates, provenance) but **not** DSL source — the DSL is
    already on disk via ``emit_variants_module`` (declarative cells)
    and the per-turn checkpoint dirs. Intended for grep/inspection,
    not round-trip persistence.
    """
    payload: list[dict] = []
    for in_key, by_out in lib.items():
        in_repr = [
            {"arg": k, "reshape": list(c.reshape),
             "permutation": list(c.permutation)}
            for k, c in in_key
        ]
        for out_key, cell in by_out.items():
            out_repr = [
                {"out": k, "reshape": list(c.reshape),
                 "permutation": list(c.permutation)}
                for k, c in out_key
            ]
            payload.append({
                "input_contracts": in_repr,
                "output_contracts": out_repr,
                "entries": [
                    {"cycles": e.cycles, "on_chip": e.on_chip,
                     "provenance": e.provenance,
                     "children_picks": list(e.children_picks.keys())}
                    for e in cell
                ],
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
