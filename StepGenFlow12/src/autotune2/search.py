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
   constructed by ``build_synthetic_wrapper_for_node``. It expects
   every TensorArg input to be **RAW** (Phase 5 v1 limitation: the
   wrapper passes ``tensors[arg_name]`` directly into the node's
   function; on-chip args would require synthesizing
   ``offchip_load`` + contract-realizing DSL ops, which is the
   parent-DSL writer's responsibility and not mechanically derivable.
   Deferred to a future version). The wrapper terminates with
   ``offchip_store`` so the translator's root-return-wrapping logic
   has a stream to wrap.

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
   unique ``(input_contracts, output_contracts)`` cell, when the
   library is serialized to ``variants.py``. Cells with multiple
   Pareto-front entries share the variant index — the
   ``cartesian_compose`` step inside ``search_parent`` enumerates
   them.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import torch

from src.autotune2.compose import (
    ScoreFn,
    cartesian_compose,
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
from src.autotune2.pareto import insert_pareto
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
# composed_source -> VerifyResult
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
    loop. Exhausting this budget on an attempt aborts that attempt; the
    outer loop then starts a fresh attempt (new conversation) up to
    ``max_attempts`` times.
    """

    max_attempts: int = 5
    """Max fresh-conversation attempts per node (after the pass-1 baseline).

    Each attempt begins with a fresh user prompt that includes a summary
    of already-accepted Pareto-front entries (rendered via
    ``render_accepted_summary``) so the LLM targets gaps.
    """

    check_order: str = "correctness-first"
    """Gate cascade order, mirrors pass-1's ``check_order`` knob.

    One of ``"correctness-first"``, ``"compliance-first"``, or
    ``"always-both"``. Wired through to the 4-gate cascade
    (correctness, compliance regex, LLM judge, post-validator) inside
    ``build_real_verifier_fn``.
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


def build_synthetic_wrapper_for_node(
    *,
    node_name: str,
    parent_contract: Contract,
) -> str:
    """Build a ``def tiled_reference(dims, tensors): ...`` wrapper that
    calls ``<node_name>`` with kernel-level tensor refs.

    Phase 5 v1 requires that every ``TensorArg`` input is RAW (i.e. the
    parent's call site forwarded it without on-chip loading). Asserts
    loudly when this isn't met — supporting on-chip-arg-at-non-root
    scoring requires generating ``offchip_load`` + contract-realizing
    DSL ops, which is the parent-DSL writer's responsibility and can't
    be mechanically derived from the contract alone.
    """
    arg_names = parent_contract.arg_names
    arg_specs = parent_contract.arg_specs
    arg_is_raw = parent_contract.arg_is_raw
    out_shapes = parent_contract.out_shapes

    for name, spec, raw in zip(arg_names, arg_specs, arg_is_raw):
        if isinstance(spec, TensorArg):
            assert raw, (
                f"build_synthetic_wrapper_for_node({node_name!r}): on-chip "
                f"TensorArg {name!r} not supported in Phase 5 v1 — would "
                f"require synthesizing offchip_load + contract-realizing "
                f"DSL ops. Deferred."
            )

    lookups = ", ".join(f'tensors["{n}"]' for n in arg_names)
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
        body = (
            f"    result = {node_name}({lookups}, out_shapes={out_shapes_repr})\n"
            f"    return offchip_store(promote_outer(result))\n"
        )
    else:
        out_names = [f"_out_{i}" for i in range(n_outputs)]
        destruct = ", ".join(out_names)
        lines = [
            f"    {destruct} = {node_name}({lookups}, out_shapes={out_shapes_repr})",
        ]
        # Store every output but the last; return the final store so the
        # tiled_reference has a well-defined return value (mirrors single-
        # output case).
        for nm in out_names[:-1]:
            lines.append(f"    offchip_store(promote_outer({nm}))")
        lines.append(f"    return offchip_store(promote_outer({out_names[-1]}))")
        body = "\n".join(lines) + "\n"
    return f"def tiled_reference(dims, tensors):\n{body}"


def build_node_tensors_dict(parent_contract: Contract) -> dict:
    """Build a ``tensors`` dict for scoring this node in isolation.

    For each TensorArg: ``torch.zeros(spec.shape)`` (timing model
    inspects shape only). For each list arg: the recorded
    ``tiled_values`` entry (already a list[Tensor] or list[int]).
    """
    out: dict = {}
    for name, spec, val in zip(
        parent_contract.arg_names,
        parent_contract.arg_specs,
        parent_contract.tiled_values,
    ):
        if isinstance(spec, TensorArg):
            out[name] = torch.zeros(spec.shape)
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
    """Assign sequential variant indices, one per unique (in,out) cell.

    Multiple Pareto-front entries within a single cell share one index —
    they all represent the same boundary contract, differing only in
    internal DSL. Deterministic in lib iteration order.
    """
    reg: dict[int, dict] = {}
    idx = 0
    for in_key, by_out in lib.items():
        in_contracts = dict(in_key)
        for out_key in by_out:
            out_contracts = dict(out_key)
            reg[idx] = {
                "input_contracts": in_contracts,
                "output_contracts": out_contracts,
            }
            idx += 1
    return reg


def cell_for_variant(lib: NodeLibrary, variant_index: int) -> list[DesignEntry]:
    """Resolve a variant index back to its Pareto-front cell."""
    idx = 0
    for in_key, by_out in lib.items():
        for out_key, cell in by_out.items():
            if idx == variant_index:
                return cell
            idx += 1
    raise AssertionError(
        f"cell_for_variant: variant_index {variant_index} out of range; "
        f"library has {idx} cells"
    )


def render_library_as_variant_summaries(
    lib: NodeLibrary,
) -> list[VariantSummary]:
    """Build the summary list shown to the LLM. One ``VariantSummary``
    per cell, with the cell's Pareto-best entry as the representative
    (cycles, on_chip) coordinates.
    """
    summaries: list[VariantSummary] = []
    idx = 0
    for in_key, by_out in lib.items():
        in_contracts = dict(in_key)
        for out_key, cell in by_out.items():
            assert cell, (
                f"render_library_as_variant_summaries: cell at index {idx} is "
                "empty; library should never have empty cells"
            )
            # Show the cell's lowest-cycle entry as the representative; the
            # full front is enumerated at composition time via cartesian_compose.
            best = min(cell, key=lambda e: (e.cycles, e.on_chip))
            summaries.append(VariantSummary(
                variant_index=idx,
                input_contracts=in_contracts,
                output_contracts=dict(out_key),
                cycles=best.cycles,
                on_chip=best.on_chip,
            ))
            idx += 1
    return summaries


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
    wrapper_source: str,
    descendant_dsls: list[str],
    children_picks: dict[str, DesignEntry],
) -> DesignEntry:
    """Insert the pass-1 baseline with identity contracts into the library."""
    identity_in = {
        name: vanilla_contract_for(spec.shape)
        for name, spec, raw in zip(
            parent_contract.arg_names, parent_contract.arg_specs,
            parent_contract.arg_is_raw)
        if isinstance(spec, TensorArg) and not raw
    }
    output_names = tuple(f"out_{i}" for i in range(len(parent_contract.out_shapes)))
    identity_out = {
        name: vanilla_contract_for(tuple(parent_contract.out_shapes[i]))
        for i, name in enumerate(output_names)
    }
    composed = compose_source(
        parent_dsl=wrapper_source + "\n" + pass1_dsl,
        descendant_dsls_postorder=descendant_dsls,
    )
    cycles, on_chip = score_fn(composed)
    entry = DesignEntry(
        dsl=pass1_dsl,
        input_contracts=identity_in,
        output_contracts=identity_out,
        cycles=cycles,
        on_chip=on_chip,
        provenance="pass1_baseline",
        children_picks=dict(children_picks),
    )
    cell = library_cell(lib, identity_in, identity_out)
    cell.append(entry)
    return entry


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


def _write_turn_artifacts(
    turn_dir: Path,
    *,
    user_prompt: str,
    agent_response: AgentResponse,
    status: str,
    extracted_code: str | None = None,
    composed_source: str | None = None,
    verify_result: VerifyResult | None = None,
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
) -> NodeLibrary:
    """Populate one leaf node's library.

    Seeds with the pass-1 baseline (identity contracts), then runs up to
    ``config.max_attempts`` fresh-conversation attempts. Each attempt
    accumulates up to ``config.max_turns_per_attempt`` LLM turns of
    parse/verify feedback before giving up and starting fresh. Fresh
    attempts render a Pareto-front summary of already-accepted entries
    into the user prompt so the LLM targets gaps.

    ``parent_contract`` is ``None`` only for the root-as-leaf case
    (single-node plan tree): the pass-1 DSL is already
    ``tiled_reference(dims, tensors)`` so no synthetic wrapper is
    needed and the library uses empty identity contracts.
    """
    assert node.is_leaf, f"search_leaf called on non-leaf node {node.path!r}"
    lib: NodeLibrary = {}
    is_root = parent_contract is None
    wrapper = "" if is_root else build_synthetic_wrapper_for_node(
        node_name=node.name, parent_contract=parent_contract,
    )

    if system_prompt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "system_prompt.txt").write_text(system_prompt)

    if is_root:
        baseline = _seed_root_baseline(
            lib=lib, node_name=node.name, pass1_dsl=pass1_dsl,
            score_fn=score_fn, descendant_dsls=[], children_picks={},
        )
    else:
        baseline = _seed_baseline(
            lib=lib, node_name=node.name, parent_contract=parent_contract,
            pass1_dsl=pass1_dsl, score_fn=score_fn,
            wrapper_source=wrapper, descendant_dsls=[],
            children_picks={},
        )

    arg_vanilla_shapes = (
        {} if is_root else _on_chip_vanilla_shapes(parent_contract)
    )
    output_vanilla_shapes = (
        {} if is_root else _output_vanilla_shapes(parent_contract)
    )
    accepted: list[DesignEntry] = [baseline]

    for attempt in range(config.max_attempts):
        attempt_dir = ckpt_dir / f"attempt_{attempt}"
        accepted_summary = render_accepted_summary(
            accepted,
            arg_vanilla_shapes=arg_vanilla_shapes,
            output_vanilla_shapes=output_vanilla_shapes,
        )
        user_prompt = build_autotune2_user_prompt(
            is_leaf=True,
            node_name=node.name,
            function_signature=prompt_inputs.function_signature,
            pytorch_reference=prompt_inputs.pytorch_reference,
            pass1_dsl=pass1_dsl,
            dims_block=prompt_inputs.dims_block,
            tensors_block=prompt_inputs.tensors_block,
            accepted_summary=accepted_summary,
        )
        conversation: list[dict] = [{"role": "user", "content": user_prompt}]

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

            cycles, on_chip = score_fn(composed)
            entry = DesignEntry(
                dsl=parsed.dsl,
                input_contracts=parsed.input_contracts,
                output_contracts=parsed.output_contracts,
                cycles=cycles,
                on_chip=on_chip,
                provenance=f"llm_attempt_{attempt}_turn_{turn}",
            )
            cell = library_cell(
                lib, parsed.input_contracts, parsed.output_contracts)
            if insert_pareto(cell, entry):
                accepted.append(entry)
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="ACCEPTED",
                extracted_code=parsed.dsl,
                composed_source=composed,
                verify_result=verify,
            )
            break  # success → break inner loop, start a fresh attempt

    # Persist registry artifact for downstream variant binding.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emit_variants_module(
        out_path=ckpt_dir / "variants.py",
        child_name=node.name,
        variants=library_to_variant_registry(lib),
    )
    return lib


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
) -> NodeLibrary:
    """Populate one parent node's library.

    ``children_libraries`` provides each immediate child's already-
    populated library (keyed by child_path). ``children_picks_baseline``
    provides the pass-1-baseline pick per child (used to score the
    parent's pass-1 baseline against the children's pass-1 baselines).

    Outer loop: ``config.max_attempts`` fresh-conversation attempts.
    Inner loop: ``config.max_turns_per_attempt`` parse/verify-feedback
    turns within one attempt. Each attempt's user prompt renders the
    current Pareto front so the LLM targets gaps. On a successful turn
    the parent's DSL is composed with the Cartesian product of the
    picked children's Pareto fronts; every non-dominated composition is
    admitted to the library.
    """
    lib: NodeLibrary = {}
    is_root = parent_contract is None

    # Build the synthetic wrapper (non-root) or use the parent DSL directly (root).
    wrapper = "" if is_root else build_synthetic_wrapper_for_node(
        node_name=node.name, parent_contract=parent_contract,
    )

    if system_prompt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "system_prompt.txt").write_text(system_prompt)

    # Baseline: parent's pass-1 DSL composed with each child's pass-1 entry.
    baseline_descendants: list[str] = []
    for child_entry in children_picks_baseline.values():
        baseline_descendants.extend(gather_descendants_postorder(child_entry))
        baseline_descendants.append(child_entry.dsl)
    if is_root:
        baseline = _seed_root_baseline(
            lib=lib, node_name=node.name, pass1_dsl=pass1_dsl,
            score_fn=score_fn, descendant_dsls=baseline_descendants,
            children_picks=children_picks_baseline,
        )
    else:
        baseline = _seed_baseline(
            lib=lib, node_name=node.name, parent_contract=parent_contract,
            pass1_dsl=pass1_dsl, score_fn=score_fn,
            wrapper_source=wrapper, descendant_dsls=baseline_descendants,
            children_picks=children_picks_baseline,
        )

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
    arg_vanilla_shapes = (
        {} if is_root else _on_chip_vanilla_shapes(parent_contract)
    )
    output_vanilla_shapes = (
        {} if is_root else _output_vanilla_shapes(parent_contract)
    )
    accepted: list[DesignEntry] = [baseline]

    for attempt in range(config.max_attempts):
        attempt_dir = ckpt_dir / f"attempt_{attempt}"
        accepted_summary = render_accepted_summary(
            accepted,
            arg_vanilla_shapes=arg_vanilla_shapes,
            output_vanilla_shapes=output_vanilla_shapes,
        )
        user_prompt = build_autotune2_user_prompt(
            is_leaf=False,
            node_name=node.name,
            function_signature=prompt_inputs.function_signature,
            pytorch_reference=prompt_inputs.pytorch_reference,
            pass1_dsl=pass1_dsl,
            dims_block=prompt_inputs.dims_block,
            tensors_block=prompt_inputs.tensors_block,
            child_variant_blocks=child_blocks,
            accepted_summary=accepted_summary,
        )
        conversation: list[dict] = [{"role": "user", "content": user_prompt}]

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
                children_fronts: dict[str, list[DesignEntry]] = {
                    child.path: cell_for_variant(
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
            children_order = [c.path for c in node.children]

            # Score every Cartesian combination and admit non-dominated entries.
            cell = library_cell(
                lib, parsed.input_contracts, parsed.output_contracts)
            admitted_this_turn: list[DesignEntry] = []
            last_failure: str = ""
            # Track the first composed source + last verify result for the
            # per-turn checkpoint. Cartesian compositions can fan out widely;
            # logging every one would explode the turn_dir. The first combo
            # is representative for the entry's DSL content (descendants
            # differ but the parent DSL is identical), and the last verify
            # result is what's surfaced in the user-feedback message.
            first_composed: str | None = None
            last_verify: VerifyResult | None = None
            for chosen, descendants in _iter_parent_compositions(
                parent_dsl=parsed.dsl,
                children_fronts=children_fronts,
                children_order=children_order,
            ):
                composed = compose_source(
                    parent_dsl=wrapper + ("\n" if wrapper else "") + parsed.dsl,
                    descendant_dsls_postorder=descendants,
                )
                if first_composed is None:
                    first_composed = composed
                verify = await verifier(composed)
                last_verify = verify
                if not verify.passed:
                    last_failure = verify.feedback
                    continue
                cycles, on_chip = score_fn(composed)
                entry = DesignEntry(
                    dsl=parsed.dsl,
                    input_contracts=parsed.input_contracts,
                    output_contracts=parsed.output_contracts,
                    cycles=cycles,
                    on_chip=on_chip,
                    provenance=f"llm_attempt_{attempt}_turn_{turn}",
                    children_picks=chosen,
                )
                if insert_pareto(cell, entry):
                    admitted_this_turn.append(entry)

            if admitted_this_turn:
                accepted.extend(admitted_this_turn)
                _write_turn_artifacts(
                    turn_dir,
                    user_prompt=turn_user_prompt,
                    agent_response=agent_response,
                    status=f"ACCEPTED ({len(admitted_this_turn)} entries)",
                    extracted_code=parsed.dsl,
                    composed_source=first_composed,
                    verify_result=last_verify,
                )
                break  # success → break inner loop, start a fresh attempt
            _write_turn_artifacts(
                turn_dir,
                user_prompt=turn_user_prompt,
                agent_response=agent_response,
                status="VERIFY_FAIL_ALL_COMPOSITIONS",
                extracted_code=parsed.dsl,
                composed_source=first_composed,
                verify_result=last_verify,
            )
            _append_turn_feedback(conversation, (
                "Your variant did not pass verification under any "
                "Cartesian combination of the picked children's Pareto "
                "front. Last gate feedback follows; please emit a "
                "corrected DSL implementation:\n\n"
                f"{last_failure}"
            ))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emit_variants_module(
        out_path=ckpt_dir / "variants.py",
        child_name=node.name,
        variants=library_to_variant_registry(lib),
    )
    return lib


def _iter_parent_compositions(
    *,
    parent_dsl: str,
    children_fronts: dict[str, list[DesignEntry]],
    children_order: list[str],
):
    """Yield (chosen_dict, descendant_dsls_postorder) per Cartesian combo.

    Mirrors ``cartesian_compose`` from compose.py but exposes the chosen
    per-child entry plus the full transitive descendant DSL list (so
    grandchildren etc. are included). Doesn't call any scorer — used
    inside ``search_parent``'s per-turn loop where verification + scoring
    happen on the composed source.
    """
    import itertools
    fronts = [children_fronts[p] for p in children_order]
    for combo in itertools.product(*fronts):
        chosen = {p: e for p, e in zip(children_order, combo)}
        descendants: list[str] = []
        for path in children_order:
            entry = chosen[path]
            descendants.extend(gather_descendants_postorder(entry))
            descendants.append(entry.dsl)
        yield chosen, descendants


def _seed_root_baseline(
    *,
    lib: NodeLibrary,
    node_name: str,
    pass1_dsl: str,
    score_fn: ScoreFn,
    descendant_dsls: list[str],
    children_picks: dict[str, DesignEntry],
) -> DesignEntry:
    """Like ``_seed_baseline`` but for the root (no synthetic wrapper, no
    parent_contract). The root's library has a single cell keyed by
    empty input/output contracts."""
    composed = compose_source(
        parent_dsl=pass1_dsl,
        descendant_dsls_postorder=descendant_dsls,
    )
    cycles, on_chip = score_fn(composed)
    entry = DesignEntry(
        dsl=pass1_dsl,
        input_contracts={},
        output_contracts={},
        cycles=cycles,
        on_chip=on_chip,
        provenance="pass1_baseline",
        children_picks=dict(children_picks),
    )
    cell = library_cell(lib, {}, {})
    cell.append(entry)
    return entry


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


async def autotune(
    *,
    plan_tree: Tree,
    pass1_dsls: dict[str, str],
    pass1_contracts: dict[str, Contract],
    ckpt_dir: Path,
    score_fn: ScoreFn,
    agent_factory: Callable[[str], AgentFn],
    verifier: VerifierFn,
    prompt_inputs: dict[str, NodePromptInputs],
    system_prompts: dict[str, str],
    config: SearchConfig = SearchConfig(),
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
        node_system_prompt = system_prompts[node.path]
        node_agent = agent_factory(node_system_prompt)
        if node.is_leaf:
            return await search_leaf(
                node=node,
                parent_contract=(
                    None if node.path == root_path
                    else pass1_contracts[node.path]
                ),
                pass1_dsl=pass1_dsls[node.path],
                ckpt_dir=node_ckpt,
                score_fn=score_fn,
                agent=node_agent,
                verifier=verifier,
                prompt_inputs=prompt_inputs[node.path],
                config=config,
                system_prompt=node_system_prompt,
            )

        # Parent: gather each child's baseline entry (the Pareto-best of
        # the baseline cell, i.e. the first cell by construction in
        # ``_seed_baseline``).
        baseline_picks: dict[str, DesignEntry] = {}
        for child in node.children:
            first_cell = next(iter(next(iter(child_libs[child.path].values())).values()))
            baseline_picks[child.path] = min(
                first_cell, key=lambda e: (e.cycles, e.on_chip)
            )
        return await search_parent(
            node=node,
            parent_contract=(
                None if node.path == root_path
                else pass1_contracts[node.path]
            ),
            pass1_dsl=pass1_dsls[node.path],
            children_libraries=child_libs,
            children_picks_baseline=baseline_picks,
            ckpt_dir=node_ckpt,
            score_fn=score_fn,
            agent=node_agent,
            verifier=verifier,
            prompt_inputs=prompt_inputs[node.path],
            config=config,
            system_prompt=node_system_prompt,
        )

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
