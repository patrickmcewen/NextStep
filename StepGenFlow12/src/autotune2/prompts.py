"""Autotune2-specific prompt assembly + response parsing.

Phase 4 of the autotuner pipeline. Autotune2 builds self-contained
system and user prompts (no pass-1 dependency at the prompt-text
level). The only content shared with pass-1 is the literal contents of
``step_dsl.py``, which both passes embed verbatim in their system
prompt as the DSL surface reference.

Framing differs from pass-1:

  - Pass-1 is first-time lowering: PyTorch → DSL.
  - Autotune2 is variant generation: produce a Pareto-distinct DSL
    design starting from an already-verified pass-1 implementation.

The Phase 5 search driver invokes ``build_autotune2_system_prompt``
once per node (passed as the agent's ``instructions``) and
``build_autotune2_user_prompt`` once per LLM attempt (the user message
that opens each fresh conversation; subsequent gate-feedback messages
within an attempt are appended directly by the search loop).

Output protocol
---------------
The LLM is asked to emit a fenced YAML block immediately above the
Python DSL block:

    ```yaml
    parent_input_contracts:       # empty when no input is on-chip
      Q: {reshape: [8, 8, 64], permutation: [1, 0, 2]}
    ```

    ```python
    def attention_block(...):
        ...
    ```

Output contracts are NOT declared by the LLM — they are derived
mechanically from the built graph (the wrapper's ``OffChipStore``
nodes carry the actual produced stream+tile shapes). See
``runtime._derive_output_contracts_from_graph``.

``parse_autotune2_response`` splits the response back into a structured
``AutotuneResponse`` (parent_input_contracts dict of ``TensorContract``,
and the raw DSL source string). All schema
violations raise ``AssertionError`` with the offending fragment in the
message; the search driver routes the assertion text back into the
agent loop as a feedback message.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from src.autotune2.contracts import TensorContract, vanilla_contract_for

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"
_AUTOTUNE_PROMPTS_DIR = _PROMPTS_DIR / "autotune"
_TILE_SHRINK_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "tile_shrink"
_PARALLEL_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "parallel"
_BATCHING_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "batching"
_GENERAL_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "general"
_CURATION_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "curation"
_ACE_CONTEXT_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "ace_context"
_SIM_MANAGER_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "sim_manager"
_SHARED_PROMPTS_DIR = _AUTOTUNE_PROMPTS_DIR / "shared"
_SYSTEM_PROMPT_PATH = _TILE_SHRINK_PROMPTS_DIR / "autotune2_system.txt"
_SYSTEM_PROMPT_PATH_PARALLEL = (
    _PARALLEL_PROMPTS_DIR / "autotune2_system_parallel.txt"
)
_SYSTEM_PROMPT_PATH_GENERAL = _GENERAL_PROMPTS_DIR / "autotune2_system_general.txt"
_MEMORY_NOTES_PATH = _SHARED_PROMPTS_DIR / "dsl_memory_notes.txt"
_TILE_SHRINK_FEWSHOT_PATH = (
    _TILE_SHRINK_PROMPTS_DIR / "autotune_tile_shrink_fewshot.txt"
)
_PARALLEL_FEWSHOT_PATH = _PARALLEL_PROMPTS_DIR / "autotune_parallel_fewshot.txt"
_BATCHING_FEWSHOT_PATH = _BATCHING_PROMPTS_DIR / "autotune_batching_fewshot.txt"
# Per-example DSL files inlined into the parallel few-shot via placeholders
# (see _PARALLEL_FEWSHOT_EXAMPLES below). Each file is a validated
# tiled_reference checked by StepDB/validate_functional_dsl.py.
_PARALLEL_EXAMPLES_DIR = (
    _PROJECT_ROOT.parent / "StepDB" / "examples" / "parallel"
)
_PARALLEL_FEWSHOT_EXAMPLES = {
    "{gemm_seq_code}": _PARALLEL_EXAMPLES_DIR / "gemm_seq.py",
    "{gemm_par_indep_code}": _PARALLEL_EXAMPLES_DIR / "gemm_par_indep.py",
    "{gemm_par_shared_code}": _PARALLEL_EXAMPLES_DIR / "gemm_par_shared.py",
    "{element_wise_add_seq_code}": _PARALLEL_EXAMPLES_DIR / "element_wise_add_seq.py",
    "{element_wise_add_par_code}": _PARALLEL_EXAMPLES_DIR / "element_wise_add_par.py",
    "{vector_reduce_sum_seq_code}": _PARALLEL_EXAMPLES_DIR / "vector_reduce_sum_seq.py",
    "{vector_reduce_sum_par_code}": _PARALLEL_EXAMPLES_DIR / "vector_reduce_sum_par.py",
    "{chained_unary_seq_code}": _PARALLEL_EXAMPLES_DIR / "chained_unary_seq.py",
    "{chained_unary_par_code}": _PARALLEL_EXAMPLES_DIR / "chained_unary_par.py",
    "{sdpa_core_max_seq_code}": _PARALLEL_EXAMPLES_DIR / "sdpa_core_max_seq.py",
    "{sdpa_core_max_par_shallow_code}": _PARALLEL_EXAMPLES_DIR / "sdpa_core_max_par_shallow.py",
    "{sdpa_core_max_par_deep_code}": _PARALLEL_EXAMPLES_DIR / "sdpa_core_max_par_deep.py",
}
_GENERAL_FEWSHOT_PATH = _GENERAL_PROMPTS_DIR / "autotune_general_fewshot.txt"
_CURATION_SYSTEM_PROMPT_PATH = _CURATION_PROMPTS_DIR / "system.txt"
_CURATION_USER_PROMPT_PATH = _CURATION_PROMPTS_DIR / "user.txt"
_ACE_CONTEXT_CURATOR_SYSTEM_PROMPT_PATH = _ACE_CONTEXT_PROMPTS_DIR / "system.txt"
_ACE_CONTEXT_CURATOR_USER_PROMPT_PATH = _ACE_CONTEXT_PROMPTS_DIR / "user.txt"
_SIM_DECISION_SYSTEM_PROMPT_PATH = _SIM_MANAGER_PROMPTS_DIR / "system.txt"
_SIM_DECISION_USER_PROMPT_PATH = _SIM_MANAGER_PROMPTS_DIR / "user.txt"
_STEP_DSL_IR_OP_GROUPS_PATH = _SHARED_PROMPTS_DIR / "step_dsl_ir_op_groups.txt"
_STEP_DSL_MEMORY_PY = _PROJECT_ROOT / "src" / "step_dsl_memory.py"
_OUTPUT_PROTOCOL_PATHS = {
    True: _SHARED_PROMPTS_DIR / "autotune2_output_protocol_leaf.txt",
    False: _SHARED_PROMPTS_DIR / "autotune2_output_protocol_parent.txt",
}
_USER_PROMPT_PATHS = {
    "tile_shrink": _TILE_SHRINK_PROMPTS_DIR / "autotune2_user_tile_shrink.txt",
    "parallel": _PARALLEL_PROMPTS_DIR / "autotune2_user_parallel.txt",
    "general": _GENERAL_PROMPTS_DIR / "autotune2_user_general.txt",
}
_SYSTEM_PROMPT_SPECS = {
    "tile_shrink": (
        _SYSTEM_PROMPT_PATH,
        _TILE_SHRINK_FEWSHOT_PATH,
        "{tile_shrink_fewshot}",
    ),
    "parallel": (
        _SYSTEM_PROMPT_PATH_PARALLEL,
        _PARALLEL_FEWSHOT_PATH,
        "{parallel_fewshot}",
    ),
    "general": (
        _SYSTEM_PROMPT_PATH_GENERAL,
        _GENERAL_FEWSHOT_PATH,
        "{general_fewshot}",
    ),
}


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------


def render_contract_human(
    contract: TensorContract,
    vanilla_shape: tuple[int, ...],
) -> str:
    """Return ``"vanilla"`` or ``"vanilla.reshape(...).permute(...)"``.

    Identity contracts (``reshape == vanilla_shape`` and identity
    permutation) collapse to the literal string ``"vanilla"``; this
    keeps the variant table compact in the common case.
    """
    if contract.is_identity_of(vanilla_shape):
        return "vanilla"
    rs = ",".join(str(d) for d in contract.reshape)
    ps = ",".join(str(p) for p in contract.permutation)
    return f"vanilla.reshape(({rs})).permute({ps})"


# ---------------------------------------------------------------------------
# Variant table rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSummary:
    """One library entry summarized for the LLM.

    Stores only the contract metadata + Pareto coordinates — the DSL
    body of the variant is NOT shown (per HANDOFF.md: the LLM never
    sees stub bodies, only declarative contract metadata).

    ``cycle_source`` is the tag of the simulator that produced
    ``cycles`` — ``"analytical"`` (STeP timing model) or ``"rust"``
    (cycle-approximate rust sim). Surfaced inline next to every
    rendered cycle count so the LLM can tell which numbers in a
    mixed-source library are directly comparable to each other (see
    the mixed-source caveat in the autotune2 system prompt).
    """

    variant_index: int
    input_contracts: dict[str, TensorContract]
    output_contracts: dict[str, TensorContract]
    cycles: int
    on_chip: int
    cycle_source: str = "analytical"


def render_variant_block(
    *,
    child_name: str,
    arg_vanilla_shapes: dict[str, tuple[int, ...]],
    arg_is_raw: dict[str, bool],
    output_vanilla_shapes: dict[str, tuple[int, ...]],
    variants: list[VariantSummary],
) -> str:
    """Render a per-child variant table as markdown.

    Sections:
      1. Per-arg classification (vanilla shape + RAW/on-chip tag).
      2. Per-output vanilla shape header.
      3. One stanza per variant: index, Pareto coordinates, input/output
         contracts in ``vanilla.reshape(...).permute(...)`` form.

    Identity-everywhere variants render as ``(all vanilla)`` for both
    input and output contract sections.
    """
    assert set(arg_vanilla_shapes.keys()) == set(arg_is_raw.keys()), (
        f"render_variant_block({child_name!r}): arg_vanilla_shapes keys "
        f"{set(arg_vanilla_shapes.keys())!r} must equal arg_is_raw keys "
        f"{set(arg_is_raw.keys())!r}"
    )
    for v in variants:
        for arg in v.input_contracts:
            assert arg in arg_vanilla_shapes, (
                f"render_variant_block({child_name!r}): variant "
                f"{v.variant_index} has input_contract for {arg!r} not in "
                f"arg_vanilla_shapes={set(arg_vanilla_shapes.keys())!r}"
            )
            assert not arg_is_raw[arg], (
                f"render_variant_block({child_name!r}): variant "
                f"{v.variant_index} has input_contract for RAW arg {arg!r}; "
                f"contracts apply only to on-chip args"
            )
        for out in v.output_contracts:
            assert out in output_vanilla_shapes, (
                f"render_variant_block({child_name!r}): variant "
                f"{v.variant_index} has output_contract for {out!r} not in "
                f"output_vanilla_shapes={set(output_vanilla_shapes.keys())!r}"
            )

    lines = [f"#### Child: {child_name}", ""]

    lines.append("  arg classification:")
    for arg, shape in arg_vanilla_shapes.items():
        tag = "RAW" if arg_is_raw[arg] else "on-chip"
        lines.append(f"    {arg} (vanilla {shape}): {tag}")
    lines.append("")

    lines.append("  outputs:")
    for out, shape in output_vanilla_shapes.items():
        lines.append(f"    {out} (vanilla {shape})")
    lines.append("")

    lines.append("  variants:")
    for v in variants:
        lines.append(
            f"    [{v.variant_index}] "
            f"cycles={v.cycles} ({v.cycle_source}), "
            f"on_chip={v.on_chip}"
        )
        if v.input_contracts:
            lines.append("        input contracts:")
            for arg, c in v.input_contracts.items():
                lines.append(
                    f"          {arg}: "
                    f"{render_contract_human(c, arg_vanilla_shapes[arg])}"
                )
        else:
            lines.append("        input contracts: (none — all inputs RAW)")
        if v.output_contracts:
            lines.append("        output contracts:")
            for out, c in v.output_contracts.items():
                lines.append(
                    f"          {out}: "
                    f"{render_contract_human(c, output_vanilla_shapes[out])}"
                )
        else:
            lines.append("        output contracts: (all vanilla)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-contained system + user prompt builders
# ---------------------------------------------------------------------------


def build_autotune2_system_prompt(
    *, is_leaf: bool, dsl_code: str, fewshot: str = "tile_shrink",
    max_tile: int | None = None,
) -> str:
    """Self-contained autotune2 system prompt.

    ``fewshot`` selects which worked-example pack to inject and which
    system-prompt template to load:

      - ``"tile_shrink"`` (default):
        ``prompts/autotune/tile_shrink/autotune2_system.txt`` +
        ``prompts/autotune/tile_shrink/autotune_tile_shrink_fewshot.txt``
        (placeholder ``{tile_shrink_fewshot}``). The load/consumer/
        reduction-order rewrite recipe.
      - ``"parallel"``:
        ``prompts/autotune/parallel/autotune2_system_parallel.txt`` +
        ``prompts/autotune/parallel/autotune_parallel_fewshot.txt``
        (placeholder ``{parallel_fewshot}``). Shared vs. independent
        parallelism worked examples.
      - ``"general"``:
        ``prompts/autotune/general/autotune2_system_general.txt`` +
        ``prompts/autotune/general/autotune_general_fewshot.txt``
        (placeholder ``{general_fewshot}``). Merged tile-shrink and
        parallelism guidance.

    Other placeholders are identical across variants: ``{step_dsl_code}``
    (the DSL surface, passed in), ``{memory_notes}`` (loaded from
    ``prompts/autotune/shared/dsl_memory_notes.txt``), and
    ``{output_protocol}`` (loaded from
    ``prompts/autotune/shared/autotune2_output_protocol_{leaf,parent}.txt``
    based on ``is_leaf``). ``str.replace`` is used instead of
    ``str.format`` because the protocol fragments contain literal YAML braces.

    ``max_tile`` (when set) substitutes the same pass-1 max-tile
    addendum (``src.agents._MAX_TILE_ADDENDUM_TEMPLATE``) into the
    ``{max_tile_addendum}`` placeholder so the autotune2 LLM sees the
    same load/store/reshape/stub-call bounds it would see in pass-1.
    When ``None``, the placeholder collapses to an empty string.
    """
    assert dsl_code, "build_autotune2_system_prompt: dsl_code must be non-empty"
    assert fewshot in _SYSTEM_PROMPT_SPECS, (
        f"build_autotune2_system_prompt: fewshot must be one of "
        f"{sorted(_SYSTEM_PROMPT_SPECS.keys())!r}, got {fewshot!r}"
    )
    assert max_tile is None or (isinstance(max_tile, int) and max_tile >= 1), (
        f"build_autotune2_system_prompt: max_tile must be a positive int or "
        f"None, got {max_tile!r}"
    )
    system_path, fewshot_path, fewshot_placeholder = _SYSTEM_PROMPT_SPECS[fewshot]
    protocol_path = _OUTPUT_PROTOCOL_PATHS[is_leaf]
    for path in (system_path, protocol_path,
                 _MEMORY_NOTES_PATH, fewshot_path,
                 _STEP_DSL_MEMORY_PY):
        assert path.exists(), (
            f"build_autotune2_system_prompt: required file not found at {path}"
        )
    memory_notes = _MEMORY_NOTES_PATH.read_text().replace(
        "{step_dsl_memory_code}", _STEP_DSL_MEMORY_PY.read_text()
    ).rstrip()
    from src.agents import _MAX_TILE_ADDENDUM_TEMPLATE
    max_tile_addendum = (
        _MAX_TILE_ADDENDUM_TEMPLATE.format(max_tile=int(max_tile))
        if max_tile is not None else ""
    )
    fewshot_text = (
        fewshot_path.read_text()
        .replace("{tile_shrink_fewshot}", _TILE_SHRINK_FEWSHOT_PATH.read_text().rstrip())
        .replace("{parallel_fewshot}", _PARALLEL_FEWSHOT_PATH.read_text().rstrip())
        .replace("{batching_fewshot}", _BATCHING_FEWSHOT_PATH.read_text().rstrip())
        .rstrip()
    )
    # Inline each parallel-fewshot example .py file at its placeholder. Done
    # after the {parallel_fewshot} substitution so this works whether the
    # selected fewshot pack is "parallel" (placeholders sit inline) or
    # "general" (placeholders arrive via the nested {parallel_fewshot} swap).
    for placeholder, example_path in _PARALLEL_FEWSHOT_EXAMPLES.items():
        if placeholder not in fewshot_text:
            continue
        assert example_path.exists(), (
            f"build_autotune2_system_prompt: parallel few-shot example "
            f"missing at {example_path} (placeholder {placeholder})"
        )
        fewshot_text = fewshot_text.replace(
            placeholder, example_path.read_text().rstrip()
        )
    rendered = (
        system_path.read_text()
        .replace("{step_dsl_code}", dsl_code)
        .replace("{memory_notes}", memory_notes)
        .replace(fewshot_placeholder, fewshot_text)
        .replace("{output_protocol}", protocol_path.read_text().rstrip())
        .replace("{max_tile_addendum}", max_tile_addendum)
    )
    # Append the mixed-source caveat unconditionally — the agent never
    # knows which simulation manager is wired up for its current run,
    # and the inline ``(analytical|rust)`` tag on every rendered cycle
    # value (see ``VariantSummary`` / ``render_accepted_summary``) only
    # carries the right warning weight when this caveat is in scope.
    return rendered + "\n" + _MIXED_SOURCE_CAVEAT


_MIXED_SOURCE_CAVEAT = (
    "## A note on mixed cycle sources\n\n"
    "Every cycle count surfaced to you in this run carries an inline "
    "tag — ``(analytical)`` or ``(rust)`` — identifying which simulator "
    "produced it. ``analytical`` is the STeP closed-form timing model: "
    "cheap, deterministic, and the source you saw in earlier autotune2 "
    "runs. ``rust`` is the cycle-approximate rust simulator: slower, "
    "more accurate, and the ground-truth target.\n\n"
    "Different sources are NOT directly comparable. A variant with "
    "``cycles=1000 (rust)`` is not guaranteed faster than one with "
    "``cycles=1100 (analytical)``: the analytical model can over- or "
    "under-estimate by 2x or more on some op patterns, and a library "
    "may contain both source types when the per-pass rust budget runs "
    "out mid-pass. Use the tag to:\n\n"
    "  - rank within the same tag first (rust-vs-rust, analytical-vs-"
    "analytical); cross-tag comparisons are only suggestive;\n"
    "  - prefer designs that beat the pass-1 baseline by a meaningful "
    "margin under whichever source the baseline was measured against, "
    "not just by single-digit percentages that could be model noise;\n"
    "  - assume the autotuner will rust-evaluate one final pick at the "
    "end of the run as ground truth — your job is to expose a strong "
    "Pareto front, not to micro-optimize a single cycle number.\n"
)


_VARIANT_SECTION_TEMPLATE = """

### Child variant libraries

Reference summary of verified child designs. The autotuner selects the
child implementation deterministically system-side; do not emit child
variant indices or child selection fields. These tables are included only
to show boundary contracts and Pareto coordinates measured for each child
design.

{variant_tables}
"""


_CHILD_DESIGN_EXAMPLES_TEMPLATE = """

### Child design examples

The child designs below are already verified and scored. Use them as
in-context implementation examples when rewriting this parent. Child
selection is deterministic/system-side; do not emit child variant indices
or child selection fields.

{child_design_examples}
"""


_ACCEPTED_HINT = " shown above"
_PARENT_HINT_TEXT = (
    "\n  - Calls child functions by natural name. The autotuner composes "
    "against deterministic best child designs selected system-side; do "
    "not emit child selection fields. You may call a labeled child helper "
    "shown in the examples or inline equivalent child logic directly "
    "inside this parent function."
)


def build_autotune2_user_prompt(
    *,
    is_leaf: bool,
    node_name: str,
    function_signature: str,
    pytorch_reference: str,
    baseline_dsl: str,
    dims_block: str,
    tensors_block: str,
    fewshot: str = "tile_shrink",
    child_variant_blocks: dict[str, str] | None = None,
    child_design_examples: str = "",
    accepted_summary: str = "",
    budget_block: str = "",
    ace_context: str = "",
) -> str:
    """Self-contained autotune2 user prompt for one (node, attempt) pair.

    ``fewshot`` selects which per-agent user-prompt template to load
    from ``prompts/autotune/<agent>/`` — currently ``"tile_shrink"``,
    ``"parallel"``, and ``"general"``, matching the system-prompt agents in
    ``build_autotune2_system_prompt``. Each template carries the same
    placeholders but specializes the "Your task" framing toward the
    agent's recipe (tile-shrink vs. parallelism).

    ``baseline_dsl`` is the DSL the LLM is asked to vary. For single-pass
    autotune2 this is always the pass-1 baseline; multi-pass / branching
    expansion supplies any prior library entry's DSL.

    ``accepted_summary`` is the empty string on the first attempt; on
    subsequent fresh attempts it is the rendered Pareto-front summary
    of already-accepted variants (see ``render_accepted_summary``) so
    the LLM can target gaps. Parent prompts must provide either
    ``child_variant_blocks`` (read-only child library summaries) or
    ``child_design_examples`` (source examples for system-selected best
    child designs); leaves may provide neither. ``budget_block`` is the empty
    string for an unlimited attempt or a pre-formatted markdown stanza
    describing this attempt's on-chip memory budget (the per-attempt
    parallel-fan-out signal in ``search_leaf`` / ``search_parent``).
    """
    assert fewshot in _USER_PROMPT_PATHS, (
        f"build_autotune2_user_prompt: fewshot must be one of "
        f"{sorted(_USER_PROMPT_PATHS.keys())!r}, got {fewshot!r}"
    )
    template_path = _USER_PROMPT_PATHS[fewshot]
    assert template_path.exists(), (
        f"build_autotune2_user_prompt: required template not found at "
        f"{template_path}"
    )
    if is_leaf:
        assert not child_variant_blocks, (
            "build_autotune2_user_prompt: leaf prompts must not include "
            "child_variant_blocks; got "
            f"{list((child_variant_blocks or {}).keys())!r}"
        )
        assert not child_design_examples, (
            "build_autotune2_user_prompt: leaf prompts must not include "
            "child_design_examples"
        )
        variant_section = ""
    else:
        assert child_variant_blocks or child_design_examples.strip(), (
            "build_autotune2_user_prompt: parent prompts must include at "
            "least one child variant block or child design example"
        )
        sections: list[str] = []
        if child_design_examples.strip():
            sections.append(_CHILD_DESIGN_EXAMPLES_TEMPLATE.format(
                child_design_examples=child_design_examples.strip()
            ))
        if child_variant_blocks:
            sections.append(_VARIANT_SECTION_TEMPLATE.format(
                variant_tables="\n\n".join(child_variant_blocks.values())
            ))
        variant_section = "".join(sections)
    accepted_section = (
        f"\n\n### Already-accepted variants for this node\n\n{accepted_summary}\n"
        if accepted_summary else ""
    )
    ace_context_section = (
        f"\n\n### Learned optimization context\n\n{ace_context.strip()}\n"
        if ace_context.strip() else ""
    )
    return template_path.read_text().format(
        node_name=node_name,
        function_signature=function_signature,
        pytorch_reference=pytorch_reference,
        baseline_dsl=baseline_dsl,
        dims_block=dims_block,
        tensors_block=tensors_block,
        variant_section=variant_section,
        accepted_section=accepted_section,
        budget_section=budget_block + ace_context_section,
        accepted_hint=_ACCEPTED_HINT if accepted_summary else "",
        parent_hint="" if is_leaf else _PARENT_HINT_TEXT,
    )


# ---------------------------------------------------------------------------
# Pareto-front summary for fresh-attempt prompts
# ---------------------------------------------------------------------------


def render_accepted_summary(
    accepted: list,
    *,
    arg_vanilla_shapes: dict[str, tuple[int, ...]],
    output_vanilla_shapes: dict[str, tuple[int, ...]],
) -> str:
    """Render already-accepted variants as a compact markdown block.

    Each entry surfaces (cycles, on_chip) and the boundary contracts so
    the LLM can target gaps in the Pareto front on its next attempt.
    Empty list returns the empty string so callers can pass the result
    directly to ``build_autotune2_user_prompt(accepted_summary=...)``.

    ``accepted`` is a list of ``DesignEntry`` (typed loosely to avoid a
    cycle with ``contracts.py``); the fields read are ``cycles``,
    ``on_chip``, ``input_contracts``, and ``output_contracts``.
    """
    if not accepted:
        return ""
    lines: list[str] = []
    for idx, entry in enumerate(accepted):
        # ``cycle_source`` is rendered inline so the LLM can tell which
        # cycles values in a mixed-source library are directly
        # comparable. ``getattr`` keeps loosely-typed test stubs that
        # pass non-DesignEntry objects working (default = analytical).
        source = getattr(entry, "cycle_source", "analytical")
        lines.append(
            f"  [{idx}] cycles={entry.cycles} ({source}), "
            f"on_chip={entry.on_chip}"
        )
        if entry.input_contracts:
            lines.append("        input contracts:")
            for arg, c in entry.input_contracts.items():
                vanilla = arg_vanilla_shapes.get(arg)
                rendered = (
                    render_contract_human(c, vanilla)
                    if vanilla is not None else f"<unknown arg {arg!r}>"
                )
                lines.append(f"          {arg}: {rendered}")
        else:
            lines.append("        input contracts: (none — all inputs RAW)")
        if entry.output_contracts:
            lines.append("        output contracts:")
            for out, c in entry.output_contracts.items():
                vanilla = output_vanilla_shapes.get(out)
                rendered = (
                    render_contract_human(c, vanilla)
                    if vanilla is not None else f"<unknown output {out!r}>"
                )
                lines.append(f"          {out}: {rendered}")
        else:
            lines.append("        output contracts: (all vanilla)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


@dataclass
class AutotuneResponse:
    """Parsed LLM response.

    ``input_contracts`` uses TensorContract values (validated against any
    provided vanilla shape — see ``parse_*``). ``dsl`` is the raw DSL
    function source extracted from the response's python code block.
    Output contracts are NOT parsed from the LLM response — the verifier
    derives them from the built graph; see
    ``VerifyResult.derived_output_contracts``.
    """

    input_contracts: dict[str, TensorContract] = field(default_factory=dict)
    dsl: str = ""


_FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+\-]*)\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _extract_fenced(response_text: str, lang: str) -> str | None:
    """First fenced code block whose info string matches ``lang``."""
    target = lang.lower()
    for m in _FENCED_BLOCK_RE.finditer(response_text):
        if m.group("lang").lower() == target:
            return m.group("body")
    return None


def _parse_contract_yaml(
    d,
    *,
    where: str,
) -> TensorContract:
    assert isinstance(d, dict), (
        f"{where}: contract entry must be a YAML mapping, got "
        f"{type(d).__name__}={d!r}"
    )
    assert set(d.keys()) == {"reshape", "permutation"}, (
        f"{where}: contract entry keys must be exactly "
        f"{{'reshape', 'permutation'}}, got {sorted(d.keys())!r}"
    )
    reshape = d["reshape"]
    perm = d["permutation"]
    assert isinstance(reshape, list) and all(isinstance(x, int) for x in reshape), (
        f"{where}: 'reshape' must be a list of ints, got {reshape!r}"
    )
    assert isinstance(perm, list) and all(isinstance(x, int) for x in perm), (
        f"{where}: 'permutation' must be a list of ints, got {perm!r}"
    )
    return TensorContract(reshape=tuple(reshape), permutation=tuple(perm))


def parse_autotune2_response(
    response_text: str,
    *,
    is_leaf: bool,
    expected_child_names: Iterable[str] = (),
) -> AutotuneResponse:
    """Split a fenced YAML + python response into structured fields.

    All schema violations raise ``AssertionError`` with the offending
    fragment surfaced. Callers (the search driver's per-turn feedback
    path) catch and route the message back into the agent loop.
    """
    yaml_body = _extract_fenced(response_text, "yaml")
    assert yaml_body is not None, (
        "parse_autotune2_response: response must contain a fenced ```yaml block "
        "with the autotuner output spec (parent_input_contracts); "
        "none found"
    )
    py_body = _extract_fenced(response_text, "python")
    assert py_body is not None, (
        "parse_autotune2_response: response must contain a fenced ```python "
        "block with the DSL function body; none found"
    )

    try:
        parsed = yaml.safe_load(yaml_body)
    except yaml.YAMLError as exc:
        raise AssertionError(
            "parse_autotune2_response: invalid yaml block; emit valid YAML "
            "inside the fenced ```yaml block. PyYAML error:\n"
            f"{exc}\n\nYAML block was:\n{yaml_body}"
        ) from exc
    assert isinstance(parsed, dict), (
        f"parse_autotune2_response: yaml block must be a mapping at top level, "
        f"got {type(parsed).__name__}"
    )

    allowed_keys_parent = {"parent_input_contracts"}
    required_keys_parent = {"parent_input_contracts"}
    expected_keys_leaf = {"parent_input_contracts"}
    allowed = expected_keys_leaf if is_leaf else allowed_keys_parent
    required = expected_keys_leaf if is_leaf else required_keys_parent
    unexpected = set(parsed.keys()) - allowed
    missing = required - set(parsed.keys())
    assert not unexpected, (
        f"parse_autotune2_response: unexpected yaml keys "
        f"{sorted(unexpected)!r}; allowed={sorted(allowed)!r}. Note: "
        f"`parent_output_contracts` is no longer accepted — output contracts "
        f"are derived from the built graph."
    )
    assert not missing, (
        f"parse_autotune2_response: missing yaml keys {sorted(missing)!r}; "
        f"required={sorted(required)!r}"
    )

    response = AutotuneResponse(dsl=py_body)

    in_c = parsed["parent_input_contracts"] or {}
    assert isinstance(in_c, dict), (
        f"parse_autotune2_response: parent_input_contracts must be a mapping "
        f"(or empty/None), got {type(in_c).__name__}"
    )
    response.input_contracts = {
        name: _parse_contract_yaml(
            v, where=f"parent_input_contracts[{name!r}]")
        for name, v in in_c.items()
    }
    return response


# ---------------------------------------------------------------------------
# Helpers for emission
# ---------------------------------------------------------------------------


def contract_to_yaml_dict(c: TensorContract) -> dict:
    """Inverse of ``_parse_contract_yaml`` (for embedding in YAML fixtures)."""
    return {"reshape": list(c.reshape), "permutation": list(c.permutation)}


# ---------------------------------------------------------------------------
# CurationAgent / SimDecisionAgent prompts (PR4)
# ---------------------------------------------------------------------------
#
# CurationAgent picks the K most predictive calibration records for a target
# composed source; SimDecisionAgent (used by AgentManager) decides per-variant
# whether to spend the per-pass Rust-simulator budget on this variant. Both
# emit a fenced ```json block with a fixed schema — parsers fail-loud on any
# deviation (CLAUDE.md style: no try/except on agent output).
#
# AgentManager itself is the wrapper that calls these two agents and falls
# back to analytical scoring on any agent failure (LLM RPC error, parser
# failure, etc.) — degradation is at the call-site boundary, the parsers
# stay strict.


def build_curation_system_prompt() -> str:
    """System prompt for the curation agent. Static — no inputs.

    The DSL ↔ IR op-group reference (shared with the sim-decision agent
    so both agents reason about op identity the same way) is loaded from
    ``prompts/autotune/shared/step_dsl_ir_op_groups.txt`` and substituted
    into the ``{step_dsl_ir_op_groups}`` placeholder at the bottom of the
    template.
    """
    assert _CURATION_SYSTEM_PROMPT_PATH.exists(), (
        f"curation system prompt not found: {_CURATION_SYSTEM_PROMPT_PATH}"
    )
    assert _STEP_DSL_IR_OP_GROUPS_PATH.exists(), (
        f"DSL ↔ IR op-group reference not found: {_STEP_DSL_IR_OP_GROUPS_PATH}"
    )
    return _CURATION_SYSTEM_PROMPT_PATH.read_text().replace(
        "{step_dsl_ir_op_groups}", _STEP_DSL_IR_OP_GROUPS_PATH.read_text().rstrip()
    )


def build_sim_decision_system_prompt() -> str:
    """System prompt for the in-loop simulation-decision agent. Static.

    Shares the ``{step_dsl_ir_op_groups}`` placeholder + reference file
    with ``build_curation_system_prompt``; see that builder's docstring.
    """
    assert _SIM_DECISION_SYSTEM_PROMPT_PATH.exists(), (
        f"sim-decision system prompt not found: {_SIM_DECISION_SYSTEM_PROMPT_PATH}"
    )
    assert _STEP_DSL_IR_OP_GROUPS_PATH.exists(), (
        f"DSL ↔ IR op-group reference not found: {_STEP_DSL_IR_OP_GROUPS_PATH}"
    )
    return _SIM_DECISION_SYSTEM_PROMPT_PATH.read_text().replace(
        "{step_dsl_ir_op_groups}", _STEP_DSL_IR_OP_GROUPS_PATH.read_text().rstrip()
    )


@dataclass(frozen=True)
class CurationCandidate:
    """One row the curation agent ranks.

    ``record_id`` is a caller-chosen stable string (typically the
    sha256-prefix of the composed source — same scheme as the on-disk
    sources_dir filename). ``composed_source`` is the full DSL text the
    record was measured against.
    """

    record_id: str
    composed_source: str
    analytical_cycles: int
    rust_cycles: int
    kernel: str
    preset: str


def build_curation_user_prompt(
    *,
    target_source: str,
    candidates: list[CurationCandidate],
    k: int,
) -> str:
    """Per-call user prompt for the curation agent.

    Renders the target source + N candidate blocks + an explicit "pick K".
    ``record_id`` strings are echoed exactly in the agent's reply, so they
    must be plain text — no embedded whitespace or fence characters.
    """
    assert target_source, "build_curation_user_prompt: target_source must be non-empty"
    assert candidates, (
        "build_curation_user_prompt: candidates must be non-empty "
        "(an empty candidate set has nothing to rank — the caller should "
        "short-circuit before invoking the curation agent)"
    )
    assert isinstance(k, int) and 1 <= k <= len(candidates), (
        f"build_curation_user_prompt: k must be 1 <= k <= len(candidates) "
        f"(={len(candidates)}), got {k!r}"
    )
    seen_ids: set[str] = set()
    for c in candidates:
        assert isinstance(c, CurationCandidate), (
            f"build_curation_user_prompt: every candidate must be a "
            f"CurationCandidate, got {type(c).__name__}"
        )
        assert c.record_id and not any(ch.isspace() for ch in c.record_id), (
            f"build_curation_user_prompt: record_id {c.record_id!r} must be "
            f"non-empty with no whitespace (echoed verbatim in agent reply)"
        )
        assert c.record_id not in seen_ids, (
            f"build_curation_user_prompt: duplicate record_id {c.record_id!r} "
            f"in candidate list"
        )
        seen_ids.add(c.record_id)

    blocks: list[str] = []
    for c in candidates:
        ratio = c.rust_cycles / c.analytical_cycles if c.analytical_cycles else 0.0
        blocks.append(
            f"### Record {c.record_id}\n"
            "```\n"
            f"{c.composed_source}\n"
            "```\n"
            f"analytical_cycles={c.analytical_cycles}, "
            f"rust_cycles={c.rust_cycles}, ratio={ratio:.3f}, "
            f"kernel={c.kernel}, preset={c.preset}\n"
        )
    assert _CURATION_USER_PROMPT_PATH.exists(), (
        f"curation user prompt not found: {_CURATION_USER_PROMPT_PATH}"
    )
    return (
        _CURATION_USER_PROMPT_PATH.read_text()
        .replace("{target_source}", target_source)
        .replace("{candidate_count}", str(len(candidates)))
        .replace("{candidate_blocks}", "\n".join(blocks))
        .replace("{k}", str(k))
    )


def build_sim_decision_user_prompt(
    *,
    node_path: str,
    is_root: bool,
    variant_kind: str,
    attempt_index: int,
    turn_index: int,
    composed_source: str,
    analytical_cycles: int,
    analytical_on_chip: int,
    remaining_seconds: float,
    recent_rust_avg_sec: float,
    consumed_seconds: float,
    curated: list[CurationCandidate],
) -> str:
    """Per-call user prompt for the in-loop simulation-decision agent."""
    assert variant_kind in ("baseline", "variant"), (
        f"build_sim_decision_user_prompt: variant_kind must be 'baseline' "
        f"or 'variant', got {variant_kind!r}"
    )
    if curated:
        curated_block = "\n".join(
            f"### Record {c.record_id}\n"
            "```\n"
            f"{c.composed_source}\n"
            "```\n"
            f"analytical_cycles={c.analytical_cycles}, "
            f"rust_cycles={c.rust_cycles}, "
            f"ratio="
            f"{(c.rust_cycles / c.analytical_cycles if c.analytical_cycles else 0.0):.3f}"
            f"\n"
            for c in curated
        )
    else:
        curated_block = "(no curated records available — cold start)\n"

    # `inf` for unlimited budget renders as "inf"; the decision agent
    # should read that as "no cap, run rust whenever it makes sense".
    rem = (
        "inf" if remaining_seconds == float("inf") else f"{remaining_seconds:.1f}"
    )
    assert _SIM_DECISION_USER_PROMPT_PATH.exists(), (
        f"sim-decision user prompt not found: {_SIM_DECISION_USER_PROMPT_PATH}"
    )
    return (
        _SIM_DECISION_USER_PROMPT_PATH.read_text()
        .replace("{node_path}", node_path)
        .replace("{is_root}", str(is_root))
        .replace("{variant_kind}", variant_kind)
        .replace("{attempt_index}", str(attempt_index))
        .replace("{turn_index}", str(turn_index))
        .replace("{composed_source}", composed_source)
        .replace("{analytical_cycles}", str(analytical_cycles))
        .replace("{analytical_on_chip}", str(analytical_on_chip))
        .replace("{remaining_seconds}", rem)
        .replace("{recent_rust_avg_sec}", f"{recent_rust_avg_sec:.2f}")
        .replace("{consumed_seconds}", f"{consumed_seconds:.2f}")
        .replace("{curated_count}", str(len(curated)))
        .replace("{curated_block}", curated_block)
    )


def parse_curation_response(
    response_text: str,
    *,
    candidate_ids: list[str],
    k: int,
) -> list[str]:
    """Extract the ranked record_id list from a curation-agent reply.

    Fails loudly (AssertionError) on any schema deviation — missing fence,
    wrong JSON shape, duplicate ID, ID not in ``candidate_ids``, or list
    length != ``k``. The caller (``AgentManager``) catches the assertion at
    the call-site boundary and falls back to analytical scoring; the
    parser itself stays strict so prompt-drift bugs surface immediately
    when CurationAgent is exercised directly.
    """
    import json as _json

    body = _extract_fenced(response_text, "json")
    if body is None:
        body = response_text.strip()
    assert body, "parse_curation_response: response must contain JSON"
    parsed = _json.loads(body)
    assert isinstance(parsed, dict), (
        f"parse_curation_response: top-level JSON must be an object, got "
        f"{type(parsed).__name__}"
    )
    assert "record_ids" in parsed, (
        f"parse_curation_response: JSON missing required key 'record_ids'; "
        f"got keys {sorted(parsed.keys())!r}"
    )
    record_ids = parsed["record_ids"]
    assert isinstance(record_ids, list), (
        f"parse_curation_response: 'record_ids' must be a list, got "
        f"{type(record_ids).__name__}"
    )
    assert len(record_ids) == k, (
        f"parse_curation_response: expected exactly {k} record_ids, got "
        f"{len(record_ids)}: {record_ids!r}"
    )
    candidate_set = set(candidate_ids)
    seen: set[str] = set()
    for rid in record_ids:
        assert isinstance(rid, str), (
            f"parse_curation_response: every record_id must be a string, "
            f"got {type(rid).__name__}={rid!r}"
        )
        assert rid in candidate_set, (
            f"parse_curation_response: record_id {rid!r} is not in the "
            f"candidate set (size {len(candidate_set)})"
        )
        assert rid not in seen, (
            f"parse_curation_response: duplicate record_id {rid!r} in "
            f"agent reply"
        )
        seen.add(rid)
    return list(record_ids)


def parse_sim_decision_response(response_text: str) -> tuple[str, str]:
    """Extract ``(decision, reason)`` from a sim-decision-agent reply.

    ``decision`` is asserted to be exactly ``"rust"`` or ``"analytical"``;
    ``reason`` is whatever string the agent provided (trimmed). Schema
    violations raise ``AssertionError``; the wrapping ``AgentManager``
    catches at the call-site boundary and falls back to analytical.
    """
    import json as _json

    body = _extract_fenced(response_text, "json")
    if body is None:
        body = response_text.strip()
    assert body, "parse_sim_decision_response: response must contain JSON"
    parsed = _json.loads(body)
    assert isinstance(parsed, dict), (
        f"parse_sim_decision_response: top-level JSON must be an object, "
        f"got {type(parsed).__name__}"
    )
    assert "decision" in parsed and "reason" in parsed, (
        f"parse_sim_decision_response: JSON must carry both 'decision' "
        f"and 'reason' keys; got {sorted(parsed.keys())!r}"
    )
    decision = parsed["decision"]
    reason = parsed["reason"]
    assert decision in ("rust", "analytical"), (
        f"parse_sim_decision_response: 'decision' must be 'rust' or "
        f"'analytical', got {decision!r}"
    )
    assert isinstance(reason, str), (
        f"parse_sim_decision_response: 'reason' must be a string, got "
        f"{type(reason).__name__}={reason!r}"
    )
    return decision, reason.strip()


# ---------------------------------------------------------------------------
# ACE context curator prompts
# ---------------------------------------------------------------------------


def build_ace_context_curator_system_prompt() -> str:
    """System prompt for refreshing the shared autotune2 ACE playbook."""
    assert _ACE_CONTEXT_CURATOR_SYSTEM_PROMPT_PATH.exists(), (
        f"ACE context curator system prompt not found: "
        f"{_ACE_CONTEXT_CURATOR_SYSTEM_PROMPT_PATH}"
    )
    return _ACE_CONTEXT_CURATOR_SYSTEM_PROMPT_PATH.read_text()


def build_ace_context_curator_user_prompt(
    *,
    current_playbook: str,
    events: list[dict],
    metadata: dict,
    turn_summaries: list[dict],
) -> str:
    """Render one ACE playbook-refresh request.

    The prompt is intentionally data-heavy and format-light: Python prepares
    stable JSON for events/metadata/turn summaries, while the template text
    tells the curator how to distill those facts into playbook bullets.
    """
    assert events, (
        "build_ace_context_curator_user_prompt: events must be non-empty"
    )
    assert isinstance(metadata, dict), (
        "build_ace_context_curator_user_prompt: metadata must be a dict"
    )
    assert isinstance(turn_summaries, list), (
        "build_ace_context_curator_user_prompt: turn_summaries must be a list"
    )
    assert _ACE_CONTEXT_CURATOR_USER_PROMPT_PATH.exists(), (
        f"ACE context curator user prompt not found: "
        f"{_ACE_CONTEXT_CURATOR_USER_PROMPT_PATH}"
    )
    playbook = current_playbook.strip() or "(empty playbook)"
    return (
        _ACE_CONTEXT_CURATOR_USER_PROMPT_PATH.read_text()
        .replace("{current_playbook}", playbook)
        .replace("{metadata_json}", json.dumps(metadata, indent=2, sort_keys=True))
        .replace("{events_json}", json.dumps(events, indent=2, sort_keys=True))
        .replace(
            "{turn_summaries_json}",
            json.dumps(turn_summaries, indent=2, sort_keys=True),
        )
    )


def parse_ace_context_curator_response(response_text: str) -> str:
    """Extract the next playbook from an ACE context curator reply."""
    body = _extract_fenced(response_text, "json")
    if body is None:
        body = response_text.strip()
    assert body, (
        "parse_ace_context_curator_response: response must contain JSON; "
        "got empty response"
    )
    parsed = json.loads(body)
    assert isinstance(parsed, dict), (
        f"parse_ace_context_curator_response: top-level JSON must be an "
        f"object, got {type(parsed).__name__}"
    )
    assert "playbook" in parsed, (
        f"parse_ace_context_curator_response: JSON missing required key "
        f"'playbook'; got keys {sorted(parsed.keys())!r}"
    )
    playbook = parsed["playbook"]
    assert isinstance(playbook, str) and playbook.strip(), (
        "parse_ace_context_curator_response: 'playbook' must be a "
        "non-empty string"
    )
    return playbook.strip()


# ---------------------------------------------------------------------------
# FinalPickAgent prompts (PR5)
# ---------------------------------------------------------------------------
#
# FinalPickAgent runs once at the end of an autotune2 run. Given the root
# Pareto front (one or more candidate variants, each with cycles + on_chip +
# cycle_source + composed source) plus curated calibration evidence per
# candidate, it picks the single variant most likely to have the lowest
# true (rust-measured) cycle count. The picked variant is the one that gets
# the run's single end-of-run rust evaluation, replacing today's
# ``min_cycles`` deterministic tiebreaker for runs configured with
# ``--root-pick=agent``. See HANDOFF design decision #6 + #8.
#
# Output protocol: same fenced ```json shape as the in-loop agents, with
# a single integer ``variant_index`` selecting which of the rendered
# candidates to use. Parser is strict (assertion-based); call-site
# fallback to ``min_cycles`` lives in ``runtime.final_pick``.


_FINAL_PICK_SYSTEM_PROMPT = """\
You pick a single variant from the root Pareto front of an autotune2 run.
The picked variant is rust-evaluated exactly once to produce the run's
reported number, so your goal is to maximize the chance the picked variant
has the lowest true (Rust-measured) cycle count among the candidates.

Each candidate carries:
  - an analytical or Rust-measured ``cycles`` number and an analytical
    ``on_chip`` byte count,
  - a ``cycle_source`` tag (``"analytical"`` or ``"rust"``) — Rust numbers
    are ground truth; analytical numbers can be off by 2x or more on some
    op patterns,
  - its composed DSL source,
  - a small set of curated past (analytical, Rust) calibration records on
    related code, picked by the curation agent.

Lean toward picking:
  - the lowest-cycles Rust-sourced candidate when one exists — its number
    is trustworthy and directly comparable to other Rust candidates,
  - an analytical candidate only when its cycles are clearly lower than
    every Rust candidate AND the curated records do NOT show the analytical
    model under-predicting on similar code (low rust/analytical ratio
    across the curated set),
  - the one with smaller on_chip when two candidates are otherwise tied —
    smaller on-chip footprint correlates with better data movement.

Avoid picking:
  - an analytical candidate whose cycles are suspiciously low while the
    curated records show analytical heavily under-predicts on similar code
    (high rust/analytical ratio) — the apparent speedup is likely a
    modeling artifact,
  - a Rust candidate whose cycles are dominated by another Rust candidate
    on both axes.

Output ONLY a fenced ```json block of the shape:

  ```json
  {"variant_index": 0, "reason": "<one short sentence>"}
  ```

``variant_index`` must be an integer in [0, N-1] where N is the number of
candidates rendered in the user message. ``reason`` is for telemetry only —
keep it under 25 words. No prose outside the fence.
"""


def build_final_pick_system_prompt() -> str:
    """System prompt for the end-of-run final-pick agent. Static — no inputs."""
    return _FINAL_PICK_SYSTEM_PROMPT


@dataclass(frozen=True)
class FinalPickCandidate:
    """One root-Pareto entry rendered to the final-pick agent.

    ``variant_index`` is the 0-based position in the rendered list and is
    echoed verbatim in the agent's reply. ``curated`` is the curation
    agent's per-candidate selection (may be empty on a cold-start run with
    no prior calibration records).
    """

    variant_index: int
    cycles: int
    on_chip: int
    cycle_source: str
    composed_source: str
    curated: list  # list[CurationCandidate]


def build_final_pick_user_prompt(
    *,
    root_path: str,
    kernel: str,
    preset: str,
    candidates: list,  # list[FinalPickCandidate]
) -> str:
    """Per-call user prompt for the final-pick agent.

    Renders one block per candidate with cycles + cycle_source + on_chip,
    its composed source, and the curation-agent-picked calibration evidence
    for THIS candidate (cycle pair + composed source per record). The
    ``variant_index`` of each candidate is its 0-based position in the
    list and is what the agent echoes back.
    """
    assert candidates, (
        "build_final_pick_user_prompt: candidates must be non-empty (the "
        "caller short-circuits on an empty root Pareto front)"
    )
    seen_idx: set[int] = set()
    for i, c in enumerate(candidates):
        assert isinstance(c, FinalPickCandidate), (
            f"build_final_pick_user_prompt: every candidate must be a "
            f"FinalPickCandidate, got {type(c).__name__}"
        )
        assert c.variant_index == i, (
            f"build_final_pick_user_prompt: candidates[{i}].variant_index="
            f"{c.variant_index!r} must equal its list position {i}"
        )
        assert c.variant_index not in seen_idx, (
            f"build_final_pick_user_prompt: duplicate variant_index "
            f"{c.variant_index!r}"
        )
        seen_idx.add(c.variant_index)
        assert c.cycle_source in ("analytical", "rust"), (
            f"build_final_pick_user_prompt: candidates[{i}].cycle_source="
            f"{c.cycle_source!r} must be 'analytical' or 'rust'"
        )

    blocks: list[str] = []
    for c in candidates:
        if c.curated:
            curated_block = "\n".join(
                f"  - record {r.record_id}: "
                f"analytical_cycles={r.analytical_cycles}, "
                f"rust_cycles={r.rust_cycles}, "
                f"ratio="
                f"{(r.rust_cycles / r.analytical_cycles if r.analytical_cycles else 0.0):.3f}, "
                f"kernel={r.kernel}, preset={r.preset}\n"
                f"    ```\n"
                f"    {r.composed_source.rstrip().replace(chr(10), chr(10) + '    ')}\n"
                f"    ```"
                for r in c.curated
            )
        else:
            curated_block = (
                "  (no curated records — cold start or no prior measurements "
                "on similar code)"
            )
        blocks.append(
            f"### Candidate {c.variant_index}\n"
            f"cycles={c.cycles} ({c.cycle_source}), on_chip={c.on_chip} bytes\n\n"
            "Composed source:\n"
            "```\n"
            f"{c.composed_source}\n"
            "```\n\n"
            f"Curated calibration evidence (K={len(c.curated)}):\n"
            f"{curated_block}\n"
        )
    return (
        f"## Final-pick context\n\n"
        f"root_path={root_path}, kernel={kernel}, preset={preset}\n\n"
        f"## Root Pareto candidates (N={len(candidates)})\n\n"
        + "\n".join(blocks)
        + f"\n## Pick exactly one variant_index in [0, {len(candidates) - 1}].\n"
    )


def parse_final_pick_response(
    response_text: str,
    *,
    num_candidates: int,
) -> tuple[int, str]:
    """Extract ``(variant_index, reason)`` from a final-pick-agent reply.

    Fails loudly (AssertionError) on missing fence, wrong JSON shape,
    out-of-range index, or non-integer index. The caller
    (``runtime.final_pick(strategy="agent")``) catches at the call-site
    boundary and falls back to the ``min_cycles`` deterministic pick.
    """
    import json as _json

    assert num_candidates >= 1, (
        f"parse_final_pick_response: num_candidates must be >= 1, got "
        f"{num_candidates!r}"
    )
    body = _extract_fenced(response_text, "json")
    assert body is not None, (
        "parse_final_pick_response: response must contain a fenced ```json "
        "block; none found. Raw response:\n" + response_text
    )
    parsed = _json.loads(body)
    assert isinstance(parsed, dict), (
        f"parse_final_pick_response: top-level JSON must be an object, got "
        f"{type(parsed).__name__}"
    )
    assert "variant_index" in parsed and "reason" in parsed, (
        f"parse_final_pick_response: JSON must carry both 'variant_index' "
        f"and 'reason' keys; got {sorted(parsed.keys())!r}"
    )
    idx = parsed["variant_index"]
    reason = parsed["reason"]
    assert isinstance(idx, int) and not isinstance(idx, bool), (
        f"parse_final_pick_response: 'variant_index' must be an int, got "
        f"{type(idx).__name__}={idx!r}"
    )
    assert 0 <= idx < num_candidates, (
        f"parse_final_pick_response: 'variant_index' must be in [0, "
        f"{num_candidates - 1}], got {idx!r}"
    )
    assert isinstance(reason, str), (
        f"parse_final_pick_response: 'reason' must be a string, got "
        f"{type(reason).__name__}={reason!r}"
    )
    return idx, reason.strip()
