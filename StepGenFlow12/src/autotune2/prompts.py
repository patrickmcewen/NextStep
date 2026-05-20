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
    child_picks:                  # parent prompts only
      attention_block: 3
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
``AutotuneResponse`` (child_picks dict, parent_input_contracts dict of
``TensorContract``, and the raw DSL source string). All schema
violations raise ``AssertionError`` with the offending fragment in the
message; the search driver routes the assertion text back into the
agent loop as a feedback message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from src.autotune2.contracts import TensorContract, vanilla_contract_for

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "autotune2_system.txt"
_SYSTEM_PROMPT_PATH_PARALLEL = _PROMPTS_DIR / "autotune2_system_parallel.txt"
_MEMORY_NOTES_PATH = _PROMPTS_DIR / "dsl_memory_notes.txt"
_TILE_SHRINK_FEWSHOT_PATH = _PROMPTS_DIR / "autotune_tile_shrink_fewshot.txt"
_PARALLEL_FEWSHOT_PATH = _PROMPTS_DIR / "autotune_parallel_fewshot.txt"
_STEP_DSL_MEMORY_PY = _PROJECT_ROOT / "src" / "step_dsl_memory.py"
_OUTPUT_PROTOCOL_PATHS = {
    True: _PROMPTS_DIR / "autotune2_output_protocol_leaf.txt",
    False: _PROMPTS_DIR / "autotune2_output_protocol_parent.txt",
}
_USER_PROMPT_PATHS = {
    "tile_shrink": _PROMPTS_DIR / "autotune2_user_tile_shrink.txt",
    "parallel": _PROMPTS_DIR / "autotune2_user_parallel.txt",
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
    """

    variant_index: int
    input_contracts: dict[str, TensorContract]
    output_contracts: dict[str, TensorContract]
    cycles: int
    on_chip: int


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
            f"    [{v.variant_index}] cycles={v.cycles}, on_chip={v.on_chip}"
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

      - ``"tile_shrink"`` (default): ``autotune2_system.txt`` +
        ``autotune_tile_shrink_fewshot.txt`` (placeholder
        ``{tile_shrink_fewshot}``). The load/consumer/reduction-order
        rewrite recipe.
      - ``"parallel"``: ``autotune2_system_parallel.txt`` +
        ``autotune_parallel_fewshot.txt`` (placeholder
        ``{parallel_fewshot}``). Shared vs. independent parallelism
        worked examples.

    Other placeholders are identical across variants: ``{step_dsl_code}``
    (the DSL surface, passed in), ``{memory_notes}`` (loaded from
    ``prompts/dsl_memory_notes.txt`` and shared with
    ``autotune_memory_system.txt``), and ``{output_protocol}`` (loaded
    from ``prompts/autotune2_output_protocol_{leaf,parent}.txt`` based on
    ``is_leaf``). ``str.replace`` is used instead of ``str.format``
    because the protocol fragments contain literal YAML braces.

    ``max_tile`` (when set) substitutes the same pass-1 max-tile
    addendum (``src.agents._MAX_TILE_ADDENDUM_TEMPLATE``) into the
    ``{max_tile_addendum}`` placeholder so the autotune2 LLM sees the
    same load/store/reshape/stub-call bounds it would see in pass-1.
    When ``None``, the placeholder collapses to an empty string.
    """
    assert dsl_code, "build_autotune2_system_prompt: dsl_code must be non-empty"
    assert fewshot in ("tile_shrink", "parallel"), (
        f"build_autotune2_system_prompt: fewshot must be 'tile_shrink' or "
        f"'parallel', got {fewshot!r}"
    )
    assert max_tile is None or (isinstance(max_tile, int) and max_tile >= 1), (
        f"build_autotune2_system_prompt: max_tile must be a positive int or "
        f"None, got {max_tile!r}"
    )
    if fewshot == "tile_shrink":
        system_path = _SYSTEM_PROMPT_PATH
        fewshot_path = _TILE_SHRINK_FEWSHOT_PATH
        fewshot_placeholder = "{tile_shrink_fewshot}"
    else:
        system_path = _SYSTEM_PROMPT_PATH_PARALLEL
        fewshot_path = _PARALLEL_FEWSHOT_PATH
        fewshot_placeholder = "{parallel_fewshot}"
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
    return (
        system_path.read_text()
        .replace("{step_dsl_code}", dsl_code)
        .replace("{memory_notes}", memory_notes)
        .replace(fewshot_placeholder, fewshot_path.read_text().rstrip())
        .replace("{output_protocol}", protocol_path.read_text().rstrip())
        .replace("{max_tile_addendum}", max_tile_addendum)
    )


_VARIANT_SECTION_TEMPLATE = """

### Child variant libraries

Pick exactly one ``variant_index`` per child via ``child_picks``. Each
table shows the per-arg classification, per-variant boundary contracts,
and Pareto coordinates the autotuner measured for that variant.

{variant_tables}
"""


_ACCEPTED_HINT = " shown above"
_PARENT_HINT_TEXT = (
    "\n  - Picks a child variant index for every child listed below."
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
    accepted_summary: str = "",
    budget_block: str = "",
) -> str:
    """Self-contained autotune2 user prompt for one (node, attempt) pair.

    ``fewshot`` selects which per-agent user-prompt template to load
    from ``prompts/`` — currently ``"tile_shrink"`` and ``"parallel"``,
    matching the system-prompt agents in
    ``build_autotune2_system_prompt``. Each template carries the same
    placeholders but specializes the "Your task" framing toward the
    agent's recipe (tile-shrink vs. parallelism).

    ``baseline_dsl`` is the DSL the LLM is asked to vary. For single-pass
    autotune2 this is always the pass-1 baseline; multi-pass / branching
    expansion supplies any prior library entry's DSL.

    ``accepted_summary`` is the empty string on the first attempt; on
    subsequent fresh attempts it is the rendered Pareto-front summary
    of already-accepted variants (see ``render_accepted_summary``) so
    the LLM can target gaps. ``child_variant_blocks`` is required for
    parents and forbidden for leaves. ``budget_block`` is the empty
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
        variant_section = ""
    else:
        assert child_variant_blocks, (
            "build_autotune2_user_prompt: parent prompts must include at "
            "least one child variant block"
        )
        variant_section = _VARIANT_SECTION_TEMPLATE.format(
            variant_tables="\n\n".join(child_variant_blocks.values())
        )
    accepted_section = (
        f"\n\n### Already-accepted variants for this node\n\n{accepted_summary}\n"
        if accepted_summary else ""
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
        budget_section=budget_block,
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
        lines.append(
            f"  [{idx}] cycles={entry.cycles}, on_chip={entry.on_chip}"
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

    ``child_picks`` is empty for leaf prompts. ``input_contracts`` uses
    TensorContract values (validated against any provided vanilla shape
    — see ``parse_*``). ``dsl`` is the raw DSL function source extracted
    from the response's python code block. Output contracts are NOT
    parsed from the LLM response — the verifier derives them from the
    built graph; see ``VerifyResult.derived_output_contracts``.
    """

    child_picks: dict[str, int] = field(default_factory=dict)
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
        "with the autotuner output spec (child_picks, parent_input_contracts); "
        "none found"
    )
    py_body = _extract_fenced(response_text, "python")
    assert py_body is not None, (
        "parse_autotune2_response: response must contain a fenced ```python "
        "block with the DSL function body; none found"
    )

    parsed = yaml.safe_load(yaml_body)
    assert isinstance(parsed, dict), (
        f"parse_autotune2_response: yaml block must be a mapping at top level, "
        f"got {type(parsed).__name__}"
    )

    expected_keys_parent = {"child_picks", "parent_input_contracts"}
    expected_keys_leaf = {"parent_input_contracts"}
    expected = expected_keys_leaf if is_leaf else expected_keys_parent
    unexpected = set(parsed.keys()) - expected
    missing = expected - set(parsed.keys())
    assert not unexpected, (
        f"parse_autotune2_response: unexpected yaml keys "
        f"{sorted(unexpected)!r}; allowed={sorted(expected)!r}. Note: "
        f"`parent_output_contracts` is no longer accepted — output contracts "
        f"are derived from the built graph."
    )
    assert not missing, (
        f"parse_autotune2_response: missing yaml keys {sorted(missing)!r}; "
        f"required={sorted(expected)!r}"
    )

    response = AutotuneResponse(dsl=py_body)

    if not is_leaf:
        cp = parsed["child_picks"]
        assert isinstance(cp, dict) and cp, (
            f"parse_autotune2_response: child_picks must be a non-empty mapping "
            f"of child_name -> variant_index, got {cp!r}"
        )
        if expected_child_names:
            expected_names = set(expected_child_names)
            assert set(cp.keys()) == expected_names, (
                f"parse_autotune2_response: child_picks keys {sorted(cp.keys())!r} "
                f"must equal expected_child_names {sorted(expected_names)!r}"
            )
        for name, idx in cp.items():
            assert isinstance(idx, int) and idx >= 0, (
                f"parse_autotune2_response: child_picks[{name!r}] must be a "
                f"non-negative int variant index, got {idx!r}"
            )
        response.child_picks = dict(cp)

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
