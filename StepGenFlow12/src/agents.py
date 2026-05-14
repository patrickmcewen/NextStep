"""Agent factories for StepGenFlow.

Builds the LLM agents the orchestrator and autotuner consume:

  - ``make_pass_agent``         — refactor / translate pass agent (system prompt
                                  selected by pass name, with optional bundle
                                  prompt override and few-shot examples)
  - ``make_judge_agent``        — per-pass compliance judge
  - ``make_bundle_judge_agent`` — judge templated from a bundle's compliance
                                  config (bundle mode only)
  - ``make_autotune_agent``     — autotuner agent (general / parallel variant)

All factories use ``ReasoningAwareModel`` so OpenRouter reasoning models
surface chain-of-thought as structured ``ReasoningItem``s rather than
contaminating the response content.
"""

from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel
from openai.types.chat import ChatCompletion
from openai.types.shared import Reasoning

from src.prompts import (build_pass_system_prompt, build_judge_system_prompt,
                         build_bundle_judge_system_prompt,
                         build_autotune_system_prompt,
                         build_planner_system_prompt)

_PASS1_SYSTEM_TEMPLATE = "refactor_pass1_system.txt"
_PASS1_JUDGE_TEMPLATE = "refactor_pass1_judge_system.txt"
_PROMPTS_DIR_AGENTS = __import__("pathlib").Path(__file__).resolve().parent.parent / "prompts"


_LEAF_NOTICE_BLOCK = (
    "## This Node Is a Leaf — No Child Blackboxes\n"
    "\n"
    "This planner node has **no children** to delegate to. The function body must be "
    "implemented end-to-end with the DSL operations listed below; do not invent or call "
    "any non-DSL helper. Every tensor operation must be a direct DSL call."
)

_NONLEAF_CALL_SITE_RULES_SECTION = (
    "## No Tensor-Method Transforms at Blackbox Call Sites\n"
    "\n"
    "When passing a tensor to a child blackbox, **no tensor-method transform may appear "
    "between the tensor source and the call site** — the stub recovers vanilla shape "
    "internally via ``flatten().reshape(vanilla_shape)`` and reshapes each raw reference "
    "output to the parent-declared ``out_shapes`` entry. Specifically:\n"
    "\n"
    "- **Allowed (single-output child):** `result = child_name(x, ..., out_shapes="
    "((S, T_R, T_C),))`\n"
    "- **Allowed (multi-output child):** `q, k, v = child_name(x, ..., out_shapes="
    "(s_q, s_k, s_v))`\n"
    "- **Forbidden:** `.reshape(...)`, `.permute(...)`, `.transpose(...)`, any index "
    "expression `[...]`, arithmetic (`*`, `+`, etc.), `.squeeze()`, `.unsqueeze()`, "
    "`.expand()`, `.flatten()` (any dim form)\n"
    "\n"
    "If you need to change a stream's shape *anywhere* in the function (call sites, "
    "intermediate values, return value), use the appropriate DSL op — `reshape_stream`, "
    "`reshape_pad_stream`, `streamify`, `flatten`, `bufferize`, `retile_streamify`, etc. "
    "Never use `tensor.reshape(...)` or any other ``torch.Tensor`` method.\n"
    "\n"
    "``out_shapes`` is **always a tuple of per-output entries**. Even single-output "
    "children take a 1-tuple (e.g. ``out_shapes=((S, T_R, T_C),)``). The stub only "
    "reshapes; if a child output needs a non-reshape layout change (e.g. moving the "
    "seq axis to the front), apply the permutation explicitly on the return via "
    "``bufferize`` + ``streamify``.\n"
    "\n"
)

_NONLEAF_JUDGE_REQ2_SECTION = (
    "## Requirement 2: Child blackbox call sites\n"
    "\n"
    "{child_blackbox_block}\n"
    "\n"
    "Calling these blackboxes is **optional** — they are tools the implementer may "
    "invoke. Do **not** flag any of the following:\n"
    "- A child blackbox that is never called in the function body.\n"
    "- A child blackbox called multiple times.\n"
    "- Calls placed inside loops or conditionals.\n"
    "\n"
    "The only thing to flag is what the implementer does to a tensor *between its source "
    "and the call*. When a blackbox IS called, flag any of the following applied to a "
    "tensor flowing into that call site:\n"
    "- `.permute(...)` or `.transpose(...)` or `.reshape(...)`\n"
    "- Index expressions `[...]` on a tensor flowing into the call (slicing or gathering "
    "— NOT a `for i, x in enumerate(...)` loop iterating over a static container, which "
    "is permitted)\n"
    "- Arithmetic operators (`*`, `+`, `-`, `/`, `@`)\n"
    "- `.squeeze()`, `.unsqueeze()`, `.expand()`, `.flatten()` (single-argument form that "
    "changes layout)\n"
    "\n"
    "Output permutations must be applied on the stub's return via DSL ops "
    "(``bufferize`` + ``streamify`` with proper strides), not by transforming the "
    "input before the call. For multi-output children, the call site must "
    "destructure the returned tuple (e.g. ``q, k, v = preprocess_heads(x, out_shapes="
    "(s_q, s_k, s_v))``).\n"
    "\n"
)


def _pass1_leaf_placeholders(*, is_leaf: bool, child_blackbox_block: str) -> dict:
    """Return the leaf/non-leaf placeholder values shared by the system and
    judge prompts. ``child_blackbox_block`` is the rendered child list (only
    consulted when ``is_leaf`` is False)."""
    if is_leaf:
        assert child_blackbox_block == "", (
            "_pass1_leaf_placeholders: leaf node must not carry a "
            f"child_blackbox_block; got {child_blackbox_block!r}"
        )
        return {
            "role_intro_tail": (
                "implementing the body end-to-end with DSL operations — "
                "this node has no children to delegate to."
            ),
            "dsl_only_phrase": (
                "every tensor operation must be a DSL call. **The only "
                "remaining Python should be scalar math, control flow, and "
                "list operations.**"
            ),
            "child_blackbox_block": _LEAF_NOTICE_BLOCK,
            "call_site_rules_section": "",
            "judge_enforcement_sentence": (
                "This rule is enforced by the judge. Any sub-rank-3 "
                "``out_shapes`` entry will result in `VERDICT: REJECT`."
            ),
            "load_source_tail": "",
            "offchip_subject_lead": "Off-chip values",
            "onchip_source_tail": "",
            "root_orchestrator_alt_clause": "",
        }
    assert child_blackbox_block != "", (
        "_pass1_leaf_placeholders: non-leaf node must supply a non-empty "
        "child_blackbox_block"
    )
    return {
        "role_intro_tail": "treating each child as an opaque blackbox callable.",
        "dsl_only_phrase": (
            "every tensor operation must be a DSL call OR a child blackbox "
            "call. **The only remaining Python should be scalar math, "
            "control flow, list operations, and blackbox calls.**"
        ),
        "child_blackbox_block": child_blackbox_block,
        "call_site_rules_section": _NONLEAF_CALL_SITE_RULES_SECTION,
        "judge_enforcement_sentence": (
            "This rule, together with the no-tensor-method rule above, is "
            "enforced by the judge. Any tensor-method transform between a "
            "source and a blackbox call site, or any sub-rank-3 "
            "``out_shapes`` entry, will result in `VERDICT: REJECT`."
        ),
        "load_source_tail": ", or a child blackbox's return value",
        "offchip_subject_lead": (
            "Both kinds of off-chip values **may** still be passed straight "
            "to a child blackbox without loading — the stub recovers vanilla "
            "shape internally. They"
        ),
        "onchip_source_tail": " or blackbox",
        "root_orchestrator_alt_clause": (
            " — a leaf computation or a pure orchestrator that threads "
            "tensors through child blackboxes"
        ),
    }


def _pass1_judge_placeholders(*, is_leaf: bool, child_blackbox_block: str) -> dict:
    """Placeholder values specific to the judge template (the count phrase and
    the renumbered Requirement 2 section)."""
    if is_leaf:
        return {
            "requirement_count_phrase": "three",
            "requirement_2_section": "",
            "out_shapes_req_number": "2",
            "wrapper_req_number": "3",
            "out_shapes_call_sites_phrase": (
                "at any DSL-op call sites that take ``out_shapes``"
            ),
        }
    return {
        "requirement_count_phrase": "four",
        "requirement_2_section": _NONLEAF_JUDGE_REQ2_SECTION.format(
            child_blackbox_block=child_blackbox_block
        ),
        "out_shapes_req_number": "3",
        "wrapper_req_number": "4",
        "out_shapes_call_sites_phrase": (
            "both at child-blackbox call sites and any DSL-op call sites "
            "that take ``out_shapes``"
        ),
    }


def _load_pass1_system_prompt(
    *,
    is_leaf: bool,
    child_blackbox_block: str,
    contract_block: str,
    dsl_code: str,
    few_shot_examples=None,
) -> str:
    """Render the Pass-1 system prompt template with caller-supplied blocks."""
    from src.prompts import _format_few_shot_examples
    template_path = _PROMPTS_DIR_AGENTS / _PASS1_SYSTEM_TEMPLATE
    assert template_path.exists(), f"Pass-1 system template not found: {template_path}"
    template = template_path.read_text()
    placeholders = _pass1_leaf_placeholders(
        is_leaf=is_leaf, child_blackbox_block=child_blackbox_block
    )
    return template.format(
        contract_block=contract_block,
        dsl_code=dsl_code,
        few_shot_examples=_format_few_shot_examples(few_shot_examples or []),
        **placeholders,
    )


def _load_pass1_judge_prompt(
    *,
    is_leaf: bool,
    child_blackbox_block: str,
    contract_block: str,
    function_signature: str,
) -> str:
    """Render the Pass-1 judge system prompt template with caller-supplied blocks.

    ``function_signature`` is the *exact* signature line the orchestrator
    will invoke (e.g. ``def tiled_reference(dims, tensors):`` for the root
    or ``def <node>(<arg_1>, ..., *, out_shapes):`` for a non-root node).
    The judge enforces a literal match against it, which obviates a
    root/non-root branch in the template.
    """
    template_path = _PROMPTS_DIR_AGENTS / _PASS1_JUDGE_TEMPLATE
    assert template_path.exists(), f"Pass-1 judge template not found: {template_path}"
    template = template_path.read_text()
    return template.format(
        contract_block=contract_block,
        function_signature=function_signature,
        **_pass1_judge_placeholders(
            is_leaf=is_leaf, child_blackbox_block=child_blackbox_block
        ),
    )


_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _build_model_settings(llm_config: dict) -> ModelSettings:
    """Construct per-call ModelSettings from llm_config.

    `max_tokens` is always set: OpenRouter applies a silent ~16k output
    cap when the caller omits it, which truncates reasoning-model turns
    mid-thought.

    `reasoning_effort` is optional. If present in llm_config, it's passed
    to the provider as `reasoning_effort` (supported by OpenAI o-series,
    kimi-k2.*, deepseek-r1, etc.). Non-reasoning models ignore it.
    """
    effort = llm_config.get("reasoning_effort")
    if effort is None:
        return ModelSettings(max_tokens=100000)
    assert effort in _VALID_REASONING_EFFORTS, (
        f"reasoning_effort must be one of {sorted(_VALID_REASONING_EFFORTS)}, "
        f"got {effort!r}"
    )
    return ModelSettings(max_tokens=100000, reasoning=Reasoning(effort=effort))


class ReasoningAwareModel(OpenAIChatCompletionsModel):
    """Surfaces provider reasoning traces as first-class reasoning items.

    OpenRouter reasoning models (e.g. moonshotai/kimi-k2.6, deepseek/deepseek-r1)
    return the chain-of-thought in a top-level `reasoning` field on the message.
    The stock agents-SDK converter at chatcmpl_converter.py:135 only looks for
    `reasoning_content`, so we copy it over. That causes the SDK to emit a
    ResponseReasoningItem which the orchestrator can log separately.

    We deliberately do NOT merge reasoning into `content`: if the model was
    truncated mid-reasoning, leaving `content` empty lets the orchestrator's
    existing "no code block" branch catch the failure instead of treating
    the raw chain-of-thought as the final answer.
    """

    async def _fetch_response(self, *args, **kwargs):
        result = await super()._fetch_response(*args, **kwargs)
        completion = result[0] if isinstance(result, tuple) else result
        if isinstance(completion, ChatCompletion):
            for choice in completion.choices:
                msg = choice.message
                reasoning = getattr(msg, "reasoning", None)
                if reasoning and not getattr(msg, "reasoning_content", None):
                    setattr(msg, "reasoning_content", reasoning)
            usage = completion.usage.model_dump() if completion.usage else None
            print(f"[llm] finish={completion.choices[0].finish_reason} usage={usage}")
        return result


def make_client(llm_config: dict) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from config dict.

    ``api_key`` is optional: local endpoints (e.g. a self-hosted gpt-oss
    server) don't require auth, and several configs omit the field
    entirely. Falls back to the literal string ``"None"`` because the
    AsyncOpenAI constructor rejects an empty/missing key but does not
    actually validate the value against the endpoint.
    """
    return AsyncOpenAI(
        base_url=llm_config["url"],
        api_key=llm_config.get("api_key") or "None",
        timeout=600,
    )


def make_pass_agent(llm_config: dict, pass_name: str,
                    few_shot_examples=None,
                    system_prompt_override: str | None = None) -> Agent:
    """Create an agent for any pass (lowering or translator).

    ``few_shot_examples`` is an optional list of resolved example dicts (see
    ``resolve_few_shot_examples``); only passes whose template contains
    ``{few_shot_examples}`` consume them.
    ``system_prompt_override`` replaces the default system prompt entirely
    when provided (used in bundle-dir mode to inject a user-supplied prompt).
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    if system_prompt_override is not None:
        system_prompt = system_prompt_override
    else:
        system_prompt = build_pass_system_prompt(
            pass_name, few_shot_examples=few_shot_examples)

    return Agent(
        name=f"StepPass_{pass_name}",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


_AUTOTUNE_VARIANTS = {
    "general": "autotune_system.txt",
    "parallel": "autotune_parallel_system.txt",
    "memory": "autotune_memory_system.txt",
}


def make_autotune_agent(llm_config: dict, hw_constraints: dict,
                        variant: str = "general") -> Agent:
    """Create an autotuner agent that rewrites build_graph() for performance.

    `variant` picks the system prompt:
      - "general":  tile/compute/par_dispatch knobs + larger rewrites
      - "parallel": inserts/retunes `Parallelize` / `StaticReassemble` only
    """
    assert variant in _AUTOTUNE_VARIANTS, (
        f"Unknown autotune variant: {variant!r}. "
        f"Known: {sorted(_AUTOTUNE_VARIANTS)}"
    )
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_autotune_system_prompt(
        hw_constraints, _AUTOTUNE_VARIANTS[variant])

    return Agent(
        name=f"StepAutotune_{variant}",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


def make_judge_agent(llm_config: dict, pass_name: str) -> Agent:
    """Create a judge agent for format compliance checking."""
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_judge_system_prompt(pass_name)

    return Agent(
        name=f"StepJudge_{pass_name}",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


def make_bundle_judge_agent(llm_config: dict, compliance: dict) -> Agent:
    """Bundle-mode judge: prompt is templated from the bundle's compliance config.

    Use in place of make_judge_agent(refactor_final) when running with a bundle
    so the judge checks against the abstraction's invented operator surface
    rather than the hard-coded step_dsl vocabulary.
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_bundle_judge_system_prompt(compliance)

    return Agent(
        name="StepJudge_bundle",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


def make_pass1_agent(
    llm_config: dict,
    *,
    is_leaf: bool,
    child_blackbox_block: str,
    contract_block: str,
    few_shot_examples=None,
) -> Agent:
    """Create a Pass-1 refactor agent for a single planner node.

    ``is_leaf`` selects the leaf vs non-leaf prompt variant: leaves render an
    explicit "no children" notice and strip every reference to child blackbox
    callables, so the LLM cannot hallucinate a delegate.
    ``child_blackbox_block`` is the rendered markdown block describing each
    child's callable signature (must be empty string when ``is_leaf`` is True).
    ``contract_block`` is the rendered markdown block describing the parent's
    declared input shapes and required output shape/permutation (empty string
    for the root node).
    ``few_shot_examples`` is an optional list of resolved example dicts (see
    ``resolve_few_shot_examples``).
    """
    from src.prompts import _STEP_DSL_PY
    assert _STEP_DSL_PY.exists(), f"step_dsl.py not found: {_STEP_DSL_PY}"
    dsl_code = _STEP_DSL_PY.read_text()

    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = _load_pass1_system_prompt(
        is_leaf=is_leaf,
        child_blackbox_block=child_blackbox_block,
        contract_block=contract_block,
        dsl_code=dsl_code,
        few_shot_examples=few_shot_examples,
    )
    return Agent(
        name="StepPass_pass1",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


def make_pass1_judge_agent(
    llm_config: dict,
    *,
    is_leaf: bool,
    child_blackbox_block: str,
    contract_block: str,
    function_signature: str,
) -> Agent:
    """Create a Pass-1 judge agent for a single planner node.

    Same block parameters as ``make_pass1_agent``, plus ``function_signature``:
    the exact signature line the candidate function must match (root nodes get
    ``def tiled_reference(dims, tensors):``; non-root nodes get the
    contract-derived ``def <node>(<arg_1>, ..., *, out_shapes):``).
    Leaves drop the call-site requirement entirely so the judge only checks
    signature and ``out_shapes`` rank.
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = _load_pass1_judge_prompt(
        is_leaf=is_leaf,
        child_blackbox_block=child_blackbox_block,
        contract_block=contract_block,
        function_signature=function_signature,
    )
    return Agent(
        name="StepJudge_pass1",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


def make_autotune2_agent(llm_config: dict, system_prompt: str) -> Agent:
    """Create an autotune2 agent with a caller-supplied system prompt.

    Autotune2's system prompt is built externally by
    ``src.autotune2.prompts.build_autotune2_system_prompt`` (self-contained,
    no pass-1 dependency at the prompt-text level). This factory wires
    that prompt into the same SDK ``Agent`` / model-settings shape that
    pass-1 uses so the per-turn conversation loop (``Runner.run``),
    feedback handling, token accounting, and gate cascade can all be
    reused unchanged.
    """
    assert system_prompt, "make_autotune2_agent: system_prompt must be non-empty"
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return Agent(
        name="StepAutotune2",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


def make_planner_agent(llm_config: dict) -> Agent:
    """Create the decomposition-planner agent.

    The planner is invoked once per tree node (and re-invoked with failure
    context on Phase 1 failures). Its system prompt is static.
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return Agent(
        name="StepPlanner",
        instructions=build_planner_system_prompt(),
        model=model,
        model_settings=_build_model_settings(llm_config),
    )
