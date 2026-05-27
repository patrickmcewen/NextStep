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

import re
from dataclasses import dataclass

import tiktoken

from agents import (Agent, AsyncOpenAI, ModelSettings,
                    OpenAIChatCompletionsModel, RunConfig)
from agents.retry import ModelRetrySettings, RetryPolicyContext, retry_policies
from openai import APIError, APIStatusError, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as _ChatCompletionChoice
from openai.types.shared import Reasoning

from src.prompts import (build_pass_system_prompt, build_judge_system_prompt,
                         build_bundle_judge_system_prompt,
                         build_autotune_system_prompt,
                         build_planner_system_prompt)

_PASS1_SYSTEM_TEMPLATE = "refactor_pass1_system.txt"
_PASS1_JUDGE_TEMPLATE = "refactor_pass1_judge_system.txt"
_PROMPTS_DIR_AGENTS = __import__("pathlib").Path(__file__).resolve().parent.parent / "prompts"

_DEFAULT_CONTEXT_WINDOW_TOKENS = 131072
_DEFAULT_OUTPUT_TOKEN_MARGIN = 4000
_CONTEXT_OVERFLOW_RETRY_MARGIN_TOKENS = 1024
_TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
_CHAT_MESSAGE_OVERHEAD = 4
_CHAT_REPLY_PRIMER = 2
_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length is (?P<maximum>\d+) tokens.*?"
    r"requested about (?P<requested>\d+) tokens "
    r"\((?P<input>\d+) of text input, (?P<output>\d+) in the output\)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class AgentPromptTokenBudget:
    context_window_tokens: int
    prompt_tokens: int
    output_token_margin: int
    max_tokens: int

    @property
    def has_output_room(self) -> bool:
        return self.max_tokens > 0


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    assert isinstance(content, list), (
        f"message content must be str or list, got {type(content).__name__}")
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        else:
            assert isinstance(item, dict), (
                f"content list item must be str or dict, got {type(item).__name__}")
            assert "text" in item, f"content dict lacks text field: {item!r}"
            assert isinstance(item["text"], str), (
                f"content text must be str, got {type(item['text']).__name__}")
            parts.append(item["text"])
    return "\n".join(parts)


def _token_count(text: str) -> int:
    assert isinstance(text, str), f"expected str, got {type(text).__name__}"
    return len(_TOKEN_ENCODING.encode(text))


def estimate_agent_prompt_tokens(agent, input_items) -> int:
    """Estimate request input tokens for an Agents SDK call.

    The estimate includes static agent instructions plus the current
    conversation. It intentionally counts only text content; the StepGenFlow
    LLM calls here are text-only prompts.
    """
    instructions = getattr(agent, "instructions", None) or ""
    assert isinstance(instructions, str), (
        "dynamic token budgeting requires static string agent instructions")

    tokens = _token_count(instructions) + _CHAT_MESSAGE_OVERHEAD
    if isinstance(input_items, str):
        return tokens + _token_count(input_items) + _CHAT_MESSAGE_OVERHEAD + _CHAT_REPLY_PRIMER

    assert isinstance(input_items, list), (
        f"Runner input must be str or list, got {type(input_items).__name__}")
    for message in input_items:
        assert isinstance(message, dict), (
            f"conversation item must be dict, got {type(message).__name__}")
        assert "role" in message, f"conversation item lacks role: {message!r}"
        assert "content" in message, f"conversation item lacks content: {message!r}"
        role = message["role"]
        assert isinstance(role, str), f"message role must be str, got {type(role).__name__}"
        tokens += (
            _CHAT_MESSAGE_OVERHEAD
            + _token_count(role)
            + _token_count(_content_text(message["content"]))
        )
    return tokens + _CHAT_REPLY_PRIMER


def _agent_llm_config(agent) -> dict:
    config = getattr(agent, "__llm_config__", None)
    if config is None:
        return {}
    assert isinstance(config, dict), "agent.__llm_config__ must be a dict"
    return config


def compute_dynamic_max_tokens(
    agent,
    input_items,
    *,
    context_window_tokens: int | None = None,
    output_token_margin: int | None = None,
) -> int:
    budget = compute_prompt_token_budget(
        agent,
        input_items,
        context_window_tokens=context_window_tokens,
        output_token_margin=output_token_margin,
    )
    assert budget.max_tokens > 0, (
        "prompt leaves no room for output tokens after reserved margin: "
        f"context_window_tokens={budget.context_window_tokens}, "
        f"prompt_tokens={budget.prompt_tokens}, "
        f"output_token_margin={budget.output_token_margin}"
    )
    return budget.max_tokens


def compute_prompt_token_budget(
    agent,
    input_items,
    *,
    context_window_tokens: int | None = None,
    output_token_margin: int | None = None,
) -> AgentPromptTokenBudget:
    config = _agent_llm_config(agent)
    if context_window_tokens is None:
        context_window_tokens = int(config.get(
            "context_window_tokens",
            config.get("context_length", _DEFAULT_CONTEXT_WINDOW_TOKENS),
        ))
    if output_token_margin is None:
        output_token_margin = int(config.get(
            "output_token_margin", _DEFAULT_OUTPUT_TOKEN_MARGIN))

    prompt_tokens = estimate_agent_prompt_tokens(agent, input_items)
    max_tokens = context_window_tokens - prompt_tokens - output_token_margin
    return AgentPromptTokenBudget(
        context_window_tokens=context_window_tokens,
        prompt_tokens=prompt_tokens,
        output_token_margin=output_token_margin,
        max_tokens=max_tokens,
    )


def build_dynamic_run_config(agent, input_items) -> RunConfig:
    return RunConfig(
        model_settings=ModelSettings(
            max_tokens=compute_dynamic_max_tokens(agent, input_items)))


def retry_max_tokens_after_context_overflow(
    error_message: str,
    current_max_tokens: int | None,
    *,
    retry_margin_tokens: int = _CONTEXT_OVERFLOW_RETRY_MARGIN_TOKENS,
) -> int | None:
    """Return a lower output budget for provider context-overflow 400s."""
    assert isinstance(error_message, str), (
        f"error_message must be str, got {type(error_message).__name__}")
    assert retry_margin_tokens > 0, (
        f"retry_margin_tokens must be positive, got {retry_margin_tokens}")
    match = _CONTEXT_OVERFLOW_RE.search(error_message)
    if match is None or current_max_tokens is None:
        return None
    maximum = int(match.group("maximum"))
    requested = int(match.group("requested"))
    requested_output = int(match.group("output"))
    overflow = requested - maximum
    assert requested_output > 0, (
        f"context overflow error reported non-positive output tokens: "
        f"{requested_output}")
    if overflow <= 0:
        return None
    next_max_tokens = (
        min(current_max_tokens, requested_output) - overflow - retry_margin_tokens
    )
    if next_max_tokens <= 0 or next_max_tokens >= current_max_tokens:
        return None
    return next_max_tokens


def _model_settings_from_fetch_args(args, kwargs) -> ModelSettings:
    if "model_settings" in kwargs:
        model_settings = kwargs["model_settings"]
    else:
        assert len(args) >= 3, "_fetch_response args must include model_settings"
        model_settings = args[2]
    assert isinstance(model_settings, ModelSettings), (
        f"model_settings must be ModelSettings, got "
        f"{type(model_settings).__name__}")
    return model_settings


def _replace_model_settings_in_fetch_args(
    args, kwargs, model_settings: ModelSettings
):
    if "model_settings" in kwargs:
        kwargs = dict(kwargs)
        kwargs["model_settings"] = model_settings
        return args, kwargs
    args = list(args)
    assert len(args) >= 3, "_fetch_response args must include model_settings"
    args[2] = model_settings
    return tuple(args), kwargs


def _with_llm_config(agent: Agent, llm_config: dict) -> Agent:
    agent.__llm_config__ = llm_config
    return agent


# Addendum injected into the pass-1 system prompt when max-tile mode is on.
# Mirrors the assertion surface of src/step_dsl_max_tile.py so the model knows
# up front which dimensions are bounded, instead of finding out by assertion
# failure. The ``{max_tile}`` placeholders are filled with the active bound.
_MAX_TILE_ADDENDUM_TEMPLATE = """
## MAX-TILE MODE (max = {max_tile} per dim) — READ THIS BEFORE ANY DSL CALL

This pass runs against ``step_dsl_max_tile.py`` rather than ``step_dsl.py``.
The two files are identical except that ``step_dsl_max_tile.py`` adds
assertions capping every tile dim at ``MAX_TILE_ROW = {max_tile}`` and
``MAX_TILE_COL = {max_tile}``. The intent: keep tile sizes small so a
downstream autotuner can absorb stream dims into tiles via local rewrites
without restructuring your graph. Violating any bound fails loud with an
``AssertionError``.

**Loads — ``tile_row <= {max_tile}`` and ``tile_col <= {max_tile}``:**
``offchip_load``, ``offchip_load_ref``, ``dyn_offchip_load``,
``random_offchip_load``. Push every dimension beyond the cap into
``out_shape_tiled`` (or ``tensor_shape_tiled`` for ``dyn_offchip_load``). For
example, a ``(M, N)`` weight that you previously loaded as one big tile with
``out_shape_tiled=(1,)`` becomes ``out_shape_tiled=(M // {max_tile}, N // {max_tile})``
with ``tile_row=tile_col={max_tile}`` (or anything smaller).

**Stores — input tile must satisfy ``(tile_r, tile_c) <= ({max_tile}, {max_tile})``:**
``offchip_store``, ``random_offchip_store``. Do not merge stream dims past
the bound right before the store.

**Tile-growing reshapes — output tile dim must stay within bounds:**
``accum_retile_row`` (asserts ``final tile_row <= {max_tile}``),
``accum_retile_col`` (asserts ``final tile_col <= {max_tile}``),
``restream`` (asserts ``out_shape_tiled[-2:] within ({max_tile}, {max_tile})``).
If a reduction would grow a tile past the bound, either reduce ``rank``,
shrink the upstream stream dim, or use a non-tile-growing reduction
(``accum_add`` / ``accum_max`` / ``accum_mul``) which leaves the tile shape
alone.

**``retile_streamify`` — unconstrained.** It can only shrink a tile dim, so
the output stays within bounds whenever the input does.

**Child stub calls — every ``out_shapes`` entry's tile must be within
``({max_tile}, {max_tile})``:** when you invoke a child blackbox as
``child(..., out_shapes=((..., S, T_R, T_C), ...))``, each declared
``out_shape``'s last two dims (the child's tile dims) must satisfy
``T_R <= {max_tile}`` and ``T_C <= {max_tile}``. The contract recorded at the
call site asserts this in the same shot as the load/store/reshape ops above
— picking an oversize tile for a child fails loud at the first stub call,
not later when the child runs. The same bound applies to every on-chip
tensor you hand a stub: its tile must be within the cap by the time it
reaches the call site (your upstream DSL ops are what guarantee that).

**Contract block (non-root nodes) already respects the cap.** Any
``tiled shape declared by parent`` shown in the contract for an **on-chip**
input has tile dims within ``({max_tile}, {max_tile})``; the parent's DSL
already enforced it. **RAW** inputs in the contract are shown at *vanilla*
shape and are not bounded — you load them with ``offchip_load`` and pick
``tile_row, tile_col <= {max_tile}`` yourself. The ``Required output shapes``
line is likewise bounded — the parent could not have requested a shape
outside the cap.

**Matmul at small tiles.** ``binary_matmul`` on small tiles gives a small
output tile (output tile = ``(a_tile_r, b_tile_c)``); the cross-tile
reduction over the K dimension must come from a stream dim reduced with
``accum_add`` (or in one shot via ``binary_map_accum`` with ``rank=`` set to
the K stream rank). The seed_kernels GEMM pattern — load A and B with
matched K stream dims and broadcast strides, then ``binary_map_accum(A, B,
rank=1)`` — is the canonical max-tile shape; pick any tile sizes you like
within the cap.

**Reshape ops you should reach for.** ``reshape_stream`` to split / pad
stream dims, ``flatten`` to merge them, ``promote`` / ``promote_outer`` to
insert a singleton, ``expand_ref`` / ``repeat_static`` / ``repeat_ref`` to
broadcast over a stream dim, ``parallelize`` / ``static_reassemble`` to
interleave. These are all stream-level and unaffected by tile bounds; rely
on them.

**Why this constraint.** Mixing tile and stream representations forces the
implementation to bake tile choices into the graph structure (different
``retile_streamify`` placements, different matmul forms for tiled vs
single-tile inputs). Capping the tile collapses that decision space and lets
the autotuner be the thing that picks tile sizes — by *absorbing* stream
dims into tiles via ``accum_retile_*``, which is a local rewrite that does
not require restructuring the kernel.
"""


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
    dsl_types_code: str,
    few_shot_examples=None,
    max_tile: int | None = None,
) -> str:
    """Render the Pass-1 system prompt template with caller-supplied blocks.

    ``max_tile`` selects the max-tile DSL variant: when not None, the rendered
    ``dsl_code`` block should already be the ``step_dsl_max_tile.py`` source
    with its ``MAX_TILE_ROW`` / ``MAX_TILE_COL`` constants substituted to the
    chosen bound (caller's responsibility — see ``make_pass1_agent``). The
    prompt also gets the matching addendum injected up front so the model
    sees the bound stated explicitly before reading the assertion-heavy DSL.
    ``max_tile = 1`` recovers the strict 1x1 behavior; ``None`` means no bound.

    ``dsl_types_code`` is the source of ``src/step_dsl_types.py`` — the
    shared StepTensor/Tile/dtype-tag definitions that the ops module
    references. Rendered alongside ``dsl_code`` so the LLM sees the types,
    not just the ops that use them.
    """
    from src.prompts import _format_few_shot_examples
    template_path = _PROMPTS_DIR_AGENTS / _PASS1_SYSTEM_TEMPLATE
    assert template_path.exists(), f"Pass-1 system template not found: {template_path}"
    template = template_path.read_text()
    placeholders = _pass1_leaf_placeholders(
        is_leaf=is_leaf, child_blackbox_block=child_blackbox_block
    )
    addendum = (
        _MAX_TILE_ADDENDUM_TEMPLATE.format(max_tile=int(max_tile))
        if max_tile is not None else ""
    )
    return template.format(
        contract_block=contract_block,
        dsl_code=dsl_code,
        dsl_types_code=dsl_types_code,
        few_shot_examples=_format_few_shot_examples(few_shot_examples or []),
        max_tile_addendum=addendum,
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


def _retry_on_mid_stream_api_error(context: RetryPolicyContext) -> bool:
    """Retry policy predicate for SSE-injected provider errors.

    OpenRouter (and Anthropic's OpenAI-compat endpoint) returns an HTTP 200
    and starts streaming, then injects a ``data: {"error": ...}`` SSE chunk
    when their upstream translation fails — e.g. the literal message
    ``"JSON error injected into SSE stream"`` we've seen from OpenRouter.
    The OpenAI SDK surfaces that as ``openai.APIError`` with no status code
    (the original HTTP response was already 200). Built-in policies key off
    HTTP status (``http_status``) or transport state (``network_error``), so
    neither catches this case. We retry it explicitly.

    We intentionally exclude ``APIStatusError`` (HTTP 4xx/5xx) — those are
    handled by ``provider_suggested`` / ``http_status`` policies that respect
    retry-after headers. Letting both run via ``retry_policies.any(...)``
    would double-retry on rate limits.
    """
    err = context.error
    return isinstance(err, APIError) and not isinstance(err, APIStatusError)


_RETRY_SETTINGS = ModelRetrySettings(
    max_retries=3,
    policy=retry_policies.any(
        retry_policies.network_error(),
        _retry_on_mid_stream_api_error,
    ),
)


def _build_model_settings(llm_config: dict) -> ModelSettings:
    """Construct per-call ModelSettings from llm_config.

    `max_tokens` is set here as a baseline because OpenRouter applies a silent
    ~16k output cap when the caller omits it. Real Runner.run call sites
    overlay this with build_dynamic_run_config(), which computes a per-request
    budget from the current prompt length.

    `include_usage=True` is required for non-OpenAI providers: ReasoningAwareModel
    upgrades non-streaming calls to streaming under the hood (see its docstring),
    and the SDK only auto-sets stream_options.include_usage for openai.com hosts.
    Without this, usage info is dropped for Anthropic / OpenRouter.

    `retry` covers two transient failure modes the SDK's default does not:
    network errors during streaming, and SSE-injected provider errors that
    surface as APIError post-200 (see _retry_on_mid_stream_api_error).

    `reasoning_effort` is optional. If present in llm_config, it's passed
    to the provider as `reasoning_effort` (supported by OpenAI o-series,
    kimi-k2.*, deepseek-r1, etc.). Non-reasoning models ignore it.
    """
    effort = llm_config.get("reasoning_effort")
    if effort is None:
        return ModelSettings(
            max_tokens=100000,
            include_usage=True,
            retry=_RETRY_SETTINGS,
        )
    assert effort in _VALID_REASONING_EFFORTS, (
        f"reasoning_effort must be one of {sorted(_VALID_REASONING_EFFORTS)}, "
        f"got {effort!r}"
    )
    return ModelSettings(
        max_tokens=100000,
        include_usage=True,
        retry=_RETRY_SETTINGS,
        reasoning=Reasoning(effort=effort),
    )


class ReasoningAwareModel(OpenAIChatCompletionsModel):
    """Surfaces provider reasoning + forces streaming under the hood.

    Two jobs:

    1. Streaming upgrade. Callers (Runner.run -> get_response) request
       non-streaming responses, but for long generations Anthropic stalls
       and OpenRouter injects ``: OPENROUTER PROCESSING`` SSE keepalives
       that crash json.loads. We upgrade every non-streaming call to a
       streaming one and reassemble the chunks into a ChatCompletion of
       the same shape the SDK would otherwise produce. The native
       streaming path (Runner.run_streamed -> stream_response) is left
       untouched.

    2. Reasoning passthrough. OpenRouter reasoning models (kimi-k2.6,
       deepseek-r1, ...) return chain-of-thought in a top-level
       ``reasoning`` field; the stock SDK converter only looks at
       ``reasoning_content``, so we mirror it. We deliberately do NOT
       merge reasoning into ``content``: if a turn truncates mid-reasoning,
       leaving ``content`` empty lets the orchestrator's existing "no code
       block" branch catch it instead of treating raw CoT as the answer.
    """

    async def _fetch_response(self, *args, **kwargs):
        # Streaming path (Runner.run_streamed): pass through unchanged.
        if kwargs.get("stream"):
            return await super()._fetch_response(*args, **kwargs)

        # Non-streaming path: upgrade to streaming internally and reassemble.
        kwargs["stream"] = True
        try:
            _resp, stream = await super()._fetch_response(*args, **kwargs)
        except BadRequestError as exc:
            model_settings = _model_settings_from_fetch_args(args, kwargs)
            retry_max_tokens = retry_max_tokens_after_context_overflow(
                str(exc), model_settings.max_tokens)
            if retry_max_tokens is None:
                raise
            retry_settings = model_settings.resolve(
                ModelSettings(max_tokens=retry_max_tokens))
            args, kwargs = _replace_model_settings_in_fetch_args(
                args, kwargs, retry_settings)
            print(
                "[llm] context-overflow retry: "
                f"max_tokens {model_settings.max_tokens} -> {retry_max_tokens}"
            )
            _resp, stream = await super()._fetch_response(*args, **kwargs)
        completion = await _collect_stream_into_completion(stream)

        for choice in completion.choices:
            msg = choice.message
            reasoning = getattr(msg, "reasoning", None)
            if reasoning and not getattr(msg, "reasoning_content", None):
                setattr(msg, "reasoning_content", reasoning)
        usage = completion.usage.model_dump() if completion.usage else None
        print(f"[llm] finish={completion.choices[0].finish_reason} usage={usage}")
        return completion


async def _collect_stream_into_completion(stream) -> ChatCompletion:
    """Accumulate a chat-completions stream into a ChatCompletion.

    Assumes a single choice and no tool calls (no agent in this codebase
    declares ``tools=...``). Captures content, reasoning, finish_reason,
    and usage (which requires ``stream_options.include_usage`` — set via
    ``include_usage=True`` in _build_model_settings).
    """
    completion_id: str | None = None
    model: str = ""
    created: int = 0
    role: str = "assistant"
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage = None
    n_chunks = 0

    async for chunk in stream:
        n_chunks += 1
        if completion_id is None:
            completion_id = chunk.id
            model = chunk.model
            created = chunk.created
        if chunk.usage is not None:
            usage = chunk.usage
        assert len(chunk.choices) <= 1, "multi-choice streaming is not supported"
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        assert not delta.tool_calls, "tool_calls in streamed response are not supported"
        if delta.role:
            role = delta.role
        if delta.content:
            content_parts.append(delta.content)
        r = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if r:
            reasoning_parts.append(r)
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    assert completion_id is not None, "stream produced no chunks"

    # OpenRouter / Anthropic compat occasionally close a stream cleanly without
    # setting finish_reason on any chunk (no final "stop" chunk, just [DONE]).
    # Default to "stop" so the SDK's ChatCompletion validates. If we got zero
    # content + zero reasoning, log loudly — downstream's "no code block" branch
    # will treat the turn as a failure and trigger a retry/abandon as usual.
    content_len = sum(len(p) for p in content_parts)
    reasoning_len = sum(len(p) for p in reasoning_parts)
    if finish_reason is None:
        finish_reason = "stop"
        print(
            f"[llm] WARNING: stream had no finish_reason "
            f"(chunks={n_chunks}, content_chars={content_len}, "
            f"reasoning_chars={reasoning_len}, usage={'yes' if usage else 'no'}); "
            f"defaulting to 'stop'"
        )

    message = ChatCompletionMessage(
        role=role,
        content="".join(content_parts) or None,
    )
    if reasoning_parts:
        setattr(message, "reasoning", "".join(reasoning_parts))

    return ChatCompletion(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[_ChatCompletionChoice(
            index=0, message=message, finish_reason=finish_reason, logprobs=None,
        )],
        usage=usage,
    )


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

    return _with_llm_config(Agent(
        name=f"StepPass_{pass_name}",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


_AUTOTUNE_VARIANTS = {
    "general": "autotune_system.txt",
    "parallel": "autotune/parallel/autotune_parallel_system.txt",
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

    return _with_llm_config(Agent(
        name=f"StepAutotune_{variant}",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_judge_agent(llm_config: dict, pass_name: str) -> Agent:
    """Create a judge agent for format compliance checking."""
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_judge_system_prompt(pass_name)

    return _with_llm_config(Agent(
        name=f"StepJudge_{pass_name}",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_bundle_judge_agent(llm_config: dict, compliance: dict) -> Agent:
    """Bundle-mode judge: prompt is templated from the bundle's compliance config.

    Use in place of make_judge_agent(refactor_final) when running with a bundle
    so the judge checks against the abstraction's invented operator surface
    rather than the hard-coded step_dsl vocabulary.
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_bundle_judge_system_prompt(compliance)

    return _with_llm_config(Agent(
        name="StepJudge_bundle",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_pass1_agent(
    llm_config: dict,
    *,
    is_leaf: bool,
    child_blackbox_block: str,
    contract_block: str,
    few_shot_examples=None,
    max_tile: int | None = None,
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
    ``max_tile`` (when set) runs pass-1 against ``step_dsl_max_tile.py`` with
    its ``MAX_TILE_ROW`` / ``MAX_TILE_COL`` substituted to this value in the
    displayed source, and injects the matching addendum. Callers that enable
    this **must** also assign the constants on the live module and register
    it as ``sys.modules['step_dsl']`` before invoking the executor; see
    ``orchestrator._refactor_one_node_pass1``. ``max_tile = 1`` recovers the
    strict 1x1 behavior; ``None`` (default) uses stock ``step_dsl.py``.
    """
    from src.prompts import _STEP_DSL_PY, _STEP_DSL_MAX_TILE_PY, _STEP_DSL_TYPES_PY
    if max_tile is not None:
        assert isinstance(max_tile, int) and max_tile >= 1, (
            f"max_tile must be a positive int, got {max_tile!r}")
        dsl_path = _STEP_DSL_MAX_TILE_PY
    else:
        dsl_path = _STEP_DSL_PY
    assert dsl_path.exists(), f"DSL source not found: {dsl_path}"
    assert _STEP_DSL_TYPES_PY.exists(), (
        f"DSL types source not found: {_STEP_DSL_TYPES_PY}")
    dsl_code = dsl_path.read_text()
    if max_tile is not None:
        # Substitute the live MAX_TILE_ROW / MAX_TILE_COL values into the
        # source the LLM sees, so what it reads is what the runtime enforces.
        # Use rstrip+re-add of the trailing newline to make replacement robust
        # against future formatting drift.
        dsl_code = dsl_code.replace(
            "MAX_TILE_ROW = 1\n", f"MAX_TILE_ROW = {int(max_tile)}\n", 1
        ).replace(
            "MAX_TILE_COL = 1\n", f"MAX_TILE_COL = {int(max_tile)}\n", 1
        )
        assert f"MAX_TILE_ROW = {int(max_tile)}" in dsl_code, (
            "max_tile substitution failed: literal 'MAX_TILE_ROW = 1\\n' not "
            "found in step_dsl_max_tile.py source. Did the default change?")
    dsl_types_code = _STEP_DSL_TYPES_PY.read_text()

    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = _load_pass1_system_prompt(
        is_leaf=is_leaf,
        child_blackbox_block=child_blackbox_block,
        contract_block=contract_block,
        dsl_code=dsl_code,
        dsl_types_code=dsl_types_code,
        few_shot_examples=few_shot_examples,
        max_tile=max_tile,
    )
    return _with_llm_config(Agent(
        name="StepPass_pass1",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


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
    return _with_llm_config(Agent(
        name="StepJudge_pass1",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


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
    return _with_llm_config(Agent(
        name="StepAutotune2",
        instructions=system_prompt,
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_planner_agent(llm_config: dict) -> Agent:
    """Create the decomposition-planner agent.

    The planner is invoked once per tree node (and re-invoked with failure
    context on Phase 1 failures). Its system prompt is static.
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return _with_llm_config(Agent(
        name="StepPlanner",
        instructions=build_planner_system_prompt(),
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_curation_agent(llm_config: dict) -> Agent:
    """Create the autotune2 calibration-curation agent (PR4).

    Ranks past (analytical, rust) calibration records by relevance to a
    target composed source. Called as a sub-step of the in-loop
    simulation-decision agent; system prompt is static.

    The HANDOFF design notes this agent is meant to run on a cheaper
    model (Haiku-class) since it's called per-variant. Production
    callers can pass a separate ``llm_config`` profile pointed at the
    cheap model; for first-cut deployment the same profile as the
    main autotune2 agent is fine.
    """
    from src.autotune2.prompts import build_curation_system_prompt
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return _with_llm_config(Agent(
        name="StepAutotune2Curation",
        instructions=build_curation_system_prompt(),
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_ace_context_curator_agent(llm_config: dict) -> Agent:
    """Create the autotune2 ACE playbook curator agent."""
    from src.autotune2.prompts import build_ace_context_curator_system_prompt
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return _with_llm_config(Agent(
        name="StepAutotune2AceContextCurator",
        instructions=build_ace_context_curator_system_prompt(),
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_sim_decision_agent(llm_config: dict) -> Agent:
    """Create the autotune2 in-loop simulation-decision agent (PR4).

    Decides per-variant whether to spend the per-pass Rust-simulator
    budget on this variant. Wrapped by ``AgentManager`` in
    ``src/autotune2/sim_manager.py`` — see that wrapper's docstring for
    the call sequence and the fall-back-to-analytical-on-failure path.
    System prompt is static.
    """
    from src.autotune2.prompts import build_sim_decision_system_prompt
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return _with_llm_config(Agent(
        name="StepAutotune2SimDecision",
        instructions=build_sim_decision_system_prompt(),
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)


def make_final_pick_agent(llm_config: dict) -> Agent:
    """Create the autotune2 end-of-run final-pick agent (PR5).

    Picks one variant from the root Pareto front for the run's single
    ground-truth Rust evaluation. Wrapped by
    ``runtime.final_pick(strategy="agent")``; system prompt is static.
    Distinct from ``make_sim_decision_agent`` even though both look at
    cycles + curated calibration evidence — the in-loop decision is
    rust-vs-analytical per variant under a budget, the final pick is
    "which one variant is the best in this Pareto front" with no budget
    constraint and a fixed single Rust call.
    """
    from src.autotune2.prompts import build_final_pick_system_prompt
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return _with_llm_config(Agent(
        name="StepAutotune2FinalPick",
        instructions=build_final_pick_system_prompt(),
        model=model,
        model_settings=_build_model_settings(llm_config),
    ), llm_config)
