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


def _load_pass1_system_prompt(
    *,
    child_blackbox_block: str,
    contract_block: str,
    dsl_code: str,
    few_shot_examples=None,
) -> str:
    """Render the Pass-1 system prompt template with caller-supplied blocks."""
    from src.prompts import _format_few_shot_examples, _STEP_DSL_PY
    template_path = _PROMPTS_DIR_AGENTS / _PASS1_SYSTEM_TEMPLATE
    assert template_path.exists(), f"Pass-1 system template not found: {template_path}"
    template = template_path.read_text()
    return template.format(
        child_blackbox_block=child_blackbox_block,
        contract_block=contract_block,
        dsl_code=dsl_code,
        few_shot_examples=_format_few_shot_examples(few_shot_examples or []),
    )


def _load_pass1_judge_prompt(
    *,
    child_blackbox_block: str,
    contract_block: str,
    function_signature: str,
) -> str:
    """Render the Pass-1 judge system prompt template with caller-supplied blocks.

    ``function_signature`` is the *exact* signature line the orchestrator
    will invoke (e.g. ``def tiled_reference(dims, tensors):`` for the root
    or ``def <node>(<arg_1>, ..., *, out_shapes, out_perms=None):`` for a
    non-root node). The judge enforces a literal match against it, which
    obviates a root/non-root branch in the template.
    """
    template_path = _PROMPTS_DIR_AGENTS / _PASS1_JUDGE_TEMPLATE
    assert template_path.exists(), f"Pass-1 judge template not found: {template_path}"
    template = template_path.read_text()
    return template.format(
        child_blackbox_block=child_blackbox_block,
        contract_block=contract_block,
        function_signature=function_signature,
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
    """Create an AsyncOpenAI client from config dict."""
    return AsyncOpenAI(
        base_url=llm_config["url"],
        api_key=llm_config["api_key"],
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
    child_blackbox_block: str,
    contract_block: str,
    few_shot_examples=None,
) -> Agent:
    """Create a Pass-1 refactor agent for a single planner node.

    ``child_blackbox_block`` is the rendered markdown block describing each
    child's callable signature (empty string for leaf nodes).
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
    child_blackbox_block: str,
    contract_block: str,
    function_signature: str,
) -> Agent:
    """Create a Pass-1 judge agent for a single planner node.

    Same block parameters as ``make_pass1_agent``, plus ``function_signature``:
    the exact signature line the candidate function must match (root nodes get
    ``def tiled_reference(dims, tensors):``; non-root nodes get the
    contract-derived ``def <node>(<arg_1>, ..., *, out_shapes, out_perms=None):``).
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    system_prompt = _load_pass1_judge_prompt(
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
