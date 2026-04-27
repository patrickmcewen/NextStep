"""Agent definitions for StepGenFlow4.

Creates agents for lowering passes, translator passes, the Writer, and the Analyst.
Uses plain OpenAI chat completions (no native tool calling) since the vLLM
deployment doesn't have --enable-auto-tool-choice.
"""

from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel
from openai.types.chat import ChatCompletion
from openai.types.shared import Reasoning

from src.prompts import (build_pass_system_prompt, build_judge_system_prompt,
                          build_autotune_system_prompt)


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
                    few_shot_examples=None) -> Agent:
    """Create an agent for any pass (lowering or translator).

    ``few_shot_examples`` is an optional list of resolved example dicts (see
    ``resolve_few_shot_examples``); only passes whose template contains
    ``{few_shot_examples}`` consume them.
    """
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
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


_DIAGNOSTICIAN_SYSTEM_PROMPT = """\
You diagnose failed runs of a STeP DSL code-generation pipeline.

A run consists of a sequence of refactoring passes (refactor_load, \
refactor_compute, refactor_shape, refactor_final, ...). Each pass is an \
LLM-driven loop of up to ~11 turns; each turn proposes code, runs a \
correctness check, and gets feedback. You will receive a structured \
summary of every turn that ran in this outer iteration: pass name, turn \
index, status, a tail of the model's chain-of-thought reasoning (when \
present), the head of the correctness/error output, and the tail of the \
per-op shape trace (when present).

Produce a thorough diagnosis. Use the reasoning excerpts to distinguish \
*conceptual* mistakes (the model held a wrong mental model) from \
*mechanical* mistakes (it knew the right approach but miscoded the args), \
and to spot cases where the model recognized the problem but pushed forward \
anyway. Cite specific evidence (turn numbers, op names, shapes, quoted \
phrases from reasoning).

Cover at least these four points, in this order:

1. **Blocking pass.** Which pass was the first to never reach PASS, and \
   how many turns did it spend stuck.
2. **Dominant error pattern.** Across the failing turns of that pass, \
   what error class recurs (stream-shape mismatch in op X, flatten/reshape \
   rank out of bounds, repeat_ref / expand_ref shape constraint violation, \
   a torch.stack divergence cascading from earlier shape corruption, etc.). \
   Be specific about which op and which shapes.
3. **Root-cause classification.** Pick one or more: (a) conceptual gap — \
   the model misunderstands a DSL invariant; cite which one. (b) feedback \
   blind spot — the model received feedback but didn't act on its \
   downstream implications. (c) cascade — an earlier op produced a wrong \
   shape and every later op inherits the corruption. (d) regression — the \
   model fixed one thing but broke another. Use the reasoning excerpts to \
   support the classification.
4. **Next-attempt recommendation.** One concrete change in approach for \
   the next outer iteration: a different decomposition, a new prompt hint, \
   a different sequence of ops, etc. Not vague advice.

Stream-shape invariants you should cite when relevant: every binary op \
(binary_mul, binary_add, binary_matmul, ...) requires *identical* stream \
shapes on both operands. flatten/reshape_stream rank arguments are bounded \
by stream-dim count. repeat_ref/expand_ref require x.stream == ref.stream \
modulo their documented relation. offchip_load's stride and out_shape_tiled \
together determine the output stream shape and must respect divisibility.

Length: as long as needed to be specific and useful — typically 3–6 short \
paragraphs (~300–800 words). Use the four numbered headings above. Plain \
markdown is fine. Do not restate the prompt. Do not hedge. If the trace \
and reasoning are empty (run died before any pass produced output), say \
so plainly in one sentence.
"""


def make_diagnostician_agent(llm_config: dict) -> Agent:
    """Create an agent that diagnoses a failed outer iteration."""
    client = make_client(llm_config)
    model = ReasoningAwareModel(model=llm_config["model"], openai_client=client)
    return Agent(
        name="StepDiagnostician",
        instructions=_DIAGNOSTICIAN_SYSTEM_PROMPT,
        model=model,
        model_settings=_build_model_settings(llm_config),
    )


