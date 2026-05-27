import asyncio
from types import SimpleNamespace

import pytest

from src.agents import (
    AgentPromptTokenBudget,
    _collect_stream_into_completion,
    compute_dynamic_max_tokens,
    compute_prompt_token_budget,
    estimate_agent_prompt_tokens,
    retry_max_tokens_after_context_overflow,
)


def test_dynamic_max_tokens_uses_context_minus_prompt_and_margin():
    agent = SimpleNamespace(instructions="system instructions")
    conversation = [{"role": "user", "content": "short prompt"}]

    prompt_tokens = estimate_agent_prompt_tokens(agent, conversation)
    max_tokens = compute_dynamic_max_tokens(
        agent, conversation,
        context_window_tokens=100_000,
        output_token_margin=4_096,
    )

    assert max_tokens == 100_000 - prompt_tokens - 4_096
    assert max_tokens > 32_000


def test_dynamic_max_tokens_asserts_when_prompt_leaves_no_output_room():
    agent = SimpleNamespace(instructions="")
    conversation = [{"role": "user", "content": "x " * 200}]

    with pytest.raises(AssertionError) as exc_info:
        compute_dynamic_max_tokens(
            agent, conversation,
            context_window_tokens=64,
            output_token_margin=32,
        )

    assert "prompt leaves no room" in str(exc_info.value)


def test_prompt_token_budget_reports_available_output_room():
    agent = SimpleNamespace(instructions="")
    conversation = [{"role": "user", "content": "x " * 200}]

    budget = compute_prompt_token_budget(
        agent,
        conversation,
        context_window_tokens=64,
        output_token_margin=32,
    )

    assert isinstance(budget, AgentPromptTokenBudget)
    assert budget.prompt_tokens > 64
    assert budget.max_tokens < 0
    assert not budget.has_output_room


def test_context_overflow_retry_reduces_requested_output_budget():
    message = (
        "This endpoint's maximum context length is 131072 tokens. However, "
        "you requested about 131382 tokens (78837 of text input, 52545 in "
        "the output)."
    )

    retry_max_tokens = retry_max_tokens_after_context_overflow(
        message, current_max_tokens=52_545, retry_margin_tokens=1_024)

    assert retry_max_tokens == 52_545 - (131_382 - 131_072) - 1_024


def test_context_overflow_retry_ignores_unrelated_bad_request():
    retry_max_tokens = retry_max_tokens_after_context_overflow(
        "model does not support this parameter",
        current_max_tokens=10_000,
        retry_margin_tokens=1_024,
    )

    assert retry_max_tokens is None


def test_stream_error_finish_reason_becomes_failed_turn_marker():
    async def stream():
        yield SimpleNamespace(
            id="chatcmpl-test",
            model="test-model",
            created=1,
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role="assistant",
                        content="```python\npartial_but_invalid()",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                    finish_reason="error",
                )
            ],
        )

    completion = asyncio.run(_collect_stream_into_completion(stream()))

    choice = completion.choices[0]
    assert choice.finish_reason == "stop"
    assert "LLM_STREAM_ERROR" in choice.message.content
    assert "partial_but_invalid" not in choice.message.content
