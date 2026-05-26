from types import SimpleNamespace

import pytest

from src.agents import (
    AgentPromptTokenBudget,
    compute_dynamic_max_tokens,
    compute_prompt_token_budget,
    estimate_agent_prompt_tokens,
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
