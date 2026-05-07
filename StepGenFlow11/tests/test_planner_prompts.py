from src.prompts import build_planner_system_prompt


def test_planner_system_prompt_includes_response_format():
    prompt = build_planner_system_prompt()
    assert "DECISION: leaf" in prompt
    assert "DECISION: split" in prompt
    assert "# child:" in prompt
    assert "# refactored parent" in prompt


def test_planner_system_prompt_states_natural_incentive():
    prompt = build_planner_system_prompt()
    assert "refactor pass fails" in prompt or "refactor pass succeeds" in prompt
    assert "re-invoked" in prompt or "re-plan" in prompt


def test_planner_system_prompt_states_guards():
    prompt = build_planner_system_prompt()
    assert "compose" in prompt.lower()
    assert "passthrough" in prompt.lower() or "real work" in prompt.lower()
    assert "smaller" in prompt.lower() or "monolith" in prompt.lower()


def test_planner_system_prompt_says_children_are_inspirational():
    prompt = build_planner_system_prompt()
    assert "few-shot" in prompt.lower() or "inspirational" in prompt.lower() \
        or "self-contained" in prompt.lower()
