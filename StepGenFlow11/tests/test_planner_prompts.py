from src.prompts import build_planner_system_prompt, build_planner_user_prompt, build_replan_user_prompt


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


def test_planner_user_prompt_includes_reference_code_and_dims():
    ref = "import torch\nclass Model(torch.nn.Module): pass\n"
    prompt = build_planner_user_prompt(reference_code=ref, dims={"M": 4, "N": 8})
    assert "class Model" in prompt
    assert '"M": 4' in prompt
    assert '"N": 8' in prompt


def test_replan_user_prompt_includes_failure_context_and_siblings():
    ref = "import torch\nclass Model(torch.nn.Module): pass\n"
    prompt = build_replan_user_prompt(
        reference_code=ref,
        dims={"M": 4},
        replan_iteration=2,
        node_path="root/child_b",
        failing_node="root/child_b/grandchild_x",
        last_turn_messages=["err 1", "err 2", "err 3"],
        sibling_results=[
            ("root/child_b/grandchild_y", "def tiled_reference(...): pass"),
        ],
    )
    assert "RE-PLAN CONTEXT" in prompt
    assert "re-plan #2" in prompt
    assert "root/child_b/grandchild_x" in prompt
    assert "err 3" in prompt
    assert "root/child_b/grandchild_y" in prompt
    assert "tiled_reference" in prompt


def test_replan_user_prompt_with_no_sibling_results():
    ref = "import torch\nclass Model(torch.nn.Module): pass\n"
    prompt = build_replan_user_prompt(
        reference_code=ref, dims={"M": 4}, replan_iteration=1,
        node_path="root", failing_node="root/child_a",
        last_turn_messages=[], sibling_results=[],
    )
    assert "no successful siblings" in prompt.lower() \
        or "no sibling" in prompt.lower()
