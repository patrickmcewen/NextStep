"""Unit + integration tests for orchestrator gate helpers and _run_pass_loop.

These tests cover both `--check-order=correctness-first` (existing behavior;
parity contract) and `--check-order=compliance-first` (new ordering).
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src import orchestrator as orch_mod


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_log_capture():
    msgs = []
    def log(m):
        msgs.append(m)
    return log, msgs


def test_gate_correctness_pass(tmp_path, monkeypatch):
    """match=True returns _GateResult(None, 'PASS', 0) and writes correctness_result.txt."""
    def fake_check(code, kernel, dims, tensors):
        return "match=True\nmax_diff=0.0"

    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl", fake_check)

    log, _ = _make_log_capture()
    res, shape_trace = _run(orch_mod._gate_correctness(
        code="pass", kernel_name="k", dims={}, tensors={},
        executor="dsl", turn_dir=tmp_path, log=log,
    ))

    assert res.feedback is None
    assert res.status == "PASS"
    assert res.tokens == 0
    assert shape_trace == ""
    assert (tmp_path / "correctness_result.txt").read_text() == "match=True\nmax_diff=0.0"


def test_gate_correctness_mismatch(tmp_path, monkeypatch):
    def fake_check(code, kernel, dims, tensors):
        return "match=False\nmax_diff=0.5"

    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl", fake_check)

    log, _ = _make_log_capture()
    res, _ = _run(orch_mod._gate_correctness(
        code="pass", kernel_name="k", dims={}, tensors={},
        executor="dsl", turn_dir=tmp_path, log=log,
    ))
    assert res.feedback == "## Correctness check result\nmatch=False\nmax_diff=0.5"
    assert res.status == "FAIL: match=False"
    assert res.tokens == 0


def test_gate_correctness_exception(tmp_path, monkeypatch):
    def fake_check(code, kernel, dims, tensors):
        raise ValueError("kaboom")

    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl", fake_check)

    log, _ = _make_log_capture()
    res, _ = _run(orch_mod._gate_correctness(
        code="pass", kernel_name="k", dims={}, tensors={},
        executor="dsl", turn_dir=tmp_path, log=log,
    ))
    assert res.feedback.startswith("## Error running code\n")
    assert "ValueError: kaboom" in res.feedback
    assert res.status.startswith("FAIL:")
    written = (tmp_path / "correctness_result.txt").read_text()
    assert written.startswith("ERROR:\n")


_BANNED_REFACTOR_FINAL_CODE = "def f():\n    import torch\n    return torch.matmul(a, b)\n"
_CLEAN_CODE = "def f():\n    return 0\n"


def _stub_check_banned_ops(monkeypatch, violations):
    monkeypatch.setattr(orch_mod, "_check_banned_ops", lambda code, pn: list(violations))


def test_gate_compliance_pass(tmp_path, monkeypatch):
    _stub_check_banned_ops(monkeypatch, [])
    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_compliance(
        code=_CLEAN_CODE, pass_name="refactor_final",
        compliance_override=None, judge_agent=None, tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))
    assert res.feedback is None
    assert res.status == "PASS"
    assert res.tokens == 0


def test_gate_compliance_fail_correctness_verified_true(tmp_path, monkeypatch):
    """Existing behavior: status CORRECT_BUT_NONCOMPLIANT, preamble references PASS."""
    _stub_check_banned_ops(monkeypatch, ["torch.matmul: line 3"])
    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_compliance(
        code=_BANNED_REFACTOR_FINAL_CODE, pass_name="refactor_final",
        compliance_override=None, judge_agent=None, tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))
    assert res.status == "CORRECT_BUT_NONCOMPLIANT"
    assert res.feedback.startswith("## Correctness: PASS\n\nYour code produces the correct output, but still contains disallowed operations:\n\n")
    assert "torch.matmul: line 3" in res.feedback
    assert "Replace these with the corresponding DSL function calls" in res.feedback


def test_gate_compliance_fail_correctness_verified_false(tmp_path, monkeypatch):
    """New behavior under compliance-first: status NONCOMPLIANT, preamble does not claim correctness."""
    _stub_check_banned_ops(monkeypatch, ["torch.matmul: line 3"])
    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_compliance(
        code=_BANNED_REFACTOR_FINAL_CODE, pass_name="refactor_final",
        compliance_override=None, judge_agent=None, tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=False,
    ))
    assert res.status == "NONCOMPLIANT"
    assert res.feedback.startswith("## Compliance check FAILED\n\nYour code uses disallowed operations:\n\n")
    assert "torch.matmul: line 3" in res.feedback


def test_gate_compliance_translation_pass_uses_step_hint(tmp_path, monkeypatch):
    _stub_check_banned_ops(monkeypatch, ["foo"])
    monkeypatch.setattr(orch_mod, "_TRANSLATION_PASSES", {"translate"})
    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_compliance(
        code="x", pass_name="translate",
        compliance_override=None, judge_agent=None, tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))
    assert "Replace these with the corresponding STeP operations." in res.feedback


def test_gate_compliance_refactor_final_carveout_runs_judge(tmp_path, monkeypatch):
    _stub_check_banned_ops(monkeypatch, ["torch.matmul: line 3"])

    captured_ctx = {}
    async def fake_run_judge(judge_agent, code, turn_dir, log, *, context):
        captured_ctx["ctx"] = context
        return ("Issue: foo at line 5", 42)

    monkeypatch.setattr(orch_mod, "_run_judge", fake_run_judge)
    judge_agent = object()
    log, _ = _make_log_capture()

    res = _run(orch_mod._gate_compliance(
        code=_BANNED_REFACTOR_FINAL_CODE, pass_name="refactor_final",
        compliance_override=None, judge_agent=judge_agent, tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))

    assert res.status == "CORRECT_BUT_NONCOMPLIANT"
    assert res.tokens == 42
    assert "## Judge feedback (line-specific):" in res.feedback
    assert "Issue: foo at line 5" in res.feedback
    assert "## Correctness status\nThis code has ALREADY been executed" in captured_ctx["ctx"]


def test_gate_compliance_translation_pass_skips_carveout(tmp_path, monkeypatch):
    """The carve-out is refactor_final-only — translation passes don't trigger it."""
    _stub_check_banned_ops(monkeypatch, ["foo"])
    monkeypatch.setattr(orch_mod, "_TRANSLATION_PASSES", {"translate"})

    async def boom(*a, **kw):
        raise AssertionError("judge should not run on translate")
    monkeypatch.setattr(orch_mod, "_run_judge", boom)

    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_compliance(
        code="x", pass_name="translate",
        compliance_override=None, judge_agent=object(), tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))
    assert res.status == "CORRECT_BUT_NONCOMPLIANT"
    assert res.tokens == 0
    assert "## Judge feedback" not in res.feedback


def test_gate_judge_no_agent(tmp_path):
    """No judge agent => immediate pass with zero tokens."""
    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_judge(
        judge_agent=None, code="x", tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))
    assert res == orch_mod._GateResult(None, "PASS", 0)


def test_gate_judge_pass_correctness_verified_true(tmp_path, monkeypatch):
    captured = {}
    async def fake_run_judge(judge_agent, code, turn_dir, log, *, context):
        captured["ctx"] = context
        return (None, 17)

    monkeypatch.setattr(orch_mod, "_run_judge", fake_run_judge)

    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_judge(
        judge_agent=object(), code="x", tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))
    assert res.feedback is None
    assert res.status == "PASS"
    assert res.tokens == 17
    assert "ALREADY been executed" in captured["ctx"]


def test_gate_judge_fail_correctness_verified_true(tmp_path, monkeypatch):
    async def fake_run_judge(judge_agent, code, turn_dir, log, *, context):
        return ("VIOLATIONS:\n- bad shape", 9)
    monkeypatch.setattr(orch_mod, "_run_judge", fake_run_judge)

    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_judge(
        judge_agent=object(), code="x", tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=True,
    ))
    assert res.status == "CORRECT_BUT_JUDGE_REJECTED"
    assert res.tokens == 9
    assert res.feedback.startswith(
        "## Correctness: PASS\n\nYour code produces the correct output and uses allowed operations, "
        "but does not follow canonical form:\n\n"
    )
    assert "VIOLATIONS:\n- bad shape" in res.feedback
    assert res.feedback.endswith("Fix these structural issues while keeping the output correct.")


def test_gate_judge_fail_correctness_verified_false(tmp_path, monkeypatch):
    async def fake_run_judge(judge_agent, code, turn_dir, log, *, context):
        assert "NOT yet been executed" in context, "judge ctx must reflect not-yet-verified"
        return ("VIOLATIONS:\n- bad shape", 9)
    monkeypatch.setattr(orch_mod, "_run_judge", fake_run_judge)

    log, _ = _make_log_capture()
    res = _run(orch_mod._gate_judge(
        judge_agent=object(), code="x", tensors=None,
        turn_dir=tmp_path, log=log, correctness_verified=False,
    ))
    assert res.status == "JUDGE_REJECTED"
    assert res.tokens == 9
    assert res.feedback.startswith("## Judge rejected\n\nYour code does not follow canonical form:\n\n")
    assert "VIOLATIONS:\n- bad shape" in res.feedback
    assert res.feedback.endswith("Fix these structural issues.")


def test_gate_post_validator_none(tmp_path):
    """No post_validator => pass."""
    log, _ = _make_log_capture()
    res = orch_mod._gate_post_validator(None, "code", tmp_path, log)
    assert res == orch_mod._GateResult(None, "PASS", 0)


def test_gate_post_validator_pass(tmp_path):
    log, _ = _make_log_capture()
    def pv(code, td):
        return None
    res = orch_mod._gate_post_validator(pv, "code", tmp_path, log)
    assert res == orch_mod._GateResult(None, "PASS", 0)


def test_gate_post_validator_fail(tmp_path):
    log, _ = _make_log_capture()
    def pv(code, td):
        return "## Translation failed\n..."
    res = orch_mod._gate_post_validator(pv, "code", tmp_path, log)
    assert res.feedback == "## Translation failed\n..."
    assert res.status == "CORRECT_BUT_POST_VALIDATOR_REJECTED"
    assert res.tokens == 0


# ---------------------------------------------------------------------------
# _run_pass_loop integration tests
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens

class _FakeContextWrapper:
    def __init__(self, total_tokens):
        self.usage = _FakeUsage(total_tokens)

class _FakeRunResult:
    def __init__(self, text, total_tokens=0):
        self.final_output = text
        self.context_wrapper = _FakeContextWrapper(total_tokens)
        self.new_items = []  # required by _reasoning_text


class _FakeAgent:
    """Minimal agent stub — only .instructions is required by _run_pass_loop."""
    instructions = "# fake system prompt"


def _stub_runner_run(monkeypatch, responses):
    """Patch Runner.run to yield the next response from `responses` per call."""
    queue = list(responses)
    async def fake_run(agent, conversation):
        return _FakeRunResult(queue.pop(0))
    monkeypatch.setattr(orch_mod.Runner, "run", fake_run)


def _stub_pass_loop_infra(monkeypatch):
    """Patch build_pass_user_prompt so _run_pass_loop doesn't need a real StepDB."""
    import src.orchestrator as _o
    monkeypatch.setattr(_o, "build_pass_user_prompt",
                        lambda *a, **kw: "## fake user prompt")


def test_run_pass_loop_correctness_first_clean_pass(tmp_path, monkeypatch):
    """Smoke test: _run_pass_loop under correctness-first; one turn, all gates pass."""
    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl",
                        lambda c, k, d, t: "match=True\nmax_diff=0.0")
    monkeypatch.setattr(orch_mod, "_check_banned_ops", lambda code, pn: [])
    _stub_pass_loop_infra(monkeypatch)
    _stub_runner_run(monkeypatch, ["```python\ndef tiled_reference(d, t): return 0\n```"])

    log, _ = _make_log_capture()
    out = _run(orch_mod._run_pass_loop(
        agent=_FakeAgent(), pass_name="refactor_final", kernel_name="k",
        dims={}, max_turns=2, ckpt_dir=tmp_path,
        executor="dsl", tensors=None, log=log,
        judge_agent=None, post_validator=None, compliance_override=None,
        check_order="correctness-first",
    ))

    assert out["success"] is True
    turn0 = tmp_path / "refactor_final" / "turn_0"
    assert (turn0 / "status.txt").read_text() == "PASS"


def test_run_pass_loop_correctness_first_compliance_fail_then_pass(tmp_path, monkeypatch):
    """Compliance fails on turn 0, agent fixes it on turn 1."""
    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl",
                        lambda c, k, d, t: "match=True\nmax_diff=0.0")
    state = {"violations": ["torch.matmul: line 1"]}
    monkeypatch.setattr(orch_mod, "_check_banned_ops",
                        lambda code, pn: state["violations"])
    _stub_pass_loop_infra(monkeypatch)

    def runner_responses():
        yield "```python\nbad\n```"
        state["violations"] = []  # second turn: clean
        yield "```python\ngood\n```"
    gen = runner_responses()
    async def fake_run(agent, conv):
        return _FakeRunResult(next(gen))
    monkeypatch.setattr(orch_mod.Runner, "run", fake_run)

    log, _ = _make_log_capture()
    out = _run(orch_mod._run_pass_loop(
        agent=_FakeAgent(), pass_name="refactor_final", kernel_name="k",
        dims={}, max_turns=3, ckpt_dir=tmp_path,
        executor="dsl", tensors=None, log=log,
        judge_agent=None, post_validator=None, compliance_override=None,
        check_order="correctness-first",
    ))
    assert out["success"] is True
    assert (tmp_path / "refactor_final" / "turn_0" / "status.txt").read_text() == "CORRECT_BUT_NONCOMPLIANT"
    assert (tmp_path / "refactor_final" / "turn_1" / "status.txt").read_text() == "PASS"


def test_run_pass_loop_compliance_first_compliance_short_circuits_correctness(tmp_path, monkeypatch):
    """Under compliance-first, a banned-op turn is rejected before correctness runs."""
    correctness_calls = []
    def boom(code, kernel, dims, tensors):
        correctness_calls.append(1)
        return "match=True"
    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl", boom)

    state = {"violations": ["torch.matmul: line 1"]}
    monkeypatch.setattr(orch_mod, "_check_banned_ops",
                        lambda code, pn: state["violations"])

    _stub_pass_loop_infra(monkeypatch)

    def gen():
        yield "```python\nbad\n```"
        state["violations"] = []
        yield "```python\ngood\n```"
    g = gen()
    async def fake_run(agent, conv):
        return _FakeRunResult(next(g))
    monkeypatch.setattr(orch_mod.Runner, "run", fake_run)

    log, _ = _make_log_capture()
    out = _run(orch_mod._run_pass_loop(
        agent=_FakeAgent(), pass_name="refactor_final", kernel_name="k",
        dims={}, max_turns=3, ckpt_dir=tmp_path,
        executor="dsl", tensors=None, log=log,
        judge_agent=None, post_validator=None, compliance_override=None,
        check_order="compliance-first",
    ))

    assert out["success"] is True
    # Turn 0: compliance fail short-circuits — correctness must NOT have been called.
    turn0_status = (tmp_path / "refactor_final" / "turn_0" / "status.txt").read_text()
    assert turn0_status == "NONCOMPLIANT", f"got {turn0_status!r}"
    # No correctness_result.txt because correctness never ran.
    assert not (tmp_path / "refactor_final" / "turn_0" / "correctness_result.txt").exists()
    # Turn 1 (clean) ran correctness exactly once.
    assert correctness_calls == [1]


def test_run_pass_loop_compliance_first_judge_runs_with_not_yet_verified_ctx(tmp_path, monkeypatch):
    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl",
                        lambda c, k, d, t: "match=True")
    monkeypatch.setattr(orch_mod, "_check_banned_ops", lambda code, pn: [])

    captured = {}
    async def fake_run_judge(judge_agent, code, turn_dir, log, *, context):
        captured.setdefault("ctxs", []).append(context)
        # First turn: judge rejects. Second turn: judge approves.
        if len(captured["ctxs"]) == 1:
            return ("VIOLATIONS:\n- bad", 5)
        return (None, 5)
    monkeypatch.setattr(orch_mod, "_run_judge", fake_run_judge)

    _stub_pass_loop_infra(monkeypatch)
    _stub_runner_run(monkeypatch, ["```python\nx\n```", "```python\ny\n```"])

    log, _ = _make_log_capture()
    out = _run(orch_mod._run_pass_loop(
        agent=_FakeAgent(), pass_name="refactor_final", kernel_name="k",
        dims={}, max_turns=3, ckpt_dir=tmp_path,
        executor="dsl", tensors=None, log=log,
        judge_agent=object(), post_validator=None, compliance_override=None,
        check_order="compliance-first",
    ))

    assert out["success"] is True
    assert (tmp_path / "refactor_final" / "turn_0" / "status.txt").read_text() == "JUDGE_REJECTED"
    # Both judge invocations must have used the not-yet-verified preamble.
    for ctx in captured["ctxs"]:
        assert "NOT yet been executed" in ctx


def test_run_outer_iteration_passes_check_order_to_run_pass_loop(tmp_path, monkeypatch):
    """Regression: _run_outer_iteration must forward check_order to _run_pass_loop.

    Previously the bare name `check_order` was referenced inside
    _run_outer_iteration without being in scope, causing a NameError at runtime.
    """
    captured = {}

    async def fake_run_pass_loop(*args, **kwargs):
        captured.setdefault("check_orders", []).append(kwargs.get("check_order"))
        return {"success": True, "code": "x", "total_tokens": 0}

    monkeypatch.setattr(orch_mod, "_run_pass_loop", fake_run_pass_loop)
    # Other plumbing _run_outer_iteration touches:
    monkeypatch.setattr(orch_mod, "_load_stepdb_config",
                        lambda: {"kernels": {"k": {"presets": {"p": {"dims": {}}}}}})
    monkeypatch.setattr(orch_mod, "precompute_tensors", lambda *a, **kw: {})

    pass_agents = {"refactor_final": object()}
    judge_agents = {"refactor_final": None}
    lowering_passes = [{"name": "refactor_final", "executor": "dsl"}]
    translator_passes = []

    out = _run(orch_mod._run_outer_iteration(
        i=0, max_outer=1, outer_dir=tmp_path,
        kernel_name="k", dims={}, tensors={},
        pass_agents=pass_agents, judge_agents=judge_agents,
        max_turns=1, ckpt_root=tmp_path, preset="p",
        experience_dir="exp", llm_config={},
        lowering_passes=lowering_passes,
        translator_passes=translator_passes,
        resume_dsl_code=None,
        translator="llm",
        translate_fn=None,
        compliance_override=None,
        autotune_options=None,
        check_order="compliance-first",
    ))

    # The fake _run_pass_loop was called at least once and saw the right value.
    assert "check_orders" in captured, "_run_pass_loop was never called"
    assert captured["check_orders"][0] == "compliance-first", \
        f"expected 'compliance-first' propagated, got {captured['check_orders']!r}"


import torch
from unittest.mock import patch


def test_compare_against_gold_tuple_match():
    gold = (torch.zeros(4), torch.ones(4))
    result = (torch.zeros(4), torch.ones(4))
    with patch.object(orch_mod, "_get_gold", return_value=gold):
        out = orch_mod._compare_against_gold(result, kernel_name="k", dims={})
    assert "match=True" in out


def test_compare_against_gold_tuple_arity_mismatch():
    gold = (torch.zeros(4), torch.ones(4))
    result = (torch.zeros(4),)
    with patch.object(orch_mod, "_get_gold", return_value=gold):
        out = orch_mod._compare_against_gold(result, kernel_name="k", dims={})
    assert "match=False" in out
    assert "arity" in out.lower()


def test_compare_against_gold_tuple_element_mismatch():
    gold = (torch.zeros(4), torch.ones(4))
    result = (torch.zeros(4), torch.zeros(4))
    with patch.object(orch_mod, "_get_gold", return_value=gold):
        out = orch_mod._compare_against_gold(result, kernel_name="k", dims={})
    assert "match=False" in out
    assert "element[1]" in out or "element 1" in out


def test_compare_against_gold_tuple_vs_single_mismatch():
    gold = torch.zeros(4)
    result = (torch.zeros(4), torch.zeros(4))
    with patch.object(orch_mod, "_get_gold", return_value=gold):
        out = orch_mod._compare_against_gold(result, kernel_name="k", dims={})
    assert "match=False" in out
