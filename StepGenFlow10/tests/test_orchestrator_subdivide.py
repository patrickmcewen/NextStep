"""End-to-end orchestrator test: a mock LLM that emits a directive on turn 0
and a tiled_reference on turn 1 must drive the whole flow correctly.

The test bypasses the real LLM (Runner.run is monkey-patched) but uses
real prompt construction, real parse_directive, real dispatch_directive
plumbing (with subagent's _run_pass_loop also driven by a mock LLM).
"""
import asyncio
import torch
import pytest


agents_sdk = pytest.importorskip("agents")


def _run(coro):
    # Use asyncio.run but restore a fresh event loop so tests that call
    # the deprecated asyncio.get_event_loop() still work when run after this.
    result = asyncio.run(coro)
    asyncio.set_event_loop(asyncio.new_event_loop())
    return result


class _FakeRunResult:
    def __init__(self, output):
        self.final_output = output
        self.context_wrapper = type("C", (), {"usage": None})()
        self.new_items = []  # required by _reasoning_text


class _FakeAgent:
    """Minimal agent stub — only .instructions is required by _run_pass_loop."""
    instructions = "# fake system prompt"


def test_directive_then_tiled_reference_flow(tmp_path, monkeypatch):
    """Parent emits SUB_TASKS turn 0; subagent emits tiled_reference; parent
    emits its tiled_reference turn 1; the whole pass succeeds."""
    from src import orchestrator as orch_mod
    from src import subdivide as sub_mod

    # Stub build_pass_user_prompt so the test doesn't require a real StepDB
    # kernel entry (test_kernel doesn't exist in bench_config.yaml).
    monkeypatch.setattr(orch_mod, "build_pass_user_prompt",
                        lambda *a, **kw: "## fake user prompt")

    parent_directive = """```python
import torch

def preamble(dims, tensors):
    return {"x": tensors["x"]}

def sub_reference(dims, sub_tensors):
    return sub_tensors["x"] * 2

SUB_TASKS = [
    {"name": "doubler", "preamble": preamble, "sub_reference": sub_reference},
]
```"""

    sub_tiled_ref = """```python
def tiled_reference(dims, tensors):
    return tensors["x"] * 2
```"""

    parent_tiled_ref = """```python
def tiled_reference(dims, tensors):
    return tensors["x"] * 2
```"""

    responses = [parent_directive, sub_tiled_ref, parent_tiled_ref]

    async def fake_runner_run(agent, conversation):
        return _FakeRunResult(responses.pop(0))

    monkeypatch.setattr("src.orchestrator.Runner.run", fake_runner_run)

    # Stub gate functions to accept any candidate as correct
    monkeypatch.setattr(orch_mod, "_run_dsl_correctness",
                        lambda code, kernel, dims, tensors: "match=True\nmax_diff=0.0")
    monkeypatch.setattr(orch_mod, "_check_banned_ops", lambda code, pass_name: [])

    # Stub the dsl scaffold so user code can be exec'd without step_dsl
    from src import tools as tools_mod
    monkeypatch.setattr(tools_mod, "_build_dsl_scaffold", lambda: "import torch\n")

    # Stub the pass-agent factory so dispatch_directive doesn't try to create
    # a real OpenAI client (llm_config={} has no url/api_key).
    from src import subdivide as sub_mod2
    monkeypatch.setattr(sub_mod2, "_make_subdivide_pass_agent",
                        lambda llm_config, options: _FakeAgent())

    options = sub_mod.SubdivideOptions(
        max_subdivide_turns=4,
        max_subdivides_per_outer=5,
        max_subdivide_depth=2,
    )
    registry = []
    counter = sub_mod.SubdivideCounter()
    parent_tensors = {"x": torch.tensor([1.0, 2.0, 3.0])}

    orch_mod._inject_gold("test_kernel", {}, torch.tensor([2.0, 4.0, 6.0]))

    fake_agent = _FakeAgent()
    pass_result = _run(orch_mod._run_pass_loop(
        fake_agent, "refactor_final", "test_kernel", {},
        max_turns=5, ckpt_dir=tmp_path,
        executor="dsl", tensors=parent_tensors,
        log=lambda m: None,
        judge_agent=None, dsl_code=None,
        post_validator=None,
        compliance_override=None,
        check_order="correctness-first",
        subdivide_options=options,
        registry=registry,
        depth=0,
        counter=counter,
        llm_config={},
    ))

    assert pass_result["success"] is True
    assert len(registry) == 1
    assert registry[0].name == "doubler"
    assert counter.used == 1
