"""Unit tests for src/subdivide.py."""
import pytest
import torch

from src import subdivide as sub_mod


WELL_FORMED = """
import torch

def preamble_a(dims, tensors):
    return {"x": tensors["x"]}

def sub_reference_a(dims, sub_tensors):
    return sub_tensors["x"] * 2

SUB_TASKS = [
    {"name": "doubler", "preamble": preamble_a, "sub_reference": sub_reference_a},
]
"""


WELL_FORMED_MULTI = """
import torch

def pre1(dims, tensors): return {"x": tensors["x"]}
def sub1(dims, sub_tensors): return sub_tensors["x"] * 2

def pre2(dims, tensors): return {"y": tensors["y"]}
def sub2(dims, sub_tensors): return sub_tensors["y"] + 1

SUB_TASKS = [
    {"name": "doubler", "preamble": pre1, "sub_reference": sub1},
    {"name": "adder",   "preamble": pre2, "sub_reference": sub2},
]
"""


def _opts(max_depth=2, max_count=5):
    return sub_mod.SubdivideOptions(
        max_subdivide_turns=8,
        max_subdivides_per_outer=max_count,
        max_subdivide_depth=max_depth,
    )


def test_parse_directive_single_well_formed():
    counter = sub_mod.SubdivideCounter()
    sub_tasks = sub_mod.parse_directive(
        WELL_FORMED, registry=[], depth=0, counter=counter, options=_opts()
    )
    assert len(sub_tasks) == 1
    assert sub_tasks[0]["name"] == "doubler"
    assert callable(sub_tasks[0]["preamble"])
    assert callable(sub_tasks[0]["sub_reference"])
    # Counter is NOT incremented at parse time — that happens on dispatch.
    assert counter.used == 0
    assert "def preamble_a(dims, tensors)" in sub_tasks[0]["preamble_source"]
    assert "return {\"x\": tensors[\"x\"]}" in sub_tasks[0]["preamble_source"]
    assert "def sub_reference_a(dims, sub_tensors)" in sub_tasks[0]["sub_reference_source"]
    assert "return sub_tensors[\"x\"] * 2" in sub_tasks[0]["sub_reference_source"]


def test_parse_directive_multi_well_formed():
    counter = sub_mod.SubdivideCounter()
    sub_tasks = sub_mod.parse_directive(
        WELL_FORMED_MULTI, registry=[], depth=0, counter=counter, options=_opts()
    )
    assert [t["name"] for t in sub_tasks] == ["doubler", "adder"]


def test_parse_directive_no_sub_tasks_attribute():
    code = "def tiled_reference(dims, tensors):\n    return tensors['x']\n"
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(sub_mod.NotADirective):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_empty_list():
    code = "SUB_TASKS = []\n"
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="non-empty list"):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_not_a_list():
    code = "SUB_TASKS = {}\n"
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="must be a list"):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_missing_keys():
    code = """
def pre(dims, tensors): return {}
SUB_TASKS = [{"name": "x", "preamble": pre}]
"""
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="sub_reference"):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_extra_keys():
    code = """
def pre(dims, tensors): return {}
def sub(dims, st): return st
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub, "extra": 1}]
"""
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="unexpected keys"):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_non_callable():
    code = """
SUB_TASKS = [{"name": "x", "preamble": 42, "sub_reference": 43}]
"""
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="callable"):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_empty_name():
    code = """
def pre(dims, tensors): return {}
def sub(dims, st): return st
SUB_TASKS = [{"name": "", "preamble": pre, "sub_reference": sub}]
"""
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="non-empty string"):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_dup_name_within_directive():
    code = """
def pre(dims, tensors): return {}
def sub(dims, st): return st
SUB_TASKS = [
    {"name": "x", "preamble": pre, "sub_reference": sub},
    {"name": "x", "preamble": pre, "sub_reference": sub},
]
"""
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="duplicate"):
        sub_mod.parse_directive(
            code, registry=[], depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_dup_name_against_registry():
    counter = sub_mod.SubdivideCounter()
    registry = [
        sub_mod.VerifiedSubTask(
            name="doubler",
            sub_reference_source="x",
            preamble_source="y",
            verified_sub_dsl_source="z",
        )
    ]
    with pytest.raises(AssertionError, match="already verified"):
        sub_mod.parse_directive(
            WELL_FORMED, registry=registry, depth=0, counter=counter, options=_opts()
        )


def test_parse_directive_depth_cap():
    counter = sub_mod.SubdivideCounter()
    with pytest.raises(AssertionError, match="depth"):
        sub_mod.parse_directive(
            WELL_FORMED, registry=[], depth=2, counter=counter, options=_opts(max_depth=2)
        )


def test_parse_directive_counter_cap():
    counter = sub_mod.SubdivideCounter()
    counter.used = 5
    with pytest.raises(AssertionError, match="exceeds.*per-outer"):
        sub_mod.parse_directive(
            WELL_FORMED, registry=[], depth=0, counter=counter, options=_opts(max_count=5)
        )


def test_parse_directive_disambiguates_redefined_functions():
    """If the directive code redefines a function name, _func_source must
    extract the source corresponding to the function actually bound to the
    sub-task — not the first definition with that name."""
    code = '''
def helper(dims, tensors): return {"FIRST": True}
def helper(dims, tensors): return {"SECOND": True}

def real_preamble(dims, tensors):
    return helper(dims, tensors)

def real_sub_reference(dims, sub_tensors):
    return sub_tensors["x"]

SUB_TASKS = [
    {"name": "n", "preamble": real_preamble, "sub_reference": real_sub_reference},
]
'''
    counter = sub_mod.SubdivideCounter()
    sub_tasks = sub_mod.parse_directive(
        code, registry=[], depth=0, counter=counter, options=_opts()
    )
    # The bound preamble is the LAST helper redefinition's caller; here we just
    # verify the recorded preamble_source is the real_preamble def, not a helper.
    assert "def real_preamble" in sub_tasks[0]["preamble_source"]
    assert "FIRST" not in sub_tasks[0]["preamble_source"]


def test_prepare_sub_task_single_tensor_output():
    parsed = sub_mod.parse_directive(
        WELL_FORMED, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    # preamble only forwards "x"; include a second key so preamble is a
    # subset passthrough (not a full-parent passthrough, which is rejected).
    parent_tensors = {"x": torch.tensor([1.0, 2.0, 3.0]), "_extra": torch.zeros(1)}
    prepared = sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors=parent_tensors)
    assert prepared.name == "doubler"
    assert "x" in prepared.sub_tensors
    assert torch.equal(prepared.sub_tensors["x"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(prepared.sub_gold, torch.tensor([2.0, 4.0, 6.0]))


def test_prepare_sub_task_tuple_output():
    code = """
def pre(dims, tensors): return {"x": tensors["x"]}
def sub(dims, st): return (st["x"] * 2, st["x"] * 3)
SUB_TASKS = [{"name": "splitter", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    # preamble only forwards "x"; include a second key so preamble is a
    # subset passthrough (not a full-parent passthrough, which is rejected).
    prepared = sub_mod.prepare_sub_task(
        parsed, dims={}, parent_tensors={"x": torch.tensor([1.0, 2.0]), "_extra": torch.zeros(1)}
    )
    assert isinstance(prepared.sub_gold, tuple)
    assert len(prepared.sub_gold) == 2
    assert torch.equal(prepared.sub_gold[0], torch.tensor([2.0, 4.0]))
    assert torch.equal(prepared.sub_gold[1], torch.tensor([3.0, 6.0]))


def test_prepare_sub_task_preamble_exception():
    code = """
def pre(dims, tensors): raise RuntimeError("preamble boom")
def sub(dims, st): return st
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    with pytest.raises(AssertionError, match=r"preamble raised RuntimeError"):
        sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors={})


def test_prepare_sub_task_preamble_returns_non_dict():
    code = """
def pre(dims, tensors): return [1, 2, 3]
def sub(dims, st): return st
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    with pytest.raises(AssertionError, match="dict"):
        sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors={})


def test_prepare_sub_task_sub_reference_exception():
    code = """
def pre(dims, tensors): return {}
def sub(dims, st): raise ValueError("sub boom")
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    with pytest.raises(AssertionError, match=r"sub_reference raised ValueError"):
        sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors={})


def test_prepare_sub_task_sub_reference_bad_return():
    code = """
def pre(dims, tensors): return {}
def sub(dims, st): return "not a tensor"
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    with pytest.raises(AssertionError, match="must return a torch.Tensor"):
        sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors={})


def test_prepare_sub_task_sub_reference_empty_tuple():
    code = """
def pre(dims, tensors): return {}
def sub(dims, st): return ()
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    with pytest.raises(AssertionError, match="empty tuple"):
        sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors={})


def test_prepare_sub_task_rejects_pure_passthrough_preamble():
    """A preamble that forwards every parent tensor unchanged is the
    'subdivide as escape hatch' anti-pattern and must be rejected."""
    code = """
def pre(dims, tensors): return {"x": tensors["x"], "y": tensors["y"]}
def sub(dims, st): return st["x"] + st["y"]
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    parent_tensors = {"x": torch.zeros(3), "y": torch.ones(3)}
    with pytest.raises(AssertionError, match="pure passthrough"):
        sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors=parent_tensors)


def test_prepare_sub_task_accepts_subset_passthrough():
    """Forwarding only a SUBSET of parent tensors is legitimate decomposition."""
    code = """
def pre(dims, tensors): return {"x": tensors["x"]}
def sub(dims, st): return st["x"] * 2
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    parent_tensors = {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([3.0, 4.0])}
    prepared = sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors=parent_tensors)
    assert "x" in prepared.sub_tensors
    assert "y" not in prepared.sub_tensors


def test_prepare_sub_task_accepts_transformed_passthrough():
    """Computing an intermediate from parent tensors is legitimate."""
    code = """
def pre(dims, tensors): return {"normed": tensors["x"] * 2.0}
def sub(dims, st): return st["normed"] + 1
SUB_TASKS = [{"name": "x", "preamble": pre, "sub_reference": sub}]
"""
    parsed = sub_mod.parse_directive(
        code, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )[0]
    parent_tensors = {"x": torch.tensor([1.0, 2.0])}
    prepared = sub_mod.prepare_sub_task(parsed, dims={}, parent_tensors=parent_tensors)
    assert "normed" in prepared.sub_tensors


import asyncio

_FAKE_PASS_LOOP_SUCCESS_CODE = (
    "def tiled_reference(dims, tensors):\n    return tensors['x'] * 2\n"
)


async def _fake_pass_loop_success(*args, **kwargs):
    """Fake _run_pass_loop that always succeeds with a fixed DSL output."""
    return {"success": True, "code": _FAKE_PASS_LOOP_SUCCESS_CODE, "total_tokens": 0}


async def _fake_pass_loop_fail(*args, **kwargs):
    return {"success": False, "code": "broken", "total_tokens": 0}


def _fake_make_pass_agent(*args, **kwargs):
    return object()


def _run(coro):
    return asyncio.run(coro)


def test_dispatch_directive_all_success(monkeypatch, tmp_path):
    parsed = sub_mod.parse_directive(
        WELL_FORMED, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )
    monkeypatch.setattr(sub_mod, "_run_pass_loop_for_sub_task", _fake_pass_loop_success)
    monkeypatch.setattr(sub_mod, "_make_subdivide_pass_agent", _fake_make_pass_agent)

    counter = sub_mod.SubdivideCounter()
    registry = []
    result = _run(sub_mod.dispatch_directive(
        parsed, dims={}, parent_tensors={"x": torch.tensor([1.0]), "_extra": torch.zeros(1)},
        registry=registry, depth=0, counter=counter, options=_opts(),
        ckpt_dir=tmp_path, llm_config={}, log=lambda m: None,
    ))
    assert result["success"] is True
    assert len(registry) == 1
    assert registry[0].name == "doubler"
    assert counter.used == 1


def test_dispatch_directive_one_fails_discards_all(monkeypatch, tmp_path):
    parsed = sub_mod.parse_directive(
        WELL_FORMED_MULTI, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )

    async def fake(*args, name=None, **kwargs):
        # First sub-task succeeds; second fails.
        if name == "doubler":
            return {"success": True, "code": "def tiled_reference(dims, tensors): return tensors['x']*2\n", "total_tokens": 0}
        return {"success": False, "code": "broken", "total_tokens": 0}

    monkeypatch.setattr(sub_mod, "_run_pass_loop_for_sub_task", fake)
    monkeypatch.setattr(sub_mod, "_make_subdivide_pass_agent", _fake_make_pass_agent)

    counter = sub_mod.SubdivideCounter()
    registry = []
    result = _run(sub_mod.dispatch_directive(
        parsed, dims={},
        parent_tensors={"x": torch.tensor([1.0]), "y": torch.tensor([1.0])},
        registry=registry, depth=0, counter=counter, options=_opts(),
        ckpt_dir=tmp_path, llm_config={}, log=lambda m: None,
    ))
    assert result["success"] is False
    assert "adder" in result["feedback"]
    assert registry == []
    assert counter.used == 2


def test_dispatch_directive_writes_per_sub_task_subdir(monkeypatch, tmp_path):
    parsed = sub_mod.parse_directive(
        WELL_FORMED, registry=[], depth=0,
        counter=sub_mod.SubdivideCounter(), options=_opts(),
    )
    monkeypatch.setattr(sub_mod, "_run_pass_loop_for_sub_task", _fake_pass_loop_success)
    monkeypatch.setattr(sub_mod, "_make_subdivide_pass_agent", _fake_make_pass_agent)

    counter = sub_mod.SubdivideCounter()
    registry = []
    _run(sub_mod.dispatch_directive(
        parsed, dims={}, parent_tensors={"x": torch.tensor([1.0]), "_extra": torch.zeros(1)},
        registry=registry, depth=0, counter=counter, options=_opts(),
        ckpt_dir=tmp_path, llm_config={}, log=lambda m: None,
    ))
    sub_dir = tmp_path / "sub_doubler"
    assert sub_dir.is_dir()
    assert (sub_dir / "verified_sub_dsl.py").is_file()
    assert (tmp_path / "subdivide_result.json").is_file()

    import json
    data = json.loads((tmp_path / "subdivide_result.json").read_text())
    assert data["sub_tasks"][0]["name"] == "doubler"
    assert data["sub_tasks"][0]["status"] == "ok"

    assert (sub_dir / "verified_sub_dsl.py").read_text() == _FAKE_PASS_LOOP_SUCCESS_CODE


def test_set_runners_wires_module_globals():
    """set_runners replaces the module-level seams; restore after."""
    original_runner = sub_mod._run_pass_loop_for_sub_task
    original_factory = sub_mod._make_subdivide_pass_agent

    def fake_factory(*args, **kwargs):
        return "FAKE_AGENT"

    sub_mod.set_runners(
        pass_loop_runner=_fake_pass_loop_success,
        pass_agent_factory=fake_factory,
    )
    try:
        assert sub_mod._run_pass_loop_for_sub_task is _fake_pass_loop_success
        assert sub_mod._make_subdivide_pass_agent is fake_factory
    finally:
        sub_mod._run_pass_loop_for_sub_task = original_runner
        sub_mod._make_subdivide_pass_agent = original_factory
