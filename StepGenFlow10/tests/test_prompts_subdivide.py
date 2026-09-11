import torch
from src.prompts import build_subdivide_user_prompt


def test_build_subdivide_user_prompt_contains_required_sections():
    sub_reference_source = (
        "def sub_reference(dims, sub_tensors):\n"
        "    return sub_tensors['x'] * 2\n"
    )
    preamble_source = (
        "def preamble(dims, tensors):\n"
        "    return {'x': tensors['x']}\n"
    )
    sub_tensors = {"x": torch.zeros(3, 4)}
    out = build_subdivide_user_prompt(
        name="doubler",
        sub_reference_source=sub_reference_source,
        preamble_source=preamble_source,
        dims={"seq_len": 7},
        sub_tensors=sub_tensors,
    )
    assert "## Sub-kernel: doubler" in out
    assert "sub_reference" in out
    assert "def preamble" in out
    assert "tiled_reference(dims, tensors)" in out
    assert "torch.manual_seed" in out  # the "do not call" guard line
    # The parent kernel must NOT leak in:
    assert "prefill_transformer" not in out


def test_build_subdivide_results_block_empty_returns_empty():
    from src.prompts import build_subdivide_results_block
    out = build_subdivide_results_block(registry=[])
    assert out == ""


def test_build_subdivide_results_block_single_entry():
    from src.prompts import build_subdivide_results_block
    from src.subdivide import VerifiedSubTask
    reg = [VerifiedSubTask(
        name="attention_block",
        sub_reference_source="def sub_reference(dims, sub_tensors):\n    pass\n",
        preamble_source="def preamble(dims, tensors):\n    pass\n",
        verified_sub_dsl_source="def tiled_reference(dims, tensors):\n    pass\n",
    )]
    out = build_subdivide_results_block(registry=reg)
    assert "Verified sub-task results" in out
    assert "attention_block" in out
    assert "preamble" in out
    assert "def preamble(dims, tensors):" in out  # source rendered, not just label
    assert "sub_reference" in out
    assert "def sub_reference(dims, sub_tensors):" in out
    assert "def tiled_reference(dims, tensors):" in out
    assert "verified DSL form" in out
    assert "NOT drop-in callable" in out or "not drop-in callable" in out.lower()
    assert "OffChipLoad" in out or "off-chip" in out.lower()


def test_build_subdivide_results_block_multiple_entries():
    from src.prompts import build_subdivide_results_block
    from src.subdivide import VerifiedSubTask
    reg = [
        VerifiedSubTask(name="A", sub_reference_source="a_ref", preamble_source="a_pre", verified_sub_dsl_source="a_dsl"),
        VerifiedSubTask(name="B", sub_reference_source="b_ref", preamble_source="b_pre", verified_sub_dsl_source="b_dsl"),
    ]
    out = build_subdivide_results_block(registry=reg)
    assert "Sub-task: A" in out
    assert "Sub-task: B" in out


def test_build_pass_user_prompt_includes_subdivide_results_when_present(monkeypatch, tmp_path):
    """build_pass_user_prompt grows an optional subdivide_results_block param."""
    import src.prompts as prompts
    from src.prompts import build_pass_user_prompt
    from src.subdivide import VerifiedSubTask

    monkeypatch.setattr(prompts, "_STEPDB_DIR", tmp_path)
    monkeypatch.setattr(prompts, "_load_stepdb_config", lambda: {"k": {"problem": "k.py"}})
    monkeypatch.setattr(prompts, "_get_precompute_source", lambda kernel: "# precompute")
    (tmp_path / "k.py").write_text("# fake reference\n")

    reg = [VerifiedSubTask(name="A", sub_reference_source="a", preamble_source="b", verified_sub_dsl_source="c")]
    out = build_pass_user_prompt(
        "refactor_final", "k", {"d": 1}, prev_code=None, tensors={"x": torch.zeros(2)},
        dsl_code=None, subdivide_results_block=prompts.build_subdivide_results_block(reg),
    )
    assert "Verified sub-task results" in out
    assert "Sub-task: A" in out


def test_build_subdivide_user_prompt_renders_tuple_aware_signature():
    """When sub_reference returns a tuple, the prompt's signature still says
    tiled_reference(dims, tensors) — the subagent learns the arity from the
    sub_reference body, not from the signature line."""
    sub_reference_source = (
        "def sub_reference(dims, sub_tensors):\n"
        "    x = sub_tensors['x']\n"
        "    return (x * 2, x * 3)\n"
    )
    out = build_subdivide_user_prompt(
        name="splitter",
        sub_reference_source=sub_reference_source,
        preamble_source="def preamble(d,t): return {'x': t['x']}\n",
        dims={},
        sub_tensors={"x": torch.zeros(2)},
    )
    assert "tiled_reference(dims, tensors)" in out


def test_build_pass_system_prompt_substitutes_subdivide_limits():
    from src.prompts import build_pass_system_prompt
    out = build_pass_system_prompt(
        "refactor_final",
        few_shot_examples=None,
        subdivide_options={"max_subdivide_depth": 3, "max_subdivides_per_outer": 7},
    )
    assert "{max_subdivide_depth}" not in out
    assert "{max_subdivides_per_outer}" not in out
    assert "depth 3" in out or "depth=3" in out or "at depth 3" in out or "depth of 3" in out or "capped at 3" in out
    assert "7" in out


def test_build_pass_system_prompt_no_subdivide_options_keeps_placeholders_safe():
    """When the caller doesn't pass subdivide_options, the placeholders must
    still be substituted with sane defaults so the prompt is well-formed."""
    from src.prompts import build_pass_system_prompt
    out = build_pass_system_prompt("refactor_final", few_shot_examples=None)
    assert "{max_subdivide_depth}" not in out
    assert "{max_subdivides_per_outer}" not in out
