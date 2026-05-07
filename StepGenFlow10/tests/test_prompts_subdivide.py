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
