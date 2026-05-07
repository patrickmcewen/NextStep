import sys
from pathlib import Path
STEP_TL_SRC = str(Path(__file__).resolve().parent.parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(STEP_TL_SRC).parent / "src" / "proto")
sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)

import torch
from src.prompts import build_writer_system_prompt, build_writer_user_prompt, build_analyst_prompt
from src.prompts import build_subdivide_user_prompt

def test_writer_system_prompt_contains_executable_spec():
    prompt = build_writer_system_prompt()
    assert "def _apply_unary" in prompt
    assert "def _apply_binary" in prompt
    assert "def _dispatch" in prompt

def test_writer_system_prompt_contains_import_scaffold():
    prompt = build_writer_system_prompt()
    assert "LinearOffChipLoad" in prompt

def test_writer_user_prompt_contains_reference():
    prompt = build_writer_user_prompt("element_wise_add", {"M": 32, "K": 48, "tile_m": 16, "tile_k": 16})
    assert "compute_gold" in prompt
    assert "32" in prompt

def test_writer_user_prompt_with_diagnosis():
    prompt = build_writer_user_prompt(
        "element_wise_add",
        {"M": 32, "K": 48, "tile_m": 16, "tile_k": 16},
        diagnosis="The stride was wrong."
    )
    assert "The stride was wrong." in prompt

def test_analyst_prompt_contains_trace():
    prompt = build_analyst_prompt(
        kernel_name="gemm",
        dims={"M": 32, "K": 48, "N": 64},
        traces=[{"code": "def build_graph(dims): pass", "tool_outputs": ["Error: no output"]}],
    )
    assert "gemm" in prompt
    assert "def build_graph" in prompt
    assert "Error: no output" in prompt


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
    assert "Sub-kernel: doubler" in out or "Kernel: doubler" in out
    assert "sub_reference" in out
    assert "preamble" in out
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
