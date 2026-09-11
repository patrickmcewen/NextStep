import sys
from pathlib import Path
STEP_TL_SRC = str(Path(__file__).resolve().parent.parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(STEP_TL_SRC).parent / "src" / "proto")
sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)

from src.prompts import build_writer_system_prompt, build_writer_user_prompt, build_analyst_prompt

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
