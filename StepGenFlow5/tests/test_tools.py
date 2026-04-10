"""Tests for src/tools.py internal helpers.

Tests the internal helpers directly (not the @function_tool wrappers
which need async/JSON handling).
"""
import sys
from pathlib import Path

# Make step_py importable
STEP_TL_SRC = str(Path(__file__).resolve().parent.parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(STEP_TL_SRC) / "proto")
sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)

import pytest

ELEMENT_WISE_ADD_CODE = '''
SEED = 42
def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m, tile_k = dims.get("tile_m", 16), dims.get("tile_k", 16)
    torch.manual_seed(SEED)
    A = torch.randn(M, K)
    B = torch.randn(M, K)
    step_graph = Graph()
    load_a = LinearOffChipLoad(underlying=A, stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k), tile_row=tile_m, tile_col=tile_k, par_dispatch=4)
    load_b = LinearOffChipLoad(underlying=B, stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k), tile_row=tile_m, tile_col=tile_k, par_dispatch=4)
    added = BinaryMap(graph=step_graph, in1=load_a, in2=load_b, fn=Add(),
        write_back_mu=True, compute_bw=1024)
    output = OffChipStore(graph=step_graph, input=added, par_dispatch=4)
    step_graph = infer_broadcast(step_graph)
    return step_graph, output
'''

DIMS = {"M": 32, "K": 48, "tile_m": 16, "tile_k": 16}


def test_exec_build_graph_and_inspect():
    from src.tools import _exec_build_graph, _format_node_values
    from step_py.functional import execute_values

    graph, output_op = _exec_build_graph(ELEMENT_WISE_ADD_CODE, DIMS)
    values = execute_values(graph)
    text = _format_node_values(values, graph)

    assert "LinearOffChipLoad" in text
    assert "BinaryMap" in text
    assert "OffChipStore" in text
    assert "shape" in text


def test_exec_build_graph_bad_code():
    from src.tools import _exec_build_graph

    bad_code = '''
def build_graph(dims):
    raise ValueError("intentional error")
'''
    with pytest.raises(ValueError, match="intentional error"):
        _exec_build_graph(bad_code, DIMS)


def test_check_correctness_good_code():
    from src.tools import _exec_build_graph, _validate_functional_mod
    from step_py.functional import execute
    from step_py.ops import StepOps

    StepOps._counter = 0
    graph, output_op = _exec_build_graph(ELEMENT_WISE_ADD_CODE, DIMS)
    sim = execute(graph, output_op)

    config = _validate_functional_mod.load_config()
    gold = _validate_functional_mod.run_reference("element_wise_add", DIMS, config)

    assert gold.shape == sim.shape
    max_err = (gold - sim).abs().max().item()
    rel_err = max_err / (gold.abs().max().item() + 1e-12)
    assert rel_err < 1e-5


def test_check_correctness_wrong_output():
    from src.tools import _exec_build_graph, _validate_functional_mod
    from step_py.functional import execute
    from step_py.ops import StepOps

    wrong_code = ELEMENT_WISE_ADD_CODE.replace("fn=Add()", "fn=Mul()")

    StepOps._counter = 0
    graph, output_op = _exec_build_graph(wrong_code, DIMS)
    sim = execute(graph, output_op)

    config = _validate_functional_mod.load_config()
    gold = _validate_functional_mod.run_reference("element_wise_add", DIMS, config)

    assert gold.shape == sim.shape
    max_err = (gold - sim).abs().max().item()
    rel_err = max_err / (gold.abs().max().item() + 1e-12)
    assert rel_err > 1e-5


def test_analyze_performance():
    from src.tools import _exec_build_graph
    from step_py.timing import analyze_timing

    graph, output_op = _exec_build_graph(ELEMENT_WISE_ADD_CODE, DIMS)
    result = analyze_timing(graph)

    assert "total_cycles" in result
    assert "per_node" in result
    assert int(result["total_cycles"]) > 0
