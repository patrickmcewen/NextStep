"""Tests for step_dsl_memory.py — the metric-recording wrapper around step_dsl."""

import torch

from src import step_dsl_memory as sdm


def test_tracker_context_manager_lifecycle():
    # No tracker active outside the with-block.
    assert sdm._ACTIVE is None
    with sdm.tracker() as t:
        assert sdm._ACTIVE is t
        assert isinstance(t, sdm.Tracker)
    assert sdm._ACTIVE is None


def test_tracker_default_mock_bf16_is_true():
    with sdm.tracker() as t:
        assert t.mock_bf16 is True


def test_tracker_mock_bf16_can_be_overridden():
    with sdm.tracker(mock_bf16=False) as t:
        assert t.mock_bf16 is False


def test_nested_trackers_save_and_restore():
    with sdm.tracker() as outer:
        with sdm.tracker(mock_bf16=False) as inner:
            assert sdm._ACTIVE is inner
        assert sdm._ACTIVE is outer
    assert sdm._ACTIVE is None


def test_unwrapped_call_outside_tracker_does_not_record():
    # Calling a wrapped DSL fn outside a tracker is fine and records nothing.
    a = torch.randn(2, 4, 4, dtype=torch.float32)
    b = torch.randn(2, 4, 4, dtype=torch.float32)
    out = sdm.binary_add(a, b)
    assert torch.equal(out, a + b)


def test_wrapped_function_signatures_preserved():
    # Wrapping must keep the function name and basic call signature usable.
    assert callable(sdm.offchip_load)
    assert callable(sdm.binary_matmul)
    assert callable(sdm.unary_silu)
    assert callable(sdm.offchip_store)


def test_dsl_functions_reexported():
    # Every name in step_dsl.DSL_FUNCTIONS is exported from step_dsl_memory.
    from src import step_dsl
    for name in step_dsl.DSL_FUNCTIONS:
        assert hasattr(sdm, name), f"step_dsl_memory missing {name}"


def test_records_empty_when_no_metric_fn_yet():
    # Until ops are registered in METRIC_FNS, records list stays empty even
    # if a wrapped DSL function is called inside a tracker.
    # select_gen is from the source-control family (not yet registered).
    ctrl = torch.zeros(2, 2, dtype=torch.int32)
    ctrl[:, 0] = 1
    with sdm.tracker() as t:
        sdm.select_gen(ctrl, is_multihot=True, n=2)
    assert t.records == []


def test_n_byte_float16():
    assert sdm._n_byte(torch.float16, mock_bf16=False) == 2
    assert sdm._n_byte(torch.float16, mock_bf16=True) == 2


def test_n_byte_float32_real():
    assert sdm._n_byte(torch.float32, mock_bf16=False) == 4


def test_n_byte_float32_mock_bf16():
    assert sdm._n_byte(torch.float32, mock_bf16=True) == 2


def test_n_byte_uint64():
    # Uint64 is keyed by IR datatype class name in ops.py; we never see it
    # via a torch dtype in the eager DSL, but the helper handles the name.
    assert sdm._n_byte_for_name("Uint64", mock_bf16=False) == 8
    assert sdm._n_byte_for_name("Uint64", mock_bf16=True) == 8


def test_tile_bytes():
    a = torch.randn(2, 3, 5, 7, dtype=torch.float32)
    # tile = (5, 7); n_byte = 4 (real) or 2 (mock_bf16)
    assert sdm._tile_bytes(a, mock_bf16=False) == 5 * 7 * 4
    assert sdm._tile_bytes(a, mock_bf16=True) == 5 * 7 * 2


def test_stream_total_elements_2d_is_one():
    # Pure tile, no stream dims — stream total elements is 1.
    a = torch.randn(4, 4, dtype=torch.float32)
    assert sdm._stream_total_elements(a) == 1


def test_stream_total_elements_higher_rank():
    a = torch.randn(2, 3, 5, 7, dtype=torch.float32)
    # stream shape = (2, 3); total = 6
    assert sdm._stream_total_elements(a) == 6


def test_stream_dtype_size_bytes_uses_output_tile():
    # Mirrors ops.py's stream.stream_dtype.size_in_bytes() — tile_r * tile_c * n_byte.
    a = torch.randn(1, 8, 16, dtype=torch.float16)
    # tile = (8, 16); n_byte = 2
    assert sdm._stream_dtype_size_bytes(a, mock_bf16=True) == 8 * 16 * 2


# ---------------------------------------------------------------------------
# IR-parity validation harness (Task 3)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

import pytest

# Add step_tl to sys.path so step_py.* imports work in the test process.
# (src/tools.py also does this at import time, but be explicit here so the
# harness is self-contained when run standalone.)
_DEIO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_DEIO_ROOT / "step_tl" / "src"))

from src import step_dsl  # noqa: E402  (after sys.path edit)
from src.dsl_to_step import translate  # noqa: E402
from src.tools import _exec_build_graph  # noqa: E402


def _ir_totals(dsl_src: str, dims: dict, tensors: dict) -> tuple[int, int, int]:
    """Translate dsl_src to IR, build the graph, sum per-node memory metrics."""
    translated = translate(dsl_src)
    graph, _output_op = _exec_build_graph(translated, dims, tensors)
    off = 0
    on = 0
    on_fifo = 0
    for node in graph.nodes():
        off += int(node.off_chip_traffic())
        on += int(node.on_chip_requirement(count_fifos=False))
        on_fifo += int(node.on_chip_requirement(count_fifos=True))
    return off, on, on_fifo


def _shim_totals(dsl_src: str, dims: dict, tensors: dict) -> tuple[int, int, int]:
    """Run dsl_src against step_dsl_memory under a fresh tracker; return totals.

    Uses mock_bf16=False so the shim agrees with the IR side, which always
    constructs ops without mock_bf16 (translator does not forward the kwarg).
    """
    namespace: dict = {}
    exec("import torch\nimport torch.nn.functional as F\nimport math\n", namespace)
    namespace.update({n: getattr(sdm, n) for n in step_dsl.DSL_FUNCTIONS})
    namespace["Buffered"] = sdm.Buffered
    exec(dsl_src, namespace)
    fn = namespace["tiled_reference"]
    with sdm.tracker(mock_bf16=False) as t:
        fn(dims, tensors)
    return t.total_off_chip, t.total_on_chip, t.total_on_chip_fifo


def _assert_parity(dsl_src: str, dims: dict, tensors: dict):
    ir = _ir_totals(dsl_src, dims, tensors)
    shim = _shim_totals(dsl_src, dims, tensors)
    assert shim == ir, (
        f"shim totals != IR totals\n"
        f"  shim (off, on, on_fifo) = {shim}\n"
        f"  ir   (off, on, on_fifo) = {ir}"
    )


def test_harness_imports_cleanly():
    """Smoke test: harness loads, helper functions are wired, sys.path edit works."""
    assert callable(_ir_totals)
    assert callable(_shim_totals)
    assert callable(_assert_parity)
    # step_py.ops should be importable thanks to the sys.path edit.
    import step_py.ops  # noqa: F401


def test_parity_offchip_load_then_store():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    return offchip_store(a)
'''
    A = torch.randn(8, 4, dtype=torch.float32)  # 2 tiles of 4x4
    _assert_parity(src, dims={}, tensors={"A": A})


def test_parity_dyn_offchip_load_then_store():
    src = '''
def tiled_reference(dims, tensors):
    a = dyn_offchip_load(tensors["A"], tensor_shape_tiled=(2,),
                         tile_row=4, tile_col=4)
    return offchip_store(a)
'''
    A = torch.randn(8, 4, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A})


def test_parity_offchip_load_ref():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=1, tile_col=4)
    b = offchip_load_ref(a, tensors["B"], stride=(0,), out_shape_tiled=(1,),
                         tile_row=4, tile_col=4)
    return offchip_store(b)
'''
    A = torch.randn(2, 4, dtype=torch.float32)
    B = torch.randn(4, 4, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A, "B": B})


def test_parity_random_offchip_load():
    src = '''
def tiled_reference(dims, tensors):
    addr = offchip_load(tensors["addr"], stride=(1,), out_shape_tiled=(2,),
                        tile_row=1, tile_col=1)
    a = random_offchip_load(tensors["A"], addr, tile_row=4, tile_col=4)
    return offchip_store(a)
'''
    addr = torch.zeros(2, 1, dtype=torch.float32)
    A = torch.randn(8, 8, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"addr": addr, "A": A})


def test_parity_random_offchip_store():
    """Parity test for random_offchip_store.

    The translator cannot be used here because OffChipStore (IR) rejects the
    Bool ack stream that RandomOffChipStore emits.  We build the IR graph
    directly and compare its per-node totals against the shim.
    """
    import step_py.ops as ir_ops
    from graph.graph import MultiDiGraph as Graph
    from rewrite.broadcast import infer_broadcast

    addr = torch.zeros(2, 1, dtype=torch.float32)
    data = torch.randn(8, 4, dtype=torch.float32)
    A = torch.zeros(8, 8, dtype=torch.float32)

    # --- IR side: build graph directly, no translator ---
    ir_ops.StepOps._counter = 0
    g = Graph()
    addr_load = ir_ops.LinearOffChipLoad(
        underlying=addr, stride=(1,), out_shape_tiled=(2,),
        tile_row=1, tile_col=1, par_dispatch=1,
    )
    g.add_node(addr_load)
    data_load = ir_ops.LinearOffChipLoad(
        underlying=data, stride=(1,), out_shape_tiled=(2,),
        tile_row=4, tile_col=4, par_dispatch=1,
    )
    g.add_node(data_load)
    ros = ir_ops.RandomOffChipStore(
        graph=g, underlying=A,
        wdata=data_load, waddr=addr_load,
        tile_row=4, tile_col=4, base_addr_byte=0, par_dispatch=1,
    )
    g = infer_broadcast(g)
    ir_off = sum(int(n.off_chip_traffic()) for n in g.nodes())
    ir_on = sum(int(n.on_chip_requirement(count_fifos=False)) for n in g.nodes())
    ir_on_fifo = sum(int(n.on_chip_requirement(count_fifos=True)) for n in g.nodes())

    # --- shim side ---
    src = '''
def tiled_reference(dims, tensors):
    addr = offchip_load(tensors["addr"], stride=(1,), out_shape_tiled=(2,),
                        tile_row=1, tile_col=1)
    data = offchip_load(tensors["data"], stride=(1,), out_shape_tiled=(2,),
                        tile_row=4, tile_col=4)
    return random_offchip_store(tensors["A"], data, addr,
                                tile_row=4, tile_col=4)
'''
    shim = _shim_totals(src, dims={}, tensors={"addr": addr, "data": data, "A": A})
    ir = (ir_off, ir_on, ir_on_fifo)
    assert shim == ir, (
        f"shim totals != IR totals\n"
        f"  shim (off, on, on_fifo) = {shim}\n"
        f"  ir   (off, on, on_fifo) = {ir}"
    )


def test_parity_binary_matmul():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    b = offchip_load(tensors["B"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    c = binary_matmul(a, b)
    return offchip_store(c)
'''
    A = torch.randn(8, 4, dtype=torch.float32)
    B = torch.randn(8, 4, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A, "B": B})


def test_parity_binary_mul_add_div():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    b = offchip_load(tensors["B"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    c = binary_mul(a, b)
    d = binary_add(c, b)
    e = binary_div(d, a)
    return offchip_store(e)
'''
    A = torch.randn(8, 4, dtype=torch.float32)
    B = torch.randn(8, 4, dtype=torch.float32) + 1.0
    _assert_parity(src, dims={}, tensors={"A": A, "B": B})


def test_parity_unary_chain():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    b = unary_silu(a)
    c = unary_square(b)
    d = unary_exp(c)
    e = unary_mul_imm(d, 2.5)
    f = unary_add_imm(e, 1.0)
    return offchip_store(f)
'''
    A = torch.randn(8, 4, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A})


def test_parity_accum_add():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2, 4),
                     tile_row=4, tile_col=4)
    b = accum_add(a, rank=1)
    return offchip_store(b)
'''
    A = torch.randn(8, 16, dtype=torch.float32)  # 2x4 tiles of 4x4
    _assert_parity(src, dims={}, tensors={"A": A})


def test_parity_binary_map_accum():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2, 4),
                     tile_row=4, tile_col=4)
    b = offchip_load(tensors["B"], stride=(1,), out_shape_tiled=(2, 4),
                     tile_row=4, tile_col=4)
    c = binary_map_accum(a, b, rank=1)
    return offchip_store(c)
'''
    A = torch.randn(8, 16, dtype=torch.float32)
    B = torch.randn(8, 16, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A, "B": B})


def test_parity_promote_outer():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    b = promote_outer(a)
    return offchip_store(b)
'''
    A = torch.randn(8, 4, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A})


def test_parity_flatten():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1, 1), out_shape_tiled=(2, 2),
                     tile_row=4, tile_col=4)
    b = flatten(a, min_rank=0, max_rank=1)
    return offchip_store(b)
'''
    A = torch.randn(8, 8, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A})


def test_parity_repeat_static():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    b = repeat_static(a, factor=3)
    return offchip_store(b)
'''
    A = torch.randn(8, 4, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A})


def test_parity_bufferize_streamify():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                     tile_row=4, tile_col=4)
    buf = bufferize(a, rank=1)
    out = streamify(buf, stride=(1,), out_shape_tiled=(2,))
    return offchip_store(out)
'''
    A = torch.randn(8, 4, dtype=torch.float32)
    _assert_parity(src, dims={}, tensors={"A": A})
