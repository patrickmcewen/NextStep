from src.dsl_to_step import translate


def test_reshape_stream_add_outer_dim_lowers_to_reshape_with_pad_fn():
    src = """
def tiled_reference(dims, tensors):
    tile_n = dims["tile_n"]
    x = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=1,
        tile_col=4,
    )
    grouped = reshape_stream(x, chunk_size=tile_n, rank=0, add_outer_dim=True)
    return grouped
"""

    out = translate(src)

    assert "grouped = Reshape(graph, x, chunk_size=tile_n, reshape_rank=0" in out
    assert "add_outer_dim=True" in out
    assert "pad_fn=Zero(shape=_dsl2step_in_tile(x).shape" in out
    assert "dtype=_dsl2step_in_tile(x).tile_dtype)" in out
    assert "ReshapePadStream" not in out


def test_retile_streamify_lowers_filter_mask_keyword():
    src = """
def tiled_reference(dims, tensors):
    x = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(2,),
        tile_row=2,
        tile_col=4,
    )
    y = retile_streamify(x, chunk=1, split_row=True, filter_mask=True)
    return y
"""

    out = translate(src)

    assert "y = RetileStreamify(graph, x, split_row=True, filter_mask=True, chunk=1)" in out
