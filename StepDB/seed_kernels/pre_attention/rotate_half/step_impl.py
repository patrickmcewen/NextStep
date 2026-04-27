"""STeP implementation: rotate_half (HuggingFace-style half-swap).

Mirrors the approach used in step_tl/end_to_end/attention/qkv_gen.py::rotate_half.

Pipeline (stream shape / tile shape at each step):

    Load input [batch, num_heads, head_dim] flattened to 2-D:
        stream (1, batch, 1), tile (num_heads, head_dim)

    RetileStreamify(split_row=False, chunk=head_dim/2)
        stream (1, batch, 2), tile (num_heads, head_dim/2)
        ordering at stream pos (0, b, c): c=0 is first half, c=1 is second half

    Flatten(0, 2)
        stream (batch*2,), tile (num_heads, head_dim/2)
        ordering: [first_b0, second_b0, first_b1, second_b1, ...]

    Parallelize(num_consumers=2, switch_cycles=[1,1])
        cycle-level interleave -> consumer 0: all first halves, consumer 1: all second halves

    UnaryMap(MulImmediate(-1.0)) on consumer 1 -> negated second halves

    StaticReassemble([neg_second, first], switch_cycles=[1,1])
        interleaved back: [-second_b0, first_b0, -second_b1, first_b1, ...]

    ReshapePadStream(chunk_size=2, reshape_rank=0)
        stream (batch, 2), tile (num_heads, head_dim/2)

    Accum(fn=RetileCol, accum_rank=1, init=Empty((num_heads, 0)))
        concatenates the two half-tiles into one (num_heads, head_dim) tile
        stream (batch,), tile (num_heads, head_dim)

    Two ReshapePadStreams to restore a 3-D stream shape (1, batch, 1) so
    OffChipStore's Rust backend sees a 2-D tensor_shape_tiled and stacks
    tiles vertically along batch rather than horizontally.

    OffChipStore
"""

SEED = 42


def build_graph(dims):
    batch = dims["batch"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    assert head_dim % 2 == 0, f"head_dim={head_dim} must be even"

    torch.manual_seed(SEED)
    x = torch.randn(batch, num_heads, head_dim)
    x_flat = x.reshape(batch * num_heads, head_dim).contiguous()

    step_graph = Graph()

    loaded = LinearOffChipLoad(
        underlying=x_flat,
        stride=(1, 1),
        out_shape_tiled=(batch, 1),
        tile_row=num_heads,
        tile_col=head_dim,
        par_dispatch=4,
    )

    split_half = RetileStreamify(
        graph=step_graph,
        input=loaded,
        split_row=False,
        chunk=head_dim // 2,
    )

    flat = Flatten(
        graph=step_graph,
        input=split_half,
        min_rank=0,
        max_rank=2,
    )

    splitted = Parallelize(
        graph=step_graph,
        input=flat,
        parallelize_rank=0,
        num_consumers=2,
        switch_cycles=[1, 1],
    )

    neg_second = UnaryMap(
        graph=step_graph,
        input=(splitted, 1),
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=1024,
    )

    concat = StaticReassemble(
        graph=step_graph,
        inputs=[neg_second, (splitted, 0)],
        merge_rank=0,
        switch_cycles=[1, 1],
    )

    reshaped = ReshapePadStream(
        graph=step_graph,
        input=concat,
        chunk_size=2,
        reshape_rank=0,
        write_back_mu=False,
        pad_fn=None,
        have_pad_stream=False,
    )

    full_tile = Accum(
        graph=step_graph,
        input=reshaped,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(num_heads, head_dim)),
        fn=RetileCol(),
        init_fn=Empty(shape=(num_heads, 0), dtype=Float32()),
        accum_rank=1,
        write_back_mu=True,
        compute_bw=1024,
    )

    # Restore stream shape to (1, batch, 1) so tensor_shape_tiled = (batch, 1)
    # and Rust OffChipStore lays tiles out as (batch*num_heads, head_dim).
    restored_inner = ReshapePadStream(
        graph=step_graph,
        input=full_tile,
        chunk_size=1,
        reshape_rank=0,
        write_back_mu=False,
        pad_fn=None,
        have_pad_stream=False,
    )
    restored_outer = ReshapePadStream(
        graph=step_graph,
        input=restored_inner,
        chunk_size=batch,
        reshape_rank=1,
        write_back_mu=False,
        pad_fn=None,
        have_pad_stream=False,
    )

    output = OffChipStore(
        graph=step_graph,
        input=restored_outer,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
