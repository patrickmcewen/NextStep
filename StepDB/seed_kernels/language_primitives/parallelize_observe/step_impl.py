"""STeP implementation: observable Parallelize semantic test.

Path: Load -> Parallelize(rank=1) -> per-consumer AddImmediate(i*MARKER)
      -> StaticReassemble(merge_rank=1) -> Store.

The additive per-consumer marker propagates into the output, so reading the
final tensor reveals which Parallelize consumer saw which input row. This
lets us check the Python functional emulator's per-consumer slicing against
both the PyTorch reference (which assumes round-robin per rank-1 unit, the
Rust semantics) and the actual Rust simulator output.
"""

SEED = 42


def build_graph(dims):
    B = dims["B"]
    K = dims["K"]
    tile_m = dims["tile_m"]
    tile_k = dims["tile_k"]
    par_factor = dims["par_factor"]
    M = B * tile_m
    K_total = K * tile_k

    assert B % par_factor == 0, f"B={B} not divisible by par_factor={par_factor}"

    torch.manual_seed(SEED)
    row_val = (torch.arange(B, dtype=torch.float32) + 1).repeat_interleave(tile_m).unsqueeze(1)
    A = row_val.expand(M, K_total).contiguous()

    step_graph = Graph()

    load = LinearOffChipLoad(
        underlying=A,
        stride=(K, 1),
        out_shape_tiled=(B, K),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # Strip the leading singleton from LinearOffChipLoad: (1, B, K) -> (B, K)
    flat = Flatten(step_graph, load, min_rank=1, max_rank=2)

    par = Parallelize(
        graph=step_graph,
        input=flat,
        parallelize_rank=1,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
        num_consumers=par_factor,
    )

    # MulImmediate (not AddImmediate) — Rust's `add_constant` is buggy
    # and actually multiplies (step-perf/src/functions/map_fn.rs:275),
    # so we use multiplication on both sides to keep the comparison clean.
    # Multiplier (i+1): consumer i scales its tiles by (i+1). The reference
    # mirrors this, so output values directly reveal which consumer saw each row.
    marked = []
    for i in range(par_factor):
        m = UnaryMap(
            graph=step_graph,
            input=(par, i),
            fn=MulImmediate(constant=float(i + 1)),
            write_back_mu=False,
            compute_bw=1024,
        )
        marked.append(m)

    merged = StaticReassemble(
        graph=step_graph,
        inputs=marked,
        merge_rank=1,
        switch_cycles=[1] * par_factor,
    )

    # OffChipStore requires a leading singleton dim; (B,K) -> (1,B,K).
    wrapped = ReshapePadStream(
        graph=step_graph,
        input=merged,
        chunk_size=B,
        reshape_rank=1,
        write_back_mu=False,
        have_pad_stream=False,
    )

    output = OffChipStore(
        graph=step_graph,
        input=wrapped,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
