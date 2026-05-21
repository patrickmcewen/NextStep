"""STeP implementation: GEMM with M-axis parallelism.

Two load modes for studying where parallelism helps:

  load_mode="shared" (default):
    One pair of LinearOffChipLoads feeds a Parallelize op that round-robins
    M-tiles to ``par_factor`` BinaryMapAccum consumers. Consumers share a
    single upstream stream — compute parallelism is capped by the source's
    token rate.

      A_load -> Flatten -> Parallelize
      B_load -> Flatten -> Parallelize  --\\
                                            P x BinaryMapAccum -> StaticReassemble -> OffChipStore

  load_mode="indep":
    Each consumer owns its own A/B LinearOffChipLoad. A is sliced (a view
    starting at the consumer's M-offset, stride P along the M-tile axis)
    so consumer i sees M-tiles {i, P+i, 2P+i, ...}. B is replicated by
    every consumer's load (each pulls the full B from off-chip). This
    breaks the shared-source bottleneck at the cost of redundant B traffic.

      per-consumer A_load(view), B_load -> BinaryMapAccum -> StaticReassemble -> OffChipStore

The round-robin M assignment is identical in both modes, so the output
order matches torch.matmul(A, B) and StaticReassemble(merge_rank=outer)
restores the unparallelized tile order.
"""

SEED = 42


def build_graph(dims, tensors):
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)
    tile_n = dims.get("tile_n", 16)
    par_factor = dims.get("par_factor", 2)
    par_dispatch = dims.get("par_dispatch", 4)
    compute_bw = dims.get("compute_bw", 1024)
    load_mode = dims.get("load_mode", "shared")
    assert load_mode in ("shared", "indep"), f"unknown load_mode {load_mode!r}"

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"
    m_tiles = M // tile_m
    n_tiles = N // tile_n
    k_tiles = K // tile_k
    assert m_tiles % par_factor == 0, (
        f"m_tiles={m_tiles} (M/tile_m) must be divisible by par_factor={par_factor}"
    )

    A = tensors["A"].contiguous()
    B = tensors["B"].contiguous()

    graph = Graph()

    partials = []
    if load_mode == "shared":
        a_load = LinearOffChipLoad(
            underlying=A,
            stride=(k_tiles, 0, 1),
            out_shape_tiled=(m_tiles, n_tiles, k_tiles),
            tile_row=tile_m,
            tile_col=tile_k,
            par_dispatch=par_dispatch,
        )

        b_load = LinearOffChipLoad(
            underlying=B,
            stride=(0, 1, n_tiles),
            out_shape_tiled=(m_tiles, n_tiles, k_tiles),
            tile_row=tile_k,
            tile_col=tile_n,
            par_dispatch=par_dispatch,
        )

        # LinearOffChipLoad emits stream shape (1, M_t, N_t, K_t). Drop the
        # leading singleton so Parallelize sees M_t as the outermost dim
        # (parallelize_rank must equal in_stream.rank == outermost).
        a_flat = Flatten(graph=graph, input=a_load, min_rank=2, max_rank=3)
        b_flat = Flatten(graph=graph, input=b_load, min_rank=2, max_rank=3)

        a_par = Parallelize(
            graph=graph, input=a_flat,
            parallelize_rank=a_flat.stream.rank,
            num_consumers=par_factor, switch_cycles=[1] * par_factor,
            write_back_mu=False,
        )
        b_par = Parallelize(
            graph=graph, input=b_flat,
            parallelize_rank=b_flat.stream.rank,
            num_consumers=par_factor, switch_cycles=[1] * par_factor,
            write_back_mu=False,
        )

        for i in range(par_factor):
            partial = BinaryMapAccum(
                graph=graph,
                in1=(a_par, i), in2=(b_par, i),
                fn=map_accum_fn.Matmul(),
                init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
                rank=1, write_back_mu=True, compute_bw=compute_bw,
            )
            partials.append(partial)
    else:  # load_mode == "indep"
        m_tiles_per = m_tiles // par_factor
        for i in range(par_factor):
            # Consumer i starts at M-tile i and steps by par_factor along M.
            # `start_tile_idx=i*k_tiles` shifts the load's base address by i
            # M-tile-rows into the original A — no Python-side slicing needed,
            # all consumers share the same underlying tensor object.
            a_load_i = LinearOffChipLoad(
                underlying=A,
                stride=(par_factor * k_tiles, 0, 1),
                out_shape_tiled=(m_tiles_per, n_tiles, k_tiles),
                tile_row=tile_m, tile_col=tile_k,
                par_dispatch=par_dispatch,
                start_tile_idx=i * k_tiles,
            )
            # Each consumer reads the full B independently.
            b_load_i = LinearOffChipLoad(
                underlying=B,
                stride=(0, 1, n_tiles),
                out_shape_tiled=(m_tiles_per, n_tiles, k_tiles),
                tile_row=tile_k, tile_col=tile_n,
                par_dispatch=par_dispatch,
            )
            a_flat_i = Flatten(graph=graph, input=a_load_i, min_rank=2, max_rank=3)
            b_flat_i = Flatten(graph=graph, input=b_load_i, min_rank=2, max_rank=3)

            partial = BinaryMapAccum(
                graph=graph,
                in1=a_flat_i, in2=b_flat_i,
                fn=map_accum_fn.Matmul(),
                init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
                rank=1, write_back_mu=True, compute_bw=compute_bw,
            )
            partials.append(partial)

    # Round-robin merge: with merge_rank == input.stream.rank the entire
    # outermost-dim slices interleave back, restoring the unparallelized
    # tile order (Parallelize/StaticReassemble round-trip).
    merged = StaticReassemble(
        graph=graph,
        inputs=partials,
        merge_rank=partials[0].stream.rank,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
    )

    # Re-add the leading singleton so OffChipStore's tensor_shape_tiled has
    # the expected (M_t, N_t) layout (matches the unparallelized GEMM).
    wrapped = ReshapePadStream(
        graph=graph,
        input=merged,
        chunk_size=m_tiles,
        reshape_rank=1,
        write_back_mu=False,
        have_pad_stream=False,
    )

    output_op = OffChipStore(
        graph=graph,
        input=wrapped,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
