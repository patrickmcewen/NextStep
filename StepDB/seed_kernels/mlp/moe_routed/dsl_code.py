# Implementation reasoning:
# MoE with top-k routing using flat_partition/flat_reassemble.
# flat_reassemble with control stream (1, 64, 2, 8) produces output stream
# (1, 64, 2, dyn_n) where dyn_n is the reassembled active count per slot.
# We need accum_add(rank=2) to sum over both the dyn_n and n_active=2 dims,
# giving final stream (1, 64) -> output (64, 512).

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]
    tile_n = dims["tile_n"]   # 16
    tile_f = dims["tile_f"]   # 64

    tiles_f_count = F // tile_f  # 28
    assert F % tile_f == 0

    # Load x: (B, D) -> tile (1, D), stream (B,)
    x_stream = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )
    # x_stream: (1, B, 1, D)

    # Repeat x for n_active expert slots -> (B, n_active) stream
    x_rep = repeat_static(x_stream, factor=n_active)
    # x_rep: (1, B, n_active, 1, D)

    # Load expert_weights as scalar (1,1) tiles: stream (B, n_active)
    ew_stream = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )
    # ew_stream: (1, B, n_active, 1, 1)

    # Routing control: expert_onehot (B, n_active, n_experts) -> Index stream
    ctrl = select_gen(tensors["expert_onehot"], is_multihot=False, n=n_experts)
    # ctrl: (1, B, n_active, n_experts) with Index(8)

    # Route x and expert_weights through flat_partition
    x_parts = flat_partition(x_rep, ctrl, n=n_experts, partition_rank=0)
    ew_parts = flat_partition(ew_stream, ctrl, n=n_experts, partition_rank=0)

    expert_outputs = []
    for e in range(n_experts):
        x_e = x_parts[e]    # dyn stream of (1, D) tiles
        ew_e = ew_parts[e]  # dyn stream of (1, 1) scalar tiles

        # Batch the dynamic expert-token stream before the expert body:
        # dyn x (1, D) -> ceil(dyn/tile_n) x (tile_n, D).
        x_grouped = reshape_stream(
            x_e,
            chunk_size=tile_n,
            rank=0,
            add_outer_dim=True,
        )
        x_flat = flatten(x_grouped, min_rank=1, max_rank=2)
        x_packed = accum_retile_row(x_flat, rank=1)

        # Load weights broadcast to x_packed's batched dyn stream + tiles_f dim
        gate_w = offchip_load_ref(
            ref=x_packed,
            underlying=tensors["gate_weights"][e],
            stride=(1,),
            out_shape_tiled=(tiles_f_count,),
            tile_row=D,
            tile_col=tile_f,
        )
        # gate_w: (*dyn, tiles_f, D, tile_f)

        up_w = offchip_load_ref(
            ref=x_packed,
            underlying=tensors["up_weights"][e],
            stride=(1,),
            out_shape_tiled=(tiles_f_count,),
            tile_row=D,
            tile_col=tile_f,
        )

        down_w = offchip_load_ref(
            ref=x_packed,
            underlying=tensors["down_weights"][e],
            stride=(1,),
            out_shape_tiled=(tiles_f_count,),
            tile_row=tile_f,
            tile_col=D,
        )
        # down_w: (*dyn, tiles_f, tile_f, D)

        # Extend x_packed to match the F-tile stream dim
        x_e_rep = repeat_ref(x_packed, ref=gate_w)
        # x_e_rep: (*dyn_batches, tiles_f, tile_n, D)

        # x (tile_n, D) @ gate_w (D, tile_f) -> (tile_n, tile_f)
        gate_out = binary_matmul(x_e_rep, gate_w, weight_transposed=False)
        up_out = binary_matmul(x_e_rep, up_w, weight_transposed=False)

        gate_act = unary_silu(gate_out)
        proj = binary_mul(gate_act, up_out)
        # proj: (*dyn, tiles_f, 1, tile_f)

        # proj (tile_n, tile_f) @ down_w (tile_f, D), reduced over F tiles
        down_out = binary_map_accum(proj, down_w, rank=1, weight_transposed=False)
        # down_out: (*dyn_batches, tile_n, D)

        # Split tile_n rows back into per-token rows and discard zero padding.
        expert_out = retile_streamify(
            down_out,
            chunk=1,
            split_row=True,
            filter_mask=True,
        )

        # Apply expert weight: ew_e (*dyn, 1, 1) * expert_out (*dyn, 1, D)
        weighted = binary_mul(expert_out, ew_e)

        expert_outputs.append(weighted)

    # Reassemble: flat_reassemble back
    # ctrl stream is (1, B, n_active) so reassembled will be (1, B, n_active, dyn_1, 1, D)
    reassembled = flat_reassemble(expert_outputs, control=ctrl, reassemble_rank=0)
    # reassembled: (1, B, n_active, dyn_1, 1, D) -> stream (1, B, n_active, dyn_1)

    # Sum over both the dyn reassembly dim AND the n_active dim (rank=2)
    result = accum_add(reassembled, rank=2)
    # result: (1, B, 1, D)

    return offchip_store(result)
