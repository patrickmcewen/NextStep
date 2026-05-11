# MoE dispatch implemented with STeP DSL ops.
#   * expert_onehot and expert_weights are off‑chip and loaded as 1×1 tiles.
#   * The token stream is repeated for the two activation slots (repeat_static)
#     and a singleton stream dim is inserted (promote) so that it matches the
#     address stream shape produced by expert_addr_gen.
#   * expert_addr_gen turns the one‑hot routing map into an integer address
#     (offset) for each token‑slot pair.
#   * random_offchip_load fetches the per‑expert weight tiles (gate, up,
#     down) indexed by those addresses.
#   * The usual MoE computation (gate → silu → up → elementwise mul → down)
#     is performed with binary_matmul, unary_silu and binary_mul.
#   * The down‑projected per‑slot outputs are weighted by expert_weights
#     (broadcast via promote_outer) and summed over the two slots
#     (accum_add with rank=2).
#   * Finally the remaining stream dimensions are collapsed to the required
#     (seq_len, 1, hidden_dim) shape using flatten, and the result is stored
#     off‑chip.
def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down,
                                         expert_weights, expert_onehot,
                                         *, out_shapes, out_perms=None):
    # Load the one‑hot routing mask (tile 1×1)
    onehot_shape = expert_onehot.shape                     # (seq_len, 2, 8)
    stride_onehot = (
        onehot_shape[-2] * onehot_shape[-1],               # rows * cols
        onehot_shape[-1],                                 # cols
        1,
    )
    expert_onehot_stream = offchip_load(
        expert_onehot,
        stride=stride_onehot,
        out_shape_tiled=onehot_shape,
        tile_row=1,
        tile_col=1,
    )

    # Load the per‑slot scalar expert weights (tile 1×1)
    weight_shape = expert_weights.shape                     # (seq_len, 2)
    stride_weight = (weight_shape[-1], 1)                  # (cols, 1)
    expert_weights_stream = offchip_load(
        expert_weights,
        stride=stride_weight,
        out_shape_tiled=weight_shape,
        tile_row=1,
        tile_col=1,
    )

    # Repeat the normalized token stream for the two activation slots
    # (seq_len, 2, 1, hidden_dim) → (seq_len, 2, 1, hidden_dim)
    normed_rep = repeat_static(normed_2, factor=onehot_shape[1])
    # Add a singleton stream dimension so it matches the address stream shape
    normed_rep = promote(normed_rep, rank=0)   # (seq_len, 2, 1, 1, hidden_dim)

    # Convert the one‑hot routing map to integer addresses for each token‑slot
    addr = expert_addr_gen(expert_onehot_stream, expert_addr_base=0, num_tile_per_expert=1)

    # Load weight tiles for the selected expert of each token‑slot
    gate_tile = random_offchip_load(
        w_gate,
        addr,
        tile_row=w_gate.shape[1],
        tile_col=w_gate.shape[2],
    )
    up_tile = random_offchip_load(
        w_up,
        addr,
        tile_row=w_up.shape[1],
        tile_col=w_up.shape[2],
    )
    down_tile = random_offchip_load(
        w_down,
        addr,
        tile_row=w_down.shape[1],
        tile_col=w_down.shape[2],
    )

    # MoE forward pass
    gate_out = binary_matmul(normed_rep, gate_tile)        # (S,2,1,1,inter_dim)
    gate_act = unary_silu(gate_out)                        # (S,2,1,1,inter_dim)
    up_out   = binary_matmul(normed_rep, up_tile)          # (S,2,1,1,inter_dim)
    hidden   = binary_mul(gate_act, up_out)                # (S,2,1,1,inter_dim)
    down_out = binary_matmul(hidden, down_tile)            # (S,2,1,1,hidden_dim)

    # Apply the scalar expert weights (broadcast across the hidden dim)
    weighted = binary_mul(promote_outer(down_out), expert_weights_stream)

    # Sum contributions from the two slots for each token
    summed = accum_add(weighted, rank=2)   # collapses the two slot streams

    # Collapse all remaining stream dimensions into the sequence dimension
    final = flatten(summed, min_rank=0, max_rank=3)   # (seq_len, 1, hidden_dim)

    # Write the result off‑chip
    return offchip_store(final)