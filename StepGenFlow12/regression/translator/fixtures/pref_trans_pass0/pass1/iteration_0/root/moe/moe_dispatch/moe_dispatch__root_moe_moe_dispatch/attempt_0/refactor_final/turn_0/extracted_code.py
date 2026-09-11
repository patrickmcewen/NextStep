# Implementation reasoning:
# 1. Duplicate each token for the two active experts (repeat_static factor=2).
# 2. Add a leading singleton stream dimension (promote_outer) so that the token
#    stream shape can match the weight streams after we load them.
# 3. Add a trailing singleton stream dimension (repeat_static factor=1) to make
#    the token stream shape (1, seq_len, 2, 1, 1) which matches the streams
#    produced by random_offchip_load for the expert weights.
# 4. Use expert_onehot (int tensor) with expert_addr_gen to compute tile
#    addresses for each weight matrix.  Load per‑token weight tiles with
#    random_offchip_load and add a leading singleton (promote_outer) to match
#    the token stream.
# 5. Perform the two linear projections (gate and up) with binary_matmul,
#    apply SiLU to the gate, and multiply element‑wise with the up projection.
# 6. Apply the down projection with binary_matmul.
# 7. Load the scalar expert_weights (float) with offchip_load; its stream shape
#    (1, seq_len, 2, 1, 1) already matches the token/weight streams.
# 8. Scale the down projection by the expert weight using binary_mul
#    (broadcasts the scalar across the 512 columns).
# 9. Accumulate over the two positions and the two singleton stream dimensions
#    with accum_add(rank=3), yielding a stream of shape (1, seq_len, 1, 512).
#10. Store the final result off‑chip; offchip_store strips the leading singleton
#    and returns the vanilla (seq_len × 512) tensor as required.

def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # 1. Duplicate tokens for the two top‑k positions
    x = repeat_static(normed_2, factor=2)          # (seq_len, 2, 1, 512)

    # 2. Add leading singleton stream dimension
    x = promote_outer(x)                           # (1, seq_len, 2, 1, 512)

    # 3. Add trailing singleton stream dimension so stream shape matches weight streams
    x = repeat_static(x, factor=1)                 # (1, seq_len, 2, 1, 1, 512)

    # 4. Generate addresses for weight tiles based on the one‑hot routing tensor
    gate_addr = expert_addr_gen(expert_onehot, expert_addr_base=0, num_tile_per_expert=1)
    up_addr   = expert_addr_gen(expert_onehot, expert_addr_base=0, num_tile_per_expert=1)
    down_addr = expert_addr_gen(expert_onehot, expert_addr_base=0, num_tile_per_expert=1)

    # 5. Load per‑token weight tiles (random_offchip_load) and promote to match token stream
    gate_w = random_offchip_load(w_gate, gate_addr, tile_row=512, tile_col=1792)
    up_w   = random_offchip_load(w_up,   up_addr,   tile_row=512, tile_col=1792)
    down_w = random_offchip_load(w_down, down_addr, tile_row=1792, tile_col=512)

    gate_w = promote_outer(gate_w)   # (1, seq_len, 2, 1, 1, 512, 1792)
    up_w   = promote_outer(up_w)     # (1, seq_len, 2, 1, 1, 512, 1792)
    down_w = promote_outer(down_w)   # (1, seq_len, 2, 1, 1, 1792, 512)

    # 6. Compute gate and up projections
    gate_out = binary_matmul(x, gate_w)      # (..., 1, 1792)
    up_out   = binary_matmul(x, up_w)        # (..., 1, 1792)

    # 7. Apply SiLU to gate and multiply with up projection
    gate_act = unary_silu(gate_out)
    hidden   = binary_mul(gate_act, up_out)  # (..., 1, 1792)

    # 8. Down projection
    down_out = binary_matmul(hidden, down_w) # (..., 1, 512)

    # 9. Load scalar expert weights (float) as a stream matching the token stream
    seq_len = normed_2.shape[0]
    exp_w = offchip_load(
        expert_weights,
        stride=[2, 1],
        out_shape_tiled=(seq_len, 2),
        tile_row=1,
        tile_col=1,
    )  # (1, seq_len, 2, 1, 1)

    # 10. Scale down projection by expert weight (broadcast across columns)
    weighted = binary_mul(down_out, exp_w)

    # 11. Accumulate over the two positions and the two singleton dimensions
    moe_out = accum_add(weighted, rank=3)   # (1, seq_len, 1, 512)

    # 12. Write result off‑chip
    return offchip_store(moe_out)