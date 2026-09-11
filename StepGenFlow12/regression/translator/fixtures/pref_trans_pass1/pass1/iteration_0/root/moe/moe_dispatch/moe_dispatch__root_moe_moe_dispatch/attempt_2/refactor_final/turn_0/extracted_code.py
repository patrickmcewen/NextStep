# MoE dispatch implementation using DSL ops.
#   1. Convert the one‑hot routing tensor to a MultiHot selector (control) and
#      to an Index selector (expert_idx).  The Index is used by `expert_addr_gen`
#      to compute per‑token address streams for the expert weight matrices.
#   2. `random_offchip_load` loads the appropriate expert weight tile for each
#      token‑top‑position using the address streams.
#   3. `offchip_load_ref` broadcasts the normalized activations to the same
#      stream shape as the weight tiles.
#   4. Perform the gate, up, and down linear projections with `binary_matmul`
#      (including the SiLU non‑linearity) and finally weight the result by the
#      per‑token expert weight (`expert_weights`).
#   5. Reduce over the per‑token top‑position dimension (and the extra
#      singleton streaming dimensions introduced by address generation) with
#      `accum_add`, then flatten the leading singleton to obtain the final
#      (seq_len, 1, hidden_dim) stream.
def moe_dispatch__root_moe_moe_dispatch(
    normed_2,
    w_gate,
    w_up,
    w_down,
    expert_weights,
    expert_onehot,
    *,
    out_shapes,
    out_perms=None,
):
    # 1️⃣  Control selectors ----------------------------------------------------
    # MultiHot selector: one‑hot per token per top‑position across experts.
    control = select_gen(expert_onehot, is_multihot=True, n=8)
    # Index selector (same data, but treated as a single‑hot Index).
    expert_idx = select_gen(expert_onehot, is_multihot=False, n=8)

    # 2️⃣  Address streams for each weight matrix --------------------------------
    gate_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    up_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    down_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)

    # 3️⃣  Load the per‑expert weight tiles (one tile per expert) ----------------
    gate_tile = random_offchip_load(w_gate, gate_addr, tile_row=512, tile_col=1792)
    up_tile = random_offchip_load(w_up, up_addr, tile_row=512, tile_col=1792)
    down_tile = random_offchip_load(w_down, down_addr, tile_row=1792, tile_col=512)

    # 4️⃣  Broadcast the normalized activations to the same stream shape ---------
    #   normed_2 has shape (seq_len, dim) → (seq_len, 1, dim) as a tile.
    #   We broadcast it using the address stream as reference.
    normed_ref = offchip_load_ref(
        gate_addr,
        normed_2,
        stride=[1],               # grid_c = dim//tile_col = 1
        out_shape_tiled=[1],      # broadcast a single tile
        tile_row=1,
        tile_col=512,
    )

    # 5️⃣  Compute gate and up projections --------------------------------------
    gate_out = binary_matmul(normed_ref, gate_tile)   # (1, seq, 2, 1,1,1, 1,1792)
    up_out = binary_matmul(normed_ref, up_tile)       # same stream shape

    # 6️⃣  Apply SiLU to gate and multiply by up ---------------------------------
    gate_act = unary_silu(gate_out)
    hidden = binary_mul(gate_act, up_out)

    # 7️⃣  Down projection -------------------------------------------------------
    down_out = binary_matmul(hidden, down_tile)       # (1, seq, 2, 1,1,1, 1,512)

    # 8️⃣  Load per‑token expert weights (scalar) and broadcast ------------------
    # Load the (seq_len, 2) matrix as a stream of scalar tiles.
    weights_stream = offchip_load(
        expert_weights,
        stride=[2, 1],                     # grid_c = 1
        out_shape_tiled=[expert_weights.shape[0], expert_weights.shape[1]],
        tile_row=1,
        tile_col=1,
    )  # shape (1, seq_len, 2, 1, 1)

    # Add the three singleton streaming dimensions that appear in the address
    # streams (they come from `expert_addr_gen`).  We repeatedly split the
    # innermost stream dimension (size 2) into (2,1) to insert a singleton after
    # it, then split that singleton again, etc.
    w1 = reshape_stream(weights_stream, chunk_size=1, rank=0)  # → (1, seq, 2, 1, 1, 1)
    w2 = reshape_stream(w1, chunk_size=1, rank=0)              # → (1, seq, 2, 1, 1, 1, 1)
    w3 = reshape_stream(w2, chunk_size=1, rank=0)              # → (1, seq, 2, 1, 1, 1, 1, 1)
    # After three splits we have exactly the same streaming shape as `down_out`.
    weights_exp = w3

    # 9️⃣  Weight the down‑projected values by the expert weights ----------------
    weighted_down = binary_mul(down_out, weights_exp)

    # 10️⃣ Reduce over the top‑position dimension and the three extra singletons.
    #   Stream shape before reduction: (1, seq_len, 2, 1, 1, 1)
    #   Dropping the innermost 4 dimensions leaves (1, seq_len).
    summed = accum_add(weighted_down, rank=4)

    # 11️⃣  Remove the leading singleton stream dimension.
    #   After the reduction we have shape (1, seq_len, 1, 512).
    #   Flatten the outer two dims (1, seq_len) → (seq_len).
    moe_output = flatten(summed, min_rank=0, max_rank=1)

    # The resulting tensor has shape (seq_len, 1, 512) as required.
    return moe_output