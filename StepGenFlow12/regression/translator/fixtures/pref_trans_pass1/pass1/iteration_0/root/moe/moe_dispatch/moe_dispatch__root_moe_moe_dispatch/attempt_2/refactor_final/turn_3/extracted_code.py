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
    # -----------------------------------------------------------------
    # 1️⃣  Routing selectors (one‑hot for masking, Index for address calc)
    # -----------------------------------------------------------------
    control = select_gen(expert_onehot, is_multihot=True, n=8)   # MultiHot(8)
    expert_idx = select_gen(expert_onehot, is_multihot=False, n=8)  # Index(8)

    # -----------------------------------------------------------------
    # 2️⃣  Address streams for the three expert weight tensors
    # -----------------------------------------------------------------
    gate_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    up_addr   = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    down_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)

    # -----------------------------------------------------------------
    # 3️⃣  Load the per‑expert weight tiles (one tile per expert)
    # -----------------------------------------------------------------
    gate_tile = random_offchip_load(w_gate, gate_addr, tile_row=512, tile_col=1792)
    up_tile   = random_offchip_load(w_up,   up_addr,   tile_row=512, tile_col=1792)
    down_tile = random_offchip_load(w_down, down_addr, tile_row=1792, tile_col=512)

    # -----------------------------------------------------------------
    # 4️⃣  Broadcast the normalized activations to the address‑stream shape:
    #     (1, seq_len, 2, 1, 1) with tile (1, 512)
    # -----------------------------------------------------------------
    # normed_2 is already on‑chip: stream (64) × tile (1,512)
    act = promote_outer(normed_2)          # → (1, 64) × tile(1,512)
    act = promote(act, rank=0)             # → (1, 64, 1) × tile(1,512)
    act = promote(act, rank=0)             # → (1, 64, 1, 1) × tile(1,512)
    act = promote(act, rank=0)             # → (1, 64, 1, 1, 1) × tile(1,512)
    act = expand_ref(act, gate_addr, expand_rank=3)  # → (1,64,2,1,1) × tile(1,512)

    # -----------------------------------------------------------------
    # 5️⃣  Gate and up projections
    # -----------------------------------------------------------------
    gate_out = binary_matmul(act, gate_tile)   # tile (1,1792)
    up_out   = binary_matmul(act, up_tile)     # tile (1,1792)

    # -----------------------------------------------------------------
    # 6️⃣  SiLU non‑linearity and hidden computation
    # -----------------------------------------------------------------
    gate_act = unary_silu(gate_out)
    hidden   = binary_mul(gate_act, up_out)

    # -----------------------------------------------------------------
    # 7️⃣  Down projection
    # -----------------------------------------------------------------
    down_out = binary_matmul(hidden, down_tile)   # tile (1,512)

    # -----------------------------------------------------------------
    # 8️⃣  Load per‑token expert‑weight scalars and broadcast them
    # -----------------------------------------------------------------
    w = offchip_load(
        expert_weights,
        stride=[2, 1],
        out_shape_tiled=[normed_2.shape[0], 2],
        tile_row=1,
        tile_col=1,
    )
    w = promote(w, rank=0)   # → (1, seq_len, 2, 1) × tile(1,1)
    w = promote(w, rank=0)   # → (1, seq_len, 2, 1, 1) × tile(1,1)

    # -----------------------------------------------------------------
    # 9️⃣  Apply the expert weights
    # -----------------------------------------------------------------
    weighted_down = binary_mul(down_out, w)   # tile (1,512)

    # -----------------------------------------------------------------
    # 🔟  Reduce over top‑position dimension (and the two singletons)
    # -----------------------------------------------------------------
    summed = accum_add(weighted_down, rank=3)   # stream (1, seq_len)

    # -----------------------------------------------------------------
    # 1️⃣1️⃣  Flatten leading singleton → final shape (seq_len, 1, 512)
    # -----------------------------------------------------------------
    moe_output = flatten(summed, min_rank=0, max_rank=1)

    return moe_output