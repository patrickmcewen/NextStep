# MoE dispatch implemented with DSL primitives.
#   • Load the routed‑expert weight matrices with `random_offchip_load`.
#   • Broadcast the normalized activations to a stream shape that matches the
#     address streams (seq_len × 2 top‑positions) using `offchip_load` and two
#     `reshape_stream` calls (adds the two required singleton stream dimensions).
#   • Perform the gate, up, SiLU, and down projections with `binary_matmul`.
#   • Load the per‑token expert‑weight scalars, broadcast them to the same stream
#     shape via two `reshape_stream` calls, and multiply.
#   • Reduce over the top‑position dimension (and the extra singleton dims) with
#     `accum_add`, then `flatten` the leading singleton‑outer dimension to obtain
#     the final (seq_len, 1, hidden_dim) stream.
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
    # --------------------------------------------------------------------- #
    # 1️⃣  Control selectors (one‑hot & index) for expert routing
    # --------------------------------------------------------------------- #
    control = select_gen(expert_onehot, is_multihot=True, n=8)   # MultiHot(8)
    expert_idx = select_gen(expert_onehot, is_multihot=False, n=8)  # Index(8)

    # --------------------------------------------------------------------- #
    # 2️⃣  Address streams for the three weight tensors (gate, up, down)
    # --------------------------------------------------------------------- #
    gate_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    up_addr   = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    down_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)

    # --------------------------------------------------------------------- #
    # 3️⃣  Load the per‑expert weight tiles (one tile per expert)
    # --------------------------------------------------------------------- #
    gate_tile = random_offchip_load(w_gate, gate_addr, tile_row=512, tile_col=1792)
    up_tile   = random_offchip_load(w_up,   up_addr,   tile_row=512, tile_col=1792)
    down_tile = random_offchip_load(w_down, down_addr, tile_row=1792, tile_col=512)

    # --------------------------------------------------------------------- #
    # 4️⃣  Broadcast the normalized activations:
    #     - stream over (seq_len, top‑pos) with stride [1, 0] (repeat per top‑pos)
    #     - then add two singleton stream dimensions so the shape matches the
    #       address streams (1, seq_len, 2, 1, 1)
    # --------------------------------------------------------------------- #
    seq_len = normed_2.shape[0]            # 64 (static)
    dim     = normed_2.shape[1]            # 512
    # Load a (seq_len, 2) stream of the activation rows
    act_tile = offchip_load(
        normed_2.tensor,                     # raw underlying tensor
        stride=[1, 0],                       # repeat each row for the 2 top‑pos
        out_shape_tiled=[seq_len, 2],
        tile_row=1,
        tile_col=dim,
    )
    # Add the two required singleton dimensions after the top‑pos dim
    act_tile = reshape_stream(act_tile, chunk_size=1, rank=0)  # → …,2,1
    act_tile = reshape_stream(act_tile, chunk_size=1, rank=0)  # → …,2,1,1

    # --------------------------------------------------------------------- #
    # 5️⃣  Gate and up projections
    # --------------------------------------------------------------------- #
    gate_out = binary_matmul(act_tile, gate_tile)   # (…,1,1792)
    up_out   = binary_matmul(act_tile, up_tile)     # (…,1,1792)

    # --------------------------------------------------------------------- #
    # 6️⃣  SiLU non‑linearity and hidden computation
    # --------------------------------------------------------------------- #
    gate_act = unary_silu(gate_out)
    hidden   = binary_mul(gate_act, up_out)

    # --------------------------------------------------------------------- #
    # 7️⃣  Down projection
    # --------------------------------------------------------------------- #
    down_out = binary_matmul(hidden, down_tile)   # (…,1,512)

    # --------------------------------------------------------------------- #
    # 8️⃣  Load per‑token expert‑weight scalars and broadcast to the full stream
    # --------------------------------------------------------------------- #
    w_tile = offchip_load(
        expert_weights,
        stride=[2, 1],                               # row‑major over (seq_len, top)
        out_shape_tiled=[expert_weights.shape[0], expert_weights.shape[1]],
        tile_row=1,
        tile_col=1,
    )
    # Broadcast over the two extra singleton dimensions
    w_tile = reshape_stream(w_tile, chunk_size=1, rank=0)
    w_tile = reshape_stream(w_tile, chunk_size=1, rank=0)

    # --------------------------------------------------------------------- #
    # 9️⃣  Apply the expert weights
    # --------------------------------------------------------------------- #
    weighted_down = binary_mul(down_out, w_tile)

    # --------------------------------------------------------------------- #
    # 🔟  Reduce over the top‑position dimension and the two singleton dims
    # --------------------------------------------------------------------- #
    # Stream shape before reduction: (1, seq_len, 2, 1, 1)
    # Reducing the innermost three dimensions (1, 1, 2) leaves (1, seq_len)
    summed = accum_add(weighted_down, rank=3)

    # --------------------------------------------------------------------- #
    # 1️⃣1️⃣  Remove the leading outer singleton to obtain (seq_len, 1, dim)
    # --------------------------------------------------------------------- #
    moe_output = flatten(summed, min_rank=0, max_rank=1)

    # Output shape is (64, 1, 512) as required
    return moe_output