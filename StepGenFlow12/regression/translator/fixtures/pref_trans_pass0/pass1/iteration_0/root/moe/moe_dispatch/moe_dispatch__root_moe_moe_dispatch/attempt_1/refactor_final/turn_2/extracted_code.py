# MoE dispatch (root node) expressed with STeP DSL primitives.
#   1️⃣ Convert the int64 routing mask into a tile‐stream with `select_gen`.
#   2️⃣ Load the per‑slot scalar expert weights (float) as a 1×1‑tile stream via `offchip_load`.
#   3️⃣ Broadcast the normalized token stream across the two activation slots:
#        – `promote_outer` adds a leading singleton stream dimension.
#        – `promote` inserts a trailing singleton stream dimension.
#        – `expand_ref` expands that trailing 1 into the slot dimension using the
#          shape of the expert‑weight stream as a reference.
#   4️⃣ Turn the one‑hot routing mask into integer tile addresses with `expert_addr_gen`.
#   5️⃣ Fetch the three expert weight matrices for the selected expert of each token‑slot
#      using `random_offchip_load`.  The loaded tensors contain extra singleton stream
#      dimensions (…,1,1); they are collapsed to the proper stream shape with `flatten`.
#   6️⃣ Perform the MoE forward pass:
#        gate → SiLU → up → elementwise mul → down, using `binary_matmul`,
#        `unary_silu` and `binary_mul`.
#   7️⃣ Multiply the down‑projected results by the scalar expert weights
#      (`binary_mul` broadcasts the 1×1 weight tile across the hidden dimension).
#   8️⃣ Sum the contributions from the two slots with `accum_add(rank=1)`.
#   9️⃣ Collapse the leading singleton stream dimension to obtain the required
#      output shape (seq_len, 1, hidden_dim) via `flatten`, and write the result
#      off‑chip with `offchip_store`.
def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down,
                                         expert_weights, expert_onehot,
                                         *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 1. Routing mask: int64 one‑hot → tile stream (tile: n_activated × n_routed)
    # ------------------------------------------------------------------
    onehot_stream = select_gen(
        expert_onehot,
        is_multihot=False,
        n=expert_onehot.shape[-1],
    )  # shape: stream(1, seq_len) × tile(2, n_routed_experts)

    # ------------------------------------------------------------------
    # 2. Per‑slot scalar expert weights (float) → 1×1‑tile stream
    # ------------------------------------------------------------------
    w_shape = expert_weights.shape                     # (seq_len, n_activated_experts)
    stride_w = (w_shape[-1], 1)                       # (cols, 1)
    expert_weights_stream = offchip_load(
        expert_weights,
        stride=stride_w,
        out_shape_tiled=w_shape,
        tile_row=1,
        tile_col=1,
    )  # shape: stream(1, seq_len, n_activated) × tile(1,1)

    # ------------------------------------------------------------------
    # 3. Replicate the normalized token stream across the 2 activation slots
    #    (stream shape → (1, seq_len, 2) × tile(1, hidden_dim))
    # ------------------------------------------------------------------
    #   a) add leading singleton stream dimension
    normed_outer = promote_outer(normed_2)            # stream(1, seq_len) × tile(1, hidden)
    #   b) add trailing singleton stream dimension
    normed_with_slot = promote(normed_outer, rank=0)  # stream(1, seq_len, 1) × tile(1, hidden)
    #   c) expand the trailing 1 → slot dimension (2) using the weight stream as reference
    normed_rep = expand_ref(
        normed_with_slot,
        expert_weights_stream,
        expand_rank=1,
    )  # shape: stream(1, seq_len, 2) × tile(1, hidden)

    # ------------------------------------------------------------------
    # 4. Convert one‑hot mask to per‑expert tile addresses
    # ------------------------------------------------------------------
    addr = expert_addr_gen(onehot_stream, expert_addr_base=0, num_tile_per_expert=1)

    # ------------------------------------------------------------------
    # 5. Load expert weight tiles for the selected expert of each token‑slot
    #    (collapse the extra singleton dimensions produced by random_offchip_load)
    # ------------------------------------------------------------------
    gate_raw = random_offchip_load(
        w_gate,
        addr,
        tile_row=w_gate.shape[1],
        tile_col=w_gate.shape[2],
    )
    gate_tile = flatten(gate_raw, min_rank=0, max_rank=2)   # shape: stream(1, seq_len, 2) × tile(512, 1792)

    up_raw = random_offchip_load(
        w_up,
        addr,
        tile_row=w_up.shape[1],
        tile_col=w_up.shape[2],
    )
    up_tile = flatten(up_raw, min_rank=0, max_rank=2)      # same stream, tile(512, 1792)

    down_raw = random_offchip_load(
        w_down,
        addr,
        tile_row=w_down.shape[1],
        tile_col=w_down.shape[2],
    )
    down_tile = flatten(down_raw, min_rank=0, max_rank=2)  # shape: stream(1, seq_len, 2) × tile(1792, 512)

    # ------------------------------------------------------------------
    # 6. MoE forward pass (gate → SiLU → up → elementwise mul → down)
    # ------------------------------------------------------------------
    gate_out = binary_matmul(normed_rep, gate_tile)     # tile(1, 1792)
    gate_act = unary_silu(gate_out)
    up_out = binary_matmul(normed_rep, up_tile)         # tile(1, 1792)
    hidden = binary_mul(gate_act, up_out)               # tile(1, 1792)
    down_out = binary_matmul(hidden, down_tile)         # tile(1, 512)

    # ------------------------------------------------------------------
    # 7. Apply scalar expert weights (broadcast across hidden dim)
    # ------------------------------------------------------------------
    weighted = binary_mul(down_out, expert_weights_stream)

    # ------------------------------------------------------------------
    # 8. Sum over the two activation slots
    # ------------------------------------------------------------------
    summed = accum_add(weighted, rank=1)   # stream(1, seq_len, 1) × tile(1, hidden)

    # ------------------------------------------------------------------
    # 9. Collapse leading singleton stream dim → required output shape (seq_len, 1, hidden)
    # ------------------------------------------------------------------
    final = flatten(summed, min_rank=0, max_rank=1)  # stream(seq_len) × tile(1, hidden)

    # ------------------------------------------------------------------
    # 10. Write the result off‑chip
    # ------------------------------------------------------------------
    return offchip_store(final)