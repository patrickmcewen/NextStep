# MoE compute:
# 1. Load per‑expert weight matrices (gate, up, down) from off‑chip.
# 2. Load per‑token routing scalars (expert_weights) and the one‑hot routing map
#    (expert_onehot) from off‑chip.
# 3. Replicate each token embedding for the two routed‑expert positions,
#    flatten token×position → a single stream, and flatten the routing‑weight
#    tensor the same way.
# 4. Convert the one‑hot routing map into a stream mask where the tile‑row
#    dimension (the position index) is turned into a stream dimension.
#    This yields `control` of shape (128, 1, 8) – 128 = 64 tokens × 2 positions.
# 5. Partition the token stream and the routing‑weight stream per expert using
#    `flat_partition`.
# 6. Split the loaded weight matrices per expert with `parallelize`.
# 7. For each expert:
#       * promote the token stream to shape (1, M, 1, 512) with `promote_outer`.
#       * broadcast the expert’s gate/up/down matrices across the token stream
#         via `repeat_ref`.
#       * compute the gated activation:
#             a = X @ gate,  b = X @ up,
#             h = silu(a * b)
#       * down‑project: out = h @ down
#       * multiply by the per‑token routing weight (broadcasted over tile cols).
# 8. Re‑assemble the per‑expert token streams with `flat_reassemble`,
#    flatten the extra leading dimensions, reshape back to (seq, pos, 1, dim),
#    and finally sum over the two positions with `accum_add`.
#    The result is a stream tensor of shape (64, 1, 512) ready for the parent.

def moe_compute(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------
    # 1. Load per‑expert weight tensors
    # ------------------------------------------------------------
    wg_raw = offchip_load(
        w_gate,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=512,
        tile_col=1792,
        par_dispatch=1,
    )
    w_gate_stream = flatten(wg_raw, min_rank=0, max_rank=1)   # (8, 512, 1792)

    wu_raw = offchip_load(
        w_up,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=512,
        tile_col=1792,
        par_dispatch=1,
    )
    w_up_stream = flatten(wu_raw, min_rank=0, max_rank=1)     # (8, 512, 1792)

    wd_raw = offchip_load(
        w_down,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=1792,
        tile_col=512,
        par_dispatch=1,
    )
    w_down_stream = flatten(wd_raw, min_rank=0, max_rank=1)   # (8, 1792, 512)

    # ------------------------------------------------------------
    # 2. Load per‑token routing scalars (expert_weights)
    # ------------------------------------------------------------
    ew_raw = offchip_load(
        expert_weights,
        stride=(2, 1),                 # rows advance by 2 (positions) in the tile grid
        out_shape_tiled=(64, 2),
        tile_row=1,
        tile_col=1,
        par_dispatch=1,
    )
    # ew_raw: (1, 64, 2, 1, 1)
    expert_weights_stream = flatten(ew_raw, min_rank=1, max_rank=2)   # (64, 2, 1, 1)

    # ------------------------------------------------------------
    # 3. Build the routing‑mask control stream from expert_onehot
    # ------------------------------------------------------------
    # `select_gen` adds a leading singleton stream dim.
    control_raw = select_gen(expert_onehot, is_multihot=False, n=8)   # (1, 64, 2, 8)

    # Move the tile‑row (the position index) into the stream dimension.
    # After `retile_streamify` each position becomes its own stream element.
    control_retile = retile_streamify(control_raw, chunk=1, split_row=True)  # (1, 128, 1, 8)

    # Merge the leading singleton stream dim with the real stream dim → (128, 1, 8)
    control = flatten(control_retile, min_rank=0, max_rank=1)                 # (128, 1, 8)

    # ------------------------------------------------------------
    # 4. Replicate token embeddings for the two positions and flatten
    # ------------------------------------------------------------
    # `reshape_stream` splits the token stream into an extra singleton dim.
    normed_extra = reshape_stream(
        normed_2,
        chunk_size=1,
        rank=0,
        add_outer_dim=False,
    )                               # (64, 1, 1, 512)   → stream (64, 1)

    # Expand the new singleton dimension to size 2 using the routing‑weight stream as reference.
    normed_rep = expand_ref(
        normed_extra,
        expert_weights_stream,
        expand_rank=1,
    )                               # (64, 2, 1, 512)   → stream (64, 2)

    # Collapse token × position into a single stream dimension.
    normed_flat = flatten(normed_rep, min_rank=0, max_rank=1)    # (128, 1, 512)

    # ------------------------------------------------------------
    # 5. Partition tokens and routing weights per expert according to `control`
    # ------------------------------------------------------------
    tokens_per_expert = flat_partition(normed_flat, control, n=8)   # list[8] of (M_i,1,512)
    weights_flat = flatten(expert_weights_stream, min_rank=0, max_rank=1)  # (128,1,1)
    weights_per_expert = flat_partition(weights_flat, control, n=8)        # list[8] of (M_i,1,1)

    # ------------------------------------------------------------
    # 6. Split weight matrices per expert
    # ------------------------------------------------------------
    gate_per_expert = parallelize(w_gate_stream, 8)   # each (1,512,1792)
    up_per_expert   = parallelize(w_up_stream,   8)   # each (1,512,1792)
    down_per_expert = parallelize(w_down_stream, 8)   # each (1,1792,512)

    # ------------------------------------------------------------
    # 7. Compute expert outputs, apply routing weight
    # ------------------------------------------------------------
    expert_outputs = []
    for i in range(8):
        x = tokens_per_expert[i]        # (M_i,1,512)
        w = weights_per_expert[i]       # (M_i,1,1)

        # Promote token stream to have a leading singleton dimension → (1, M_i, 1, 512)
        x_prom = promote_outer(x)

        # Broadcast weight matrices across the token stream.
        gate_exp = repeat_ref(gate_per_expert[i], x_prom)   # (1, M_i, 512, 1792)
        up_exp   = repeat_ref(up_per_expert[i],   x_prom)   # (1, M_i, 512, 1792)

        # Linear projections
        a = binary_matmul(x_prom, gate_exp)   # (1, M_i, 1, 1792)
        b = binary_matmul(x_prom, up_exp)    # (1, M_i, 1, 1792)

        # Gated activation
        m = binary_mul(a, b)                 # (1, M_i, 1, 1792)
        h = unary_silu(m)                    # (1, M_i, 1, 1792)

        # Down projection (broadcast down matrix similarly)
        down_exp = repeat_ref(down_per_expert[i], h)   # (1, M_i, 1792, 512)
        out = binary_matmul(h, down_exp)               # (1, M_i, 1, 512)

        # Apply the scalar routing weight (broadcast over tile columns)
        w_prom = promote_outer(w)           # (1, M_i, 1, 1)
        out_w = binary_mul(out, w_prom)    # (1, M_i, 1, 512)

        expert_outputs.append(out_w)

    # ------------------------------------------------------------
    # 8. Re‑assemble token stream from per‑expert pieces
    # ------------------------------------------------------------
    merged = flat_reassemble(expert_outputs, control)   # (1,128,1,1,512)

    # Collapse leading dimensions → (128,1,512)
    merged_flat = flatten(merged, min_rank=0, max_rank=2)   # (128,1,512)

    # Restore token × position layout (64 tokens, 2 positions)
    token_pos = reshape_stream(
        merged_flat,
        chunk_size=2,
        rank=0,
        add_outer_dim=False,
    )                                   # (64, 2, 1, 512)

    # Sum the two routed positions per token
    final = accum_add(token_pos, rank=1)               # (64, 1, 512)

    return final