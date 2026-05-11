# MoE compute (SwiGLU style):
#   1. Load per‑expert weight matrices (gate, up, down) from off‑chip.
#   2. Load per‑token routing scalars (expert_weights) from off‑chip.
#   3. Build a control mask from the one‑hot routing map (expert_onehot).
#   4. Replicate each token embedding for the two activated‑expert slots,
#      flatten token×slot → a single stream, and flatten the routing‑weight
#      stream the same way.
#   5. Partition the token stream and routing‑weight stream per expert using
#      the control mask.
#   6. Split the loaded weight matrices per expert.
#   7. For each expert:
#        * broadcast its gate/up/down matrices to the token stream with
#          `expand_ref`;
#        * compute the SwiGLU activation:
#            a = x @ w_gate
#            a = silu(a)
#            b = x @ w_up
#            h = a * b
#        * down‑project: out = h @ w_down;
#        * multiply by the per‑token routing weight.
#   8. Re‑assemble the per‑expert token streams with `flat_reassemble`,
#      flatten away the extra leading singleton dimension, reshape back to
#      (seq, slot, 1, dim), and finally sum over the two slots with
#      `accum_add`. The result is a stream tensor of shape (64, 1, 512) as
#      required by the parent.

def moe_compute(
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
        stride=(2, 1),                 # rows advance by 2 (the two slots)
        out_shape_tiled=(64, 2),
        tile_row=1,
        tile_col=1,
        par_dispatch=1,
    )
    # shape after offchip_load: (1, 64, 2, 1, 1) → flatten to (64, 2, 1, 1)
    expert_weights_stream = flatten(ew_raw, min_rank=1, max_rank=2)   # (64, 2, 1, 1)

    # ------------------------------------------------------------
    # 3. Build control mask from expert_onehot (int64)
    # ------------------------------------------------------------
    control_raw = select_gen(expert_onehot, is_multihot=False, n=8)   # (1,64)×tile(2,8)

    # split the tile‑row (slot) dimension into a stream dimension
    control_retile = retile_streamify(
        control_raw, chunk=1, split_row=True
    )  # (1,128)×tile(1,8)

    # merge leading singleton → (128, 1, 8)
    control = flatten(control_retile, min_rank=0, max_rank=1)  # (128, 1, 8)

    # ------------------------------------------------------------
    # 4. Replicate token embeddings for the two slots and flatten
    # ------------------------------------------------------------
    # add a singleton stream dim so we can expand to size‑2 using the weight stream as reference
    normed_extra = reshape_stream(
        normed_2, chunk_size=1, rank=0, add_outer_dim=False
    )  # (64, 1)×tile(1,512)

    # expand the singleton dim to size 2 using the routing‑weight stream as reference
    normed_exp = expand_ref(
        normed_extra, expert_weights_stream, expand_rank=1
    )  # (64, 2)×tile(1,512)

    # collapse token × slot into a single stream dimension
    normed_flat = flatten(normed_exp, min_rank=0, max_rank=1)  # (128, 1, 512)

    # ------------------------------------------------------------
    # 5. Partition tokens and routing weights per expert according to `control`
    # ------------------------------------------------------------
    tokens_per_expert = flat_partition(
        normed_flat, control, n=8
    )  # list[8] of (M_i, 1, 512)

    # flatten routing‑weight stream to match the flattened token stream
    weights_flat = flatten(expert_weights_stream, min_rank=0, max_rank=1)  # (128, 1, 1)
    weights_per_expert = flat_partition(
        weights_flat, control, n=8
    )  # list[8] of (M_i, 1, 1)

    # ------------------------------------------------------------
    # 6. Split weight matrices per expert
    # ------------------------------------------------------------
    gate_per_expert = parallelize(w_gate_stream, 8)   # each (1, 512, 1792)
    up_per_expert = parallelize(w_up_stream, 8)       # each (1, 512, 1792)
    down_per_expert = parallelize(w_down_stream, 8)   # each (1, 1792, 512)

    # ------------------------------------------------------------
    # 7. Compute expert outputs and apply routing weight
    # ------------------------------------------------------------
    expert_outputs = []
    for i in range(8):
        x = tokens_per_expert[i]        # (M_i, 1, 512)
        w = weights_per_expert[i]       # (M_i, 1, 1)

        # broadcast gate and up matrices to the token stream shape
        gate_exp = expand_ref(gate_per_expert[i], x, expand_rank=1)   # (M_i, 512, 1792)
        up_exp = expand_ref(up_per_expert[i], x, expand_rank=1)       # (M_i, 512, 1792)

        # SwiGLU: a = silu(x @ gate), b = x @ up, h = a * b
        a = binary_matmul(x, gate_exp)       # (M_i, 1, 1792)
        a = unary_silu(a)                    # (M_i, 1, 1792)   ← silu(gate)
        b = binary_matmul(x, up_exp)        # (M_i, 1, 1792)
        h = binary_mul(a, b)                 # (M_i, 1, 1792)

        # down projection (broadcast down matrix)
        down_exp = expand_ref(down_per_expert[i], h, expand_rank=1)   # (M_i, 1792, 512)
        out = binary_matmul(h, down_exp)               # (M_i, 1, 512)

        # apply the scalar routing weight (broadcast over tile cols)
        out_w = binary_mul(out, w)          # (M_i, 1, 512)

        expert_outputs.append(out_w)

    # ------------------------------------------------------------
    # 8. Re‑assemble token stream from per‑expert pieces
    # ------------------------------------------------------------
    merged = flat_reassemble(expert_outputs, control)   # (1, 128, 1, 1)×tile(1, 512)

    # collapse all stream dims into a single one (remove the leading singleton)
    merged_flat = flatten(merged, min_rank=0, max_rank=3)   # (128, 1, 512)

    # restore token × slot layout (64 tokens, 2 slots)
    token_pos = reshape_stream(
        merged_flat, chunk_size=2, rank=0, add_outer_dim=False
    )                                   # (64, 2, 1, 512)

    # sum the two routed slots per token
    final = accum_add(token_pos, rank=1)               # (64, 1, 512)

    return final