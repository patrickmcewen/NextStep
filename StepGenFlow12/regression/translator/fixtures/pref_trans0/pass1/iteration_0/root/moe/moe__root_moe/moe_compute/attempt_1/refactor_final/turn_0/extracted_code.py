# MoE compute:
# 1. Load per‑expert weight matrices (gate, up, down) from off‑chip.
# 2. Load per‑token routing scalars (expert_weights) and one‑hot routing map
#    (expert_onehot) from off‑chip.
# 3. Expand the token stream `normed_2` to have a second stream dimension of
#    size 2 (the two activated experts per token) using `expand_ref` with the
#    routing‑weight stream as the reference.
# 4. Flatten token‑and‑position dimensions into a single stream dimension.
#    The same flattening is applied to the routing‑weight tensor.
# 5. Use `flat_partition` with a flattened one‑hot mask to split the token
#    stream (and the corresponding routing weights) per expert.
# 6. Split the loaded weight matrices into per‑expert streams with
#    `parallelize`.
# 7. For each expert:
#       * broadcast its gate/up/down matrices to the token stream via
#         `expand_ref`;
#       * compute hidden = silu((x@gate) * (x@up));
#       * compute out = hidden @ down;
#       * multiply by the per‑token routing weight.
# 8. Re‑assemble the per‑expert token streams with `flat_reassemble`,
#    flatten the extra leading dimensions, reshape back to (token, pos),
#    and finally sum over the two positions with `accum_add`.
# The result is a stream tensor of shape (64, 1, 512) ready for the parent.

def moe_compute(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------
    # Load per‑expert weight matrices (gate, up, down)
    # ------------------------------------------------------------
    wg_raw = offchip_load(
        w_gate,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=512,
        tile_col=1792,
        par_dispatch=1,
    )
    w_gate_stream = flatten(wg_raw, min_rank=0, max_rank=1)   # (8,512,1792)

    wu_raw = offchip_load(
        w_up,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=512,
        tile_col=1792,
        par_dispatch=1,
    )
    w_up_stream = flatten(wu_raw, min_rank=0, max_rank=1)     # (8,512,1792)

    wd_raw = offchip_load(
        w_down,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=1792,
        tile_col=512,
        par_dispatch=1,
    )
    w_down_stream = flatten(wd_raw, min_rank=0, max_rank=1)   # (8,1792,512)

    # ------------------------------------------------------------
    # Load per‑token routing scalars (expert_weights)
    # ------------------------------------------------------------
    ew_raw = offchip_load(
        expert_weights,
        stride=(2, 1),           # rows advance by 2 (cols) in the tile grid
        out_shape_tiled=(64, 2),
        tile_row=1,
        tile_col=1,
        par_dispatch=1,
    )
    # shape (1,64,2,1,1) -> flatten away leading 1
    expert_weights_stream = flatten(ew_raw, min_rank=1, max_rank=2)  # (64,2,1,1)

    # ------------------------------------------------------------
    # Build control mask from expert_onehot (int64)
    # ------------------------------------------------------------
    # select_gen produces a stream without tile dims
    control_raw = select_gen(expert_onehot, is_multihot=False, n=8)  # (1,64,2,8)
    # merge leading 1, token dim, and position dim → (128,8)
    control = flatten(control_raw, min_rank=1, max_rank=3)          # (128,8)

    # ------------------------------------------------------------
    # Replicate token embeddings for the two positions
    # ------------------------------------------------------------
    # add a singleton stream dimension (size 1) after the existing token dim
    normed_extra = reshape_stream(
        normed_2,
        chunk_size=1,
        rank=0,
        add_outer_dim=False,   # normed_2 has one stream dim already
    )                               # (64,1,1,512)

    # expand that singleton dim to size 2 using the routing‑weight stream as reference
    normed_rep = expand_ref(
        normed_extra,
        expert_weights_stream,
        expand_rank=1,
    )                               # (64,2,1,512)

    # flatten token × position → (128,1,512)
    normed_flat = flatten(normed_rep, min_rank=0, max_rank=1)    # (128,1,512)

    # ------------------------------------------------------------
    # Partition token stream and routing weights per expert
    # ------------------------------------------------------------
    tokens_per_expert = flat_partition(normed_flat, control, n=8)   # list[8] (t_i,1,512)

    # flatten routing weights to match the flattened token stream
    weights_flat = flatten(expert_weights_stream, min_rank=0, max_rank=1)  # (128,1,1)
    weights_per_expert = flat_partition(weights_flat, control, n=8)        # list[8] (t_i,1,1)

    # split weight matrices per expert
    gate_per_expert = parallelize(w_gate_stream, 8)   # each (1,512,1792)
    up_per_expert   = parallelize(w_up_stream,   8)   # each (1,512,1792)
    down_per_expert = parallelize(w_down_stream, 8)   # each (1,1792,512)

    # ------------------------------------------------------------
    # Compute expert outputs and apply per‑token routing weight
    # ------------------------------------------------------------
    expert_outputs = []
    for i in range(8):
        x = tokens_per_expert[i]        # (t_i,1,512)
        w = weights_per_expert[i]       # (t_i,1,1)

        # broadcast gate / up matrices to token stream shape
        gate_exp = expand_ref(gate_per_expert[i], x, expand_rank=1)   # (t_i,512,1792)
        up_exp   = expand_ref(up_per_expert[i],   x, expand_rank=1)   # (t_i,512,1792)

        # linear projections
        a = binary_matmul(x, gate_exp)   # (t_i,1,1792)
        b = binary_matmul(x, up_exp)    # (t_i,1,1792)

        # gated activation
        m = binary_mul(a, b)             # (t_i,1,1792)
        h = unary_silu(m)                # (t_i,1,1792)

        # down projection
        down_exp = expand_ref(down_per_expert[i], h, expand_rank=1)   # (t_i,1792,512)
        out = binary_matmul(h, down_exp)   # (t_i,1,512)

        # apply the scalar routing weight (broadcasted over tile cols)
        out_w = binary_mul(out, w)          # (t_i,1,512)
        expert_outputs.append(out_w)

    # ------------------------------------------------------------
    # Re‑assemble token stream from per‑expert pieces
    # ------------------------------------------------------------
    merged = flat_reassemble(expert_outputs, control)   # (1,128,1,1,512)

    # collapse the leading extra dimensions → (128,1,512)
    merged_flat = flatten(merged, min_rank=0, max_rank=2)   # (128,1,512)

    # restore token × position layout (64 tokens, 2 positions)
    token_pos = reshape_stream(
        merged_flat,
        chunk_size=2,
        rank=0,
        add_outer_dim=False,
    )                                   # (64,2,1,512)

    # sum the two routed experts per token
    final = accum_add(token_pos, rank=1)               # (64,1,512)

    return final