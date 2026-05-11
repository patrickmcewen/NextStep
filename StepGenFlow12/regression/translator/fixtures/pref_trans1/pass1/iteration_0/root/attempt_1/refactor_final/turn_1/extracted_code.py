# The reference model is re‑implemented using the DSL primitives.
# Off‑chip tensors are loaded before any DSL consumer.
# The attention block is delegated to the provided child.
# RMSNorm is expressed via unary/binary ops.
# The MoE is realized with a top‑k repeat, flat‑partition per expert,
# per‑expert linear layers (gate, up, down) loaded with `offchip_load_ref`,
# weighted by `expert_weights`, re‑assembled with `flat_reassemble`,
# and summed over the top‑k dimension.
# Finally the residual is added and written off‑chip.

def tiled_reference(dims, tensors):
    # ------------------------------------------------------------------
    # 1️⃣  Attention block (child)
    # ------------------------------------------------------------------
    seq_len = tensors["input_tensor"].shape[0]          # 64
    hidden_dim = tensors["input_tensor"].shape[1]      # 512

    # Child returns a tiled stream: (seq_len, 1, hidden_dim)
    att_out = attention_block(
        tensors["input_tensor"],
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        tensors["o_proj_weight"],
        out_shapes=((seq_len, 1, hidden_dim),),
        out_perms=(None,),
    )  # shape: (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 2️⃣  Post‑attention RMSNorm  (x * rsqrt(mean(x²) + eps))
    # ------------------------------------------------------------------
    eps = 1e-6

    x_sq = unary_square(att_out)                                 # (seq_len, 1, hidden_dim)
    sum_sq = unary_rowwise_sum(x_sq)                             # (seq_len, 1, 1)
    hidden_const = unary_to_const_int(sum_sq, hidden_dim)        # (seq_len, 1, 1)
    mean = binary_div(sum_sq, hidden_const)                      # (seq_len, 1, 1)
    mean_eps = unary_add_imm(mean, eps)                          # (seq_len, 1, 1)
    inv_std = unary_rsqrt(mean_eps)                              # (seq_len, 1, 1)
    normed = binary_mul(att_out, inv_std)                        # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 3️⃣  Mixture‑of‑Experts (top‑2 routing)
    # ------------------------------------------------------------------
    num_experts = tensors["w_gate"].shape[0]          # 8
    inter_dim = tensors["w_gate"].shape[2]           # 1792

    # Repeat for the two top‑k positions → stream (seq_len, 2, 1, hidden_dim)
    token_top = repeat_static(normed, factor=2)

    # Partition tokens per expert using the one‑hot routing mask
    tokens_per_expert = flat_partition(token_top, tensors["expert_onehot"], n=num_experts)

    # Load the scalar expert weights (shape (seq_len, 2)) as a stream
    # stride (2,1) walks the (seq_len, 2) grid; tile size 1×1 creates a scalar tile.
    exp_weights_stream = offchip_load(
        underlying=tensors["expert_weights"],
        stride=(2, 1),
        out_shape_tiled=(seq_len, 2),
        tile_row=1,
        tile_col=1,
    )  # shape: (1, seq_len, 2, 1, 1)

    # Partition the scalar weights in the same way as the tokens
    weights_per_expert = flat_partition(exp_weights_stream, tensors["expert_onehot"], n=num_experts)

    # ------------------------------------------------------------------
    # Per‑expert computation
    # ------------------------------------------------------------------
    moe_expert_outputs = []   # each element will be (T_e, 1, hidden_dim)

    for e_idx in range(num_experts):
        # Tokens routed to this expert: (T_e, 1, hidden_dim)
        tok = tokens_per_expert[e_idx]

        # ------------------------------------------------------------------
        # Gate and up projection (full‑intermediate dimension)
        # ------------------------------------------------------------------
        # Broadcast the weight matrices across the token stream.
        # `out_shape_tiled` matches the token stream shape (except tile dims).
        # Stride (0,0) replicates the same tile for every token.
        gate_w = offchip_load_ref(
            ref=tok,
            underlying=tensors["w_gate"][e_idx],
            stride=(0, 0),
            out_shape_tiled=tok.shape[:-2],
            tile_row=hidden_dim,
            tile_col=inter_dim,
        )
        up_w = offchip_load_ref(
            ref=tok,
            underlying=tensors["w_up"][e_idx],
            stride=(0, 0),
            out_shape_tiled=tok.shape[:-2],
            tile_row=hidden_dim,
            tile_col=inter_dim,
        )

        gate_out = binary_matmul(tok, gate_w)   # (T_e, 1, inter_dim)
        up_out   = binary_matmul(tok, up_w)    # (T_e, 1, inter_dim)

        hidden = binary_mul(unary_silu(gate_out), up_out)   # (T_e, 1, inter_dim)

        # ------------------------------------------------------------------
        # Down projection
        # ------------------------------------------------------------------
        down_w = offchip_load_ref(
            ref=hidden,
            underlying=tensors["w_down"][e_idx],
            stride=(0, 0),
            out_shape_tiled=hidden.shape[:-2],
            tile_row=inter_dim,
            tile_col=hidden_dim,
        )
        down_out = binary_matmul(hidden, down_w)   # (T_e, 1, hidden_dim)

        # Weight by the expert scalar for each (token, top) pair
        w_scalar = weights_per_expert[e_idx]       # (T_e, 1, 1)
        weighted_down = binary_mul(down_out, w_scalar)   # (T_e, 1, hidden_dim)

        moe_expert_outputs.append(weighted_down)

    # ------------------------------------------------------------------
    # Re‑assemble the per‑expert contributions back into (seq_len, 2, 1, hidden_dim)
    # ------------------------------------------------------------------
    moe_assembled = flat_reassemble(moe_expert_outputs, tensors["expert_onehot"])
    # Sum over the auxiliary stream dimensions:
    #   - the extra dimension inserted by `flat_reassemble` (size 1)
    #   - the top‑k dimension (size 2)
    moe_output = accum_add(moe_assembled, rank=2)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 4️⃣  Final residual addition
    # ------------------------------------------------------------------
    final = binary_add(moe_output, att_out)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # Off‑chip store (root must end with offchip_store)
    # ------------------------------------------------------------------
    return offchip_store(final)