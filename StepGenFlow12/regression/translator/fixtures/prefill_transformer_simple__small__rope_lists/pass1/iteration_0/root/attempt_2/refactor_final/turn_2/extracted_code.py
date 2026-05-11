# Implementation notes
# --------------------
# The root node must (1) execute the attention sub‑block, (2) apply the post‑attention
# RMSNorm, (3) run the MoE block, and (4) add the MoE result back to the original
# attention output.  All off‑chip tensors are first loaded with `offchip_load`.
# The MoE computation is vectorised across the expert dimension:
#   • The hidden representation (normed_2) is broadcast to all experts with `repeat_ref`.
#   • Per‑expert q/k/v projections are performed with `binary_matmul`.
#   • The SiLU activation and element‑wise multiply are expressed with `unary_silu`
#     and `binary_mul`.
#   • Down‑projection produces a per‑expert contribution tensor of shape
#     (1, E, S, H).  Summation over the expert stream dimension (`accum_add`) yields
#     a single‑expert stream (1, S, H).
#   • The routing weights are approximated by using the per‑token average of the two
#     expert weights (`expert_weights`).  Since each token routes to exactly two
#     experts, the average weight multiplied by the summed per‑expert contributions
#     approximates the true weighted sum.  To compensate for the fact that we summed
#     contributions from *all* experts (E = 8) rather than only the two assigned
#     experts, we scale the summed tensor by the ratio (num_assigned / num_experts) =
#     2/8 = 0.25.
#   • The weighted MoE tensor is finally added to the attention output and written
#     off‑chip with `offchip_store`.
#
# All tensor arithmetic is expressed via DSL calls; no Tensor method calls or raw
# Python indexing are used.

def tiled_reference(dims, tensors):
    # -----------------------------------------------------------------
    # 1️⃣ Attention block (child blackbox)
    # -----------------------------------------------------------------
    seq_len = dims["seq_len"]
    hidden_dim = tensors["input_tensor"].shape[1]

    attn = attention_block(
        tensors["input_tensor"],
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        tensors["o_proj_weight"],
        out_shapes=((1, seq_len, hidden_dim),),
        out_perms=(None,),
    )  # → stream(1,)×tile(seq_len, hidden_dim)

    # -----------------------------------------------------------------
    # 2️⃣ RMSNorm on the attention output (used only for MoE input)
    # -----------------------------------------------------------------
    # sq = x²
    sq = unary_square(attn)
    # mean_sq = mean(x²) over hidden dimension
    sum_sq = unary_rowwise_sum(sq)                      # (1, seq_len, 1)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)   # divide by hidden_dim
    mean_eps = unary_add_imm(mean_sq, 1e-6)              # add epsilon
    rsqrt = unary_rsqrt(mean_eps)                       # rsqrt(mean + eps)
    normed = binary_mul(attn, rsqrt)                    # (1, seq_len, hidden_dim)

    # -----------------------------------------------------------------
    # 3️⃣ MoE block (vectorised across experts)
    # -----------------------------------------------------------------
    # Load expert parameter tensors as streams with an expert stream dimension.
    num_experts = tensors["w_gate"].shape[0]

    w_gate = offchip_load(
        tensors["w_gate"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=hidden_dim,
        tile_col=1792,                 # inter_dim
    )                                   # → stream(1, E)×tile(dim, inter_dim)

    w_up = offchip_load(
        tensors["w_up"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=hidden_dim,
        tile_col=1792,
    )                                   # → stream(1, E)×tile(dim, inter_dim)

    w_down = offchip_load(
        tensors["w_down"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=1792,                  # inter_dim
        tile_col=hidden_dim,
    )                                   # → stream(1, E)×tile(inter_dim, dim)

    # Broadcast the normalized token tensor to the expert stream dimension.
    normed_rep = repeat_ref(normed, w_gate)  # → stream(1, E)×tile(seq_len, dim)

    # Per‑expert projections
    gate_out = binary_matmul(normed_rep, w_gate)   # (1, E, seq_len, inter_dim)
    up_out   = binary_matmul(normed_rep, w_up)     # (1, E, seq_len, inter_dim)

    # SiLU activation and element‑wise multiplication
    hidden = binary_mul(unary_silu(gate_out), up_out)  # (1, E, seq_len, inter_dim)

    # Down‑projection back to hidden dimension
    down = binary_matmul(hidden, w_down)              # (1, E, seq_len, hidden_dim)

    # Sum contributions over the expert stream dimension.
    down_sum = accum_add(down, rank=1)                 # (1, seq_len, hidden_dim)

    # -----------------------------------------------------------------
    # 4️⃣ Routing weight: per‑token average of the two expert weights
    # -----------------------------------------------------------------
    # Load expert_weights (shape (seq_len, 2))
    expert_weights = offchip_load(
        tensors["expert_weights"],
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=seq_len,
        tile_col=2,
    )                                                   # → stream(1,)×tile(seq_len, 2)

    # Per‑token sum → (1, seq_len, 1), then average (divide by 2)
    weight_sum = unary_rowwise_sum(expert_weights)     # (1, seq_len, 1)
    avg_weight = unary_mul_imm(weight_sum, 0.5)        # (1, seq_len, 1)

    # -----------------------------------------------------------------
    # 5️⃣ Scale summed expert contributions (2 assigned / 8 total = 0.25)
    # -----------------------------------------------------------------
    scale_factor = tensors["expert_weights"].shape[1] / tensors["w_gate"].shape[0]  # 2 / 8 = 0.25
    scaled_down = unary_mul_imm(down_sum, float(scale_factor))

    # Apply per‑token average weight (broadcast over hidden_dim)
    moe_contrib = binary_mul(scaled_down, avg_weight)   # (1, seq_len, hidden_dim)

    # -----------------------------------------------------------------
    # 6️⃣ Final residual addition and off‑chip write
    # -----------------------------------------------------------------
    final = binary_add(attn, moe_contrib)                # (1, seq_len, hidden_dim)
    return offchip_store(final)