# The root node composes the full forward pass using DSL ops only.
# 1️⃣ Run the attention sub‑block (child blackbox) → stream (1, S, H).
# 2️⃣ Apply RMSNorm to the attention output (needed for the MoE input).
# 3️⃣ Vectorised MoE across all experts:
#    – Load per‑expert parameters with `offchip_load`.
#    – Broadcast the normalized token stream to the expert dimension (`repeat_ref`).
#    – Compute gate/up projections (`binary_matmul`), SiLU activation, element‑wise multiply.
#    – Down‑project back to hidden dimension.
#    – Sum over the expert stream (`accum_add`).
# 4️⃣ Compute a per‑token average routing weight from `expert_weights`
#    (load, sum over the two expert columns, divide by 2).
# 5️⃣ Scale the summed expert contributions by the ratio (2 routed / 8 total = 0.25)
#    and apply the per‑token average weight.
# 6️⃣ Add the MoE contribution back to the original attention output and write off‑chip.
#
# All tensor arithmetic is expressed via DSL calls; no raw Tensor methods or indexing are used.

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
    # 2️⃣ RMSNorm (used as MoE input)
    # -----------------------------------------------------------------
    sq = unary_square(attn)                                 # (1, S, H)
    sum_sq = unary_rowwise_sum(sq)                          # (1, S, 1)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)       # divide by H
    mean_eps = unary_add_imm(mean_sq, 1e-6)                  # + ε
    rsqrt = unary_rsqrt(mean_eps)                           # rsqrt(mean+ε)
    normed = binary_mul(attn, rsqrt)                        # (1, S, H)

    # -----------------------------------------------------------------
    # 3️⃣ MoE (vectorised over all experts)
    # -----------------------------------------------------------------
    num_experts = tensors["w_gate"].shape[0]   # = 8

    # Load per‑expert weight tensors.
    w_gate = offchip_load(
        tensors["w_gate"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=hidden_dim,
        tile_col=1792,
    )  # → stream(1, E)×tile(H, I)

    w_up = offchip_load(
        tensors["w_up"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=hidden_dim,
        tile_col=1792,
    )  # → stream(1, E)×tile(H, I)

    w_down = offchip_load(
        tensors["w_down"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=1792,      # intermediate dimension I
        tile_col=hidden_dim,
    )  # → stream(1, E)×tile(I, H)

    # Broadcast the normalized token tensor across the expert dimension.
    normed_rep = repeat_ref(normed, w_gate)                 # → stream(1, E)×tile(S, H)

    # Per‑expert gate and up projections.
    gate_out = binary_matmul(normed_rep, w_gate)            # (1, E, S, I)
    up_out   = binary_matmul(normed_rep, w_up)              # (1, E, S, I)

    # SiLU activation and element‑wise multiplication.
    hidden = binary_mul(unary_silu(gate_out), up_out)       # (1, E, S, I)

    # Down‑projection back to hidden dimension.
    down = binary_matmul(hidden, w_down)                    # (1, E, S, H)

    # Sum contributions over the expert stream dimension.
    down_sum = accum_add(down, rank=1)                       # (1, S, H)

    # -----------------------------------------------------------------
    # 4️⃣ Per‑token average routing weight
    # -----------------------------------------------------------------
    expert_weights = offchip_load(
        tensors["expert_weights"],
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=seq_len,
        tile_col=2,
    )  # → stream(1,1)×tile(S,2)

    weight_sum = unary_rowwise_sum(expert_weights)          # (1,1,S,1)
    avg_weight = unary_mul_imm(weight_sum, 0.5)              # divide by 2 → (1,1,S,1)

    # Collapse the redundant stream dimension so it matches `scaled_down`.
    avg_weight = flatten(avg_weight, min_rank=0, max_rank=1)  # → stream(1)×tile(S,1)

    # -----------------------------------------------------------------
    # 5️⃣ Scale summed expert contributions (2 routed / 8 total = 0.25)
    # -----------------------------------------------------------------
    scale_factor = tensors["expert_weights"].shape[1] / tensors["w_gate"].shape[0]  # 2/8 = 0.25
    scaled_down = unary_mul_imm(down_sum, float(scale_factor))   # (1, S, H)

    # Apply per‑token average weight (broadcast over hidden dimension).
    moe_contrib = binary_mul(scaled_down, avg_weight)           # (1, S, H)

    # -----------------------------------------------------------------
    # 6️⃣ Final residual addition and off‑chip write
    # -----------------------------------------------------------------
    final = binary_add(attn, moe_contrib)                       # (1, S, H)
    return offchip_store(final)