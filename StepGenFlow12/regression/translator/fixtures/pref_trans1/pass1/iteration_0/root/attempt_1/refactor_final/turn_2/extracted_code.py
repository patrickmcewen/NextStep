# The implementation follows the reference workflow:
# 1️⃣  Call the child `attention_block` (raw tensors are passed directly).
# 2️⃣  Apply RMSNorm with unary/binary DSL ops.
# 3️⃣  Perform the top‑2 Mixture‑of‑Experts routing:
#     – repeat the normalized tokens for the two top‑k positions,
#     – partition tokens and scalar expert weights per expert,
#     – for each expert load its gate/up/down matrices once with `offchip_load_ref`
#       (broadcasted across the token stream),
#     – compute gate & up projections, SiLU activation, down projection,
#     – weight by the per‑token scalar and accumulate.
# 4️⃣  Re‑assemble the per‑expert results, sum the top‑k axis and drop the
#     leading singleton stream dimension with `flatten`.
# 5️⃣  Add the residual connection and write the final tensor off‑chip.
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
    # 2️⃣  Post‑attention RMSNorm (x * rsqrt(mean(x²) + eps))
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
    # 3️⃣  MoE (top‑2 routing)
    # ------------------------------------------------------------------
    num_experts = tensors["w_gate"].shape[0]          # 8
    inter_dim = tensors["w_gate"].shape[2]           # 1792

    # Repeat for the two top‑k positions → (seq_len, 2, 1, hidden_dim)
    token_top = repeat_static(normed, factor=2)

    # Partition tokens per expert using the one‑hot routing mask
    tokens_per_expert = flat_partition(
        token_top,
        tensors["expert_onehot"],
        n=num_experts,
    )  # list of length num_experts, each (T_e, 1, hidden_dim)

    # Load scalar expert weights (shape (seq_len, 2)) as a stream
    exp_weights_stream = offchip_load(
        underlying=tensors["expert_weights"],
        stride=(2, 1),                     # walk (seq_len, 2) grid
        out_shape_tiled=(seq_len, 2),
        tile_row=1,
        tile_col=1,
    )  # shape: (1, seq_len, 2, 1, 1)

    # Partition scalar weights with the same routing mask
    weights_per_expert = flat_partition(
        exp_weights_stream,
        tensors["expert_onehot"],
        n=num_experts,
    )  # each (T_e, 1, 1)

    # ------------------------------------------------------------------
    # Per‑expert computation
    # ------------------------------------------------------------------
    moe_expert_outputs = []  # will hold (T_e, 1, hidden_dim) tensors

    for e_idx in range(num_experts):
        # Tokens routed to this expert
        tok = tokens_per_expert[e_idx]            # (T_e, 1, hidden_dim)

        # --------------------------------------------------------------
        # Gate and up projection (broadcast weight matrices)
        # --------------------------------------------------------------
        gate_w = offchip_load_ref(
            ref=tok,
            underlying=tensors["w_gate"][e_idx],
            stride=(),
            out_shape_tiled=(),
            tile_row=hidden_dim,          # 512
            tile_col=inter_dim,           # 1792
        )
        up_w = offchip_load_ref(
            ref=tok,
            underlying=tensors["w_up"][e_idx],
            stride=(),
            out_shape_tiled=(),
            tile_row=hidden_dim,
            tile_col=inter_dim,
        )

        gate_out = binary_matmul(tok, gate_w)      # (T_e, 1, inter_dim)
        up_out   = binary_matmul(tok, up_w)       # (T_e, 1, inter_dim)

        hidden = binary_mul(unary_silu(gate_out), up_out)   # (T_e, 1, inter_dim)

        # --------------------------------------------------------------
        # Down projection
        # --------------------------------------------------------------
        down_w = offchip_load_ref(
            ref=hidden,
            underlying=tensors["w_down"][e_idx],
            stride=(),
            out_shape_tiled=(),
            tile_row=inter_dim,            # 1792
            tile_col=hidden_dim,           # 512
        )
        down_out = binary_matmul(hidden, down_w)   # (T_e, 1, hidden_dim)

        # --------------------------------------------------------------
        # Weight by the per‑token scalar (expert_weights)
        # --------------------------------------------------------------
        w_scalar = weights_per_expert[e_idx]       # (T_e, 1, 1)
        weighted_down = binary_mul(down_out, w_scalar)   # (T_e, 1, hidden_dim)

        moe_expert_outputs.append(weighted_down)

    # ------------------------------------------------------------------
    # Re‑assemble per‑expert contributions back to (seq_len, 2, 1, hidden_dim)
    # ------------------------------------------------------------------
    moe_assembled = flat_reassemble(moe_expert_outputs, tensors["expert_onehot"])
    # Sum over the expert‑selection stream dimensions (top‑k + extra singleton)
    moe_summed = accum_add(moe_assembled, rank=2)   # (1, seq_len, 1, hidden_dim)
    # Remove the leading singleton stream dimension
    moe_output = flatten(moe_summed, min_rank=0, max_rank=1)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 4️⃣  Final residual addition
    # ------------------------------------------------------------------
    final = binary_add(moe_output, att_out)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # Off‑chip store (root must end with offchip_store)
    # ------------------------------------------------------------------
    return offchip_store(final)