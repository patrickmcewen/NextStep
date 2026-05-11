# Implementation respects all DSL constraints.
# Off‑chip tensors are loaded before any DSL consumer sees them.
# `expert_onehot` is turned into a stream via `select_gen`.
# Per‑expert weight matrices are loaded once with `offchip_load`,
# flattened to collapse the leading singleton stream dimensions,
# and then broadcast across the token stream with `expand_ref`.
# The MoE routing uses `flat_partition`/`flat_reassemble`,
# and the final residual addition is performed with `binary_add`.
# The root ends with `offchip_store`.

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
    # 3️⃣  Mixture‑of‑Experts (top‑2 routing)
    # ------------------------------------------------------------------
    num_experts = tensors["w_gate"].shape[0]          # 8
    inter_dim = tensors["w_gate"].shape[2]           # 1792

    # Duplicate tokens for the two top‑k positions → (seq_len, 2, 1, hidden_dim)
    token_top = repeat_static(normed, factor=2)

    # Convert the one‑hot routing mask into a tiled stream
    expert_onehot_stream = select_gen(
        tensors["expert_onehot"],
        is_multihot=False,
        n=num_experts,
    )  # shape: (1, seq_len, 2, num_experts)

    # Partition tokens per expert using the routing mask
    tokens_per_expert = flat_partition(
        token_top,
        expert_onehot_stream,
        n=num_experts,
    )  # list of length num_experts, each (T_e, 1, hidden_dim)

    # Load scalar expert weights (shape (seq_len, 2)) as a tiled stream
    exp_weights_stream = offchip_load(
        underlying=tensors["expert_weights"],
        stride=(2, 1),                     # walk the (seq_len, 2) grid row‑major
        out_shape_tiled=(seq_len, 2),
        tile_row=1,
        tile_col=1,
    )  # shape: (1, seq_len, 2, 1, 1)

    # Partition the scalar weights with the same routing mask
    weights_per_expert = flat_partition(
        exp_weights_stream,
        expert_onehot_stream,
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
        # Gate and up projection (load once, broadcast across tokens)
        # --------------------------------------------------------------
        gate_tile = offchip_load(
            underlying=tensors["w_gate"][e_idx],
            stride=(0,),
            out_shape_tiled=(1,),
            tile_row=hidden_dim,
            tile_col=inter_dim,
        )  # (1, 1, hidden_dim, inter_dim)
        gate_tile = flatten(gate_tile, min_rank=0, max_rank=1)    # (1, hidden_dim, inter_dim)
        gate_w = expand_ref(gate_tile, tok, expand_rank=1)      # (T_e, hidden_dim, inter_dim)

        up_tile = offchip_load(
            underlying=tensors["w_up"][e_idx],
            stride=(0,),
            out_shape_tiled=(1,),
            tile_row=hidden_dim,
            tile_col=inter_dim,
        )  # (1, 1, hidden_dim, inter_dim)
        up_tile = flatten(up_tile, min_rank=0, max_rank=1)        # (1, hidden_dim, inter_dim)
        up_w = expand_ref(up_tile, tok, expand_rank=1)          # (T_e, hidden_dim, inter_dim)

        gate_out = binary_matmul(tok, gate_w)      # (T_e, 1, inter_dim)
        up_out   = binary_matmul(tok, up_w)       # (T_e, 1, inter_dim)

        hidden = binary_mul(unary_silu(gate_out), up_out)   # (T_e, 1, inter_dim)

        # --------------------------------------------------------------
        # Down projection (load once, broadcast across tokens)
        # --------------------------------------------------------------
        down_tile = offchip_load(
            underlying=tensors["w_down"][e_idx],
            stride=(0,),
            out_shape_tiled=(1,),
            tile_row=inter_dim,
            tile_col=hidden_dim,
        )  # (1, 1, inter_dim, hidden_dim)
        down_tile = flatten(down_tile, min_rank=0, max_rank=1)  # (1, inter_dim, hidden_dim)
        down_w = expand_ref(down_tile, hidden, expand_rank=1)   # (T_e, inter_dim, hidden_dim)

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
    moe_assembled = flat_reassemble(moe_expert_outputs, expert_onehot_stream)
    # Sum over the top‑k dimension (size 2) and the leading singleton added by flat_reassemble
    moe_summed = accum_add(moe_assembled, rank=2)   # (1, seq_len)×tile(1, hidden_dim)
    # Collapse the leading singleton stream dimension
    moe_output = flatten(moe_summed, min_rank=0, max_rank=1)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 4️⃣  Final residual addition
    # ------------------------------------------------------------------
    final = binary_add(moe_output, att_out)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # Off‑chip store (root must end with offchip_store)
    # ------------------------------------------------------------------
    return offchip_store(final)