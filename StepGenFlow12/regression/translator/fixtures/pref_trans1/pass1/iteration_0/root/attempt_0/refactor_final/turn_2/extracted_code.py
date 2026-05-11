# Implementation reasoning:
# ----------------------------------------------------------------------
# The root node must:
#   1️⃣ Call the black‑box `attention_block` (produces a tile‑stream).
#   2️⃣ Apply the post‑attention RMSNorm using only DSL compute ops.
#   3️⃣ Execute the MoE top‑k routing:
#        • Duplicate each token for the two top‑k positions.
#        • Use `select_gen` to expose the one‑hot routing mask.
#        • Load the per‑token scalar routing weights and flatten them.
#        • Partition both the duplicated token stream and the scalar weights
#          per expert with `flat_partition`.
#        • Load the three expert weight matrices (gate, up, down) and
#          partition them per expert using the same control mask – this
#          implicitly broadcasts each expert’s matrix across all its tokens.
#        • For each expert compute:
#              gate  = token @ gate_weight
#              up    = token @ up_weight
#              hidden= silu(gate) * up
#              down  = hidden @ down_weight
#              contrib = down * scalar_weight
#        • Re‑assemble the ragged per‑expert results with `flat_reassemble`,
#          collapse the top‑k dimension, sum the two contributions, and finally
#          flatten away the leading singleton stream dimension.
#   4️⃣ Add the MoE contribution back to the original attention output and
#      write the result off‑chip.
# ----------------------------------------------------------------------
def tiled_reference(dims, tensors):
    # ------------------------------------------------------------------
    # 0️⃣  Shape helpers (pure Python)
    # ------------------------------------------------------------------
    seq_len   = dims["seq_len"]                # 64
    hidden    = tensors["input_tensor"].shape[1]   # 512
    topk      = tensors["expert_onehot"].shape[1] # 2
    n_experts = tensors["w_gate"].shape[0]        # 8
    inter_dim = tensors["w_gate"].shape[2]        # 1792

    # ------------------------------------------------------------------
    # 1️⃣  Attention block (black‑box)
    # ------------------------------------------------------------------
    att = attention_block(
        tensors["input_tensor"],
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        tensors["o_proj_weight"],
        out_shapes=((seq_len, 1, hidden),),
        out_perms=(None,),
    )  # shape: stream(seq_len,)×tile(1, hidden)

    # ------------------------------------------------------------------
    # 2️⃣  Post‑attention RMSNorm (x * rsqrt(mean(x²) + eps))
    # ------------------------------------------------------------------
    sq      = unary_square(att)                            # x²
    sum_sq  = unary_rowwise_sum(sq)                        # Σ x² over hidden
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden)          # divide by hidden
    mean_e  = unary_add_imm(mean_sq, 1e-6)                  # + eps
    rsqrt   = unary_rsqrt(mean_e)                          # rsqrt
    normed  = binary_mul(att, rsqrt)                       # RMS‑normalized stream

    # ------------------------------------------------------------------
    # 3️⃣  MoE top‑k routing & expert computation
    # ------------------------------------------------------------------
    # 3.1 Duplicate each token for the two top‑k positions
    normed_dup = repeat_static(normed, factor=topk)        # (seq_len, topk, 1, hidden)
    # Flatten into a single stream dimension: (seq_len * topk,)
    normed_flat = flatten(normed_dup, min_rank=0, max_rank=1)  # stream(seq_len*topk,)×tile(1, hidden)

    # 3.2 One‑hot routing mask (expert_onehot) → control stream
    control = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=n_experts
    )  # stream(1, seq_len)×tile(topk, n_experts)

    # 3.3 Load per‑token scalar routing weights and flatten
    wgt = offchip_load(
        tensors["expert_weights"],
        stride=(1, 1),
        out_shape_tiled=(seq_len, topk),
        tile_row=1,
        tile_col=1,
    )  # stream(1, seq_len, topk)×tile(1,1)
    wgt_flat = flatten(wgt, min_rank=0, max_rank=1)        # stream(1, seq_len*topk)×tile(1,1)

    # 3.4 Partition tokens and scalar weights per expert
    token_chunks = flat_partition(normed_flat, control, n=n_experts)   # list[len=n_experts] of stream(T_e,)×tile(1, hidden)
    wgt_chunks   = flat_partition(wgt_flat,   control, n=n_experts)   # list of stream(T_e,)×tile(1,1)

    # 3.5 Load expert weight matrices (gate, up, down)
    gate_all = offchip_load(
        tensors["w_gate"],
        stride=(1, 0),
        out_shape_tiled=(n_experts, 1),
        tile_row=hidden,
        tile_col=inter_dim,
    )  # stream(1, n_experts, 1)×tile(hidden, inter_dim)

    up_all = offchip_load(
        tensors["w_up"],
        stride=(1, 0),
        out_shape_tiled=(n_experts, 1),
        tile_row=hidden,
        tile_col=inter_dim,
    )  # stream(1, n_experts, 1)×tile(hidden, inter_dim)

    down_all = offchip_load(
        tensors["w_down"],
        stride=(1, 0),
        out_shape_tiled=(n_experts, 1),
        tile_row=inter_dim,
        tile_col=hidden,
    )  # stream(1, n_experts, 1)×tile(inter_dim, hidden)

    # 3.6 Broadcast each expert’s matrices across its tokens via flat_partition
    gate_chunks = flat_partition(gate_all, control, n=n_experts)   # list of stream(T_e,)×tile(hidden, inter_dim)
    up_chunks   = flat_partition(up_all,   control, n=n_experts)   # list of stream(T_e,)×tile(hidden, inter_dim)
    down_chunks = flat_partition(down_all, control, n=n_experts)   # list of stream(T_e,)×tile(inter_dim, hidden)

    # 3.7 Compute per‑expert contributions
    expert_outputs = []
    for e in range(n_experts):
        tok   = token_chunks[e]   # (T_e, 1, hidden)
        wgt_e = wgt_chunks[e]     # (T_e, 1, 1)

        gate_w = gate_chunks[e]   # (T_e, hidden, inter_dim)
        up_w   = up_chunks[e]     # (T_e, hidden, inter_dim)
        down_w = down_chunks[e]   # (T_e, inter_dim, hidden)

        # Gating & up‑projection
        gate_out = binary_matmul(tok, gate_w)   # (T_e, 1, inter_dim)
        up_out   = binary_matmul(tok, up_w)     # (T_e, 1, inter_dim)

        # SiLU activation and down‑projection
        hidden_act = binary_mul(unary_silu(gate_out), up_out)   # (T_e, 1, inter_dim)
        down_out   = binary_matmul(hidden_act, down_w)          # (T_e, 1, hidden)

        # Weight by the scalar routing coefficient
        contrib = binary_mul(down_out, wgt_e)                   # (T_e, 1, hidden)
        expert_outputs.append(contrib)

    # 3.8 Re‑assemble ragged token order and sum the two top‑k contributions
    moe_flat    = flat_reassemble(expert_outputs, control)     # (1, seq_len, topk, 1, 1, hidden)
    moe_merged  = flatten(moe_flat, min_rank=0, max_rank=1)    # (1, seq_len, topk, 1, hidden)
    moe_summed  = accum_add(moe_merged, rank=1)                # (1, seq_len, 1, hidden)
    # Collapse the leading singleton stream dimension → (seq_len,)×tile(1, hidden)
    moe_contrib = flatten(moe_summed, min_rank=0, max_rank=2)

    # ------------------------------------------------------------------
    # 4️⃣  Final residual addition and off‑chip store
    # ------------------------------------------------------------------
    final = binary_add(att, moe_contrib)  # (seq_len,)×tile(1, hidden)
    return offchip_store(final)