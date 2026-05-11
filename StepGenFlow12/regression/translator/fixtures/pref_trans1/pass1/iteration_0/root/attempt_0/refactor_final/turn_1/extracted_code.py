# The root node must compose the full transformer layer.
# 1️⃣  Run the attention block (blackbox) to obtain the residual stream.
# 2️⃣  Apply post‑attention RMSNorm (pure compute DSL ops).
# 3️⃣  Perform the MoE top‑k routing with flat‑partition / flat‑reassemble.
#    – duplicate tokens for each top‑k position,
#    – partition them per expert using the one‑hot routing mask,
#    – broadcast each expert’s weight matrices (w_gate, w_up, w_down) to the
#      token subset, compute the expert contribution, and weight it by the
#      scalar routing weight,
#    – re‑assemble the ragged per‑expert results, sum over the two top‑k
#      entries, and finally add the MoE output back to the attention result.
# 4️⃣  Store the final stream off‑chip.
def tiled_reference(dims, tensors):
    # ------------------------------------------------------------------
    # Shape / dimension helpers (pure Python, no tensor ops)
    # ------------------------------------------------------------------
    seq_len = dims["seq_len"]                     # 64
    hidden = tensors["input_tensor"].shape[1]    # 512
    topk = tensors["expert_onehot"].shape[1]     # 2
    n_experts = tensors["w_gate"].shape[0]       # 8
    inter_dim = tensors["w_gate"].shape[2]       # 1792

    # ------------------------------------------------------------------
    # 1️⃣ Attention block (produces the residual stream)
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
    )  # shape (seq_len, 1, hidden)

    # ------------------------------------------------------------------
    # 2️⃣ Post‑attention RMSNorm (x * rsqrt(mean(x^2) + eps))
    # ------------------------------------------------------------------
    sq = unary_square(att)                              # x²
    sum_sq = unary_rowwise_sum(sq)                      # sum over hidden
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden)       # divide by hidden
    mean_eps = unary_add_imm(mean_sq, 1e-6)             # add epsilon
    rsqrt = unary_rsqrt(mean_eps)                       # rsqrt
    normed = binary_mul(att, rsqrt)                     # RMSNorm output

    # ------------------------------------------------------------------
    # 3️⃣ MoE routing & expert computation
    # ------------------------------------------------------------------
    # 3.1 Duplicate each token for the two top‑k positions
    normed_dup = repeat_static(normed, factor=topk)    # (S, topk, 1, hidden)
    # flatten the (S, topk) stream into a single stream dimension
    normed_flat = flatten(normed_dup, min_rank=0, max_rank=1)  # (S*topk, 1, hidden)

    # 3.2 Control mask: one‑hot expert selection per (token, topk) pair
    control = select_gen(tensors["expert_onehot"], is_multihot=False, n=n_experts)
    # control shape after select_gen: (1, seq_len, topk, n_experts)

    # 3.3 Load the per‑token scalar routing weights and flatten similarly
    wgt = offchip_load(
        tensors["expert_weights"],
        stride=(1, 1),
        out_shape_tiled=(seq_len, topk),
        tile_row=1,
        tile_col=1,
    )  # (seq_len, topk, 1, 1)
    wgt_flat = flatten(wgt, min_rank=0, max_rank=1)  # (S*topk, 1, 1)

    # 3.4 Partition tokens and scalar weights per expert
    token_chunks = flat_partition(normed_flat, control, n=n_experts)   # list length n_experts
    weight_chunks = flat_partition(wgt_flat, control, n=n_experts)    # list length n_experts

    # 3.5 Load all expert weight matrices as a stream and split per expert
    gate_all = offchip_load(
        tensors["w_gate"],
        stride=(1, 0),
        out_shape_tiled=(n_experts, 1),
        tile_row=hidden,
        tile_col=inter_dim,
    )
    up_all = offchip_load(
        tensors["w_up"],
        stride=(1, 0),
        out_shape_tiled=(n_experts, 1),
        tile_row=hidden,
        tile_col=inter_dim,
    )
    down_all = offchip_load(
        tensors["w_down"],
        stride=(1, 0),
        out_shape_tiled=(n_experts, 1),
        tile_row=inter_dim,
        tile_col=hidden,
    )
    # Split the streams into per‑expert tensors (each has shape (1, 1, …))
    gate_list = parallelize(gate_all, n_experts)
    up_list = parallelize(up_all, n_experts)
    down_list = parallelize(down_all, n_experts)

    # 3.6 Compute per‑expert contributions
    expert_outputs = []
    for e in range(n_experts):
        tok_e = token_chunks[e]          # (T_e, 1, hidden)
        wgt_e = weight_chunks[e]         # (T_e, 1, 1)

        # Broadcast the expert’s weight matrices to match the token stream
        gate_e = expand_ref(gate_list[e], tok_e, 2)   # (T_e, 1, hidden, inter)
        up_e   = expand_ref(up_list[e],   tok_e, 2)   # (T_e, 1, hidden, inter)
        down_e = expand_ref(down_list[e], tok_e, 2)   # (T_e, 1, inter, hidden)

        # Gating / up‑projection
        gate_out = binary_matmul(tok_e, gate_e)       # (T_e, 1, inter)
        up_out   = binary_matmul(tok_e, up_e)         # (T_e, 1, inter)

        # SiLU activation and down‑projection
        hidden_act = binary_mul(unary_silu(gate_out), up_out)   # (T_e, 1, inter)
        down_out   = binary_matmul(hidden_act, down_e)          # (T_e, 1, hidden)

        # Scale by the routing scalar weight
        contrib = binary_mul(down_out, wgt_e)                   # (T_e, 1, hidden)
        expert_outputs.append(contrib)

    # 3.7 Re‑assemble the ragged per‑expert results back into (seq_len, topk) order
    moe_flat = flat_reassemble(expert_outputs, control)   # (seq_len, topk, 1, 1, hidden)

    # 3.8 Merge the singleton “n_active” dim with the top‑k stream dim
    moe_merged = flatten(moe_flat, min_rank=0, max_rank=1)  # (seq_len, topk, 1, hidden)

    # 3.9 Sum the two top‑k contributions per token
    moe_contrib = accum_add(moe_merged, rank=1)  # (seq_len, 1, hidden)

    # ------------------------------------------------------------------
    # 4️⃣ Final residual addition and off‑chip store
    # ------------------------------------------------------------------
    final = binary_add(att, moe_contrib)          # (seq_len, 1, hidden)
    return offchip_store(final)