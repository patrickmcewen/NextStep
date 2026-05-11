# The root node computes the full transformer layer.
# 1️⃣ Run the attention sub‑block (child blackbox).
# 2️⃣ Apply RMSNorm (requires only unary DSL ops).
# 3️⃣ Vectorised MoE:
#    – compute per‑expert projections,
#    – compute a per‑token‑per‑expert scalar weight from `expert_weights`
#      and `expert_onehot`,
#    – multiply each expert’s hidden contribution by its scalar weight,
#    – sum the weighted contributions over the expert dimension,
#    – add the original attention output (residual) and store off‑chip.
#
# All tensor arithmetic is expressed via DSL calls; no raw PyTorch
# operations appear after the initial inputs.

def tiled_reference(dims, tensors):
    seq_len = dims["seq_len"]
    hidden_dim = tensors["input_tensor"].shape[1]
    num_experts = tensors["w_gate"].shape[0]          # = 8

    # -----------------------------------------------------------------
    # 1️⃣ Attention block (child blackbox)
    # -----------------------------------------------------------------
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
    sq = unary_square(attn)                                 # (1,)×tile(seq_len, hidden_dim)
    sum_sq = unary_rowwise_sum(sq)                          # (1,)×tile(seq_len, 1)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)       # divide by hidden_dim
    mean_eps = unary_add_imm(mean_sq, 1e-6)                  # + ε
    rsqrt = unary_rsqrt(mean_eps)                           # (1,)×tile(seq_len, 1)
    normed = binary_mul(attn, rsqrt)                        # (1,)×tile(seq_len, hidden_dim)

    # -----------------------------------------------------------------
    # 3️⃣ MoE – per‑expert projections
    # -----------------------------------------------------------------
    # Load per‑expert weight matrices (float tensors)
    w_gate = offchip_load(
        tensors["w_gate"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=hidden_dim,
        tile_col=1792,
    )  # → stream(1, num_experts)×tile(hidden_dim, inter_dim)

    w_up = offchip_load(
        tensors["w_up"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=hidden_dim,
        tile_col=1792,
    )  # → stream(1, num_experts)×tile(hidden_dim, inter_dim)

    w_down = offchip_load(
        tensors["w_down"],
        stride=(1,),
        out_shape_tiled=(num_experts,),
        tile_row=1792,
        tile_col=hidden_dim,
    )  # → stream(1, num_experts)×tile(inter_dim, hidden_dim)

    # Broadcast the normalized token tensor across the expert stream dimension
    normed_rep = repeat_ref(normed, w_gate)                 # → stream(1, num_experts)×tile(seq_len, hidden_dim)

    # Per‑expert gate / up projections
    gate_out = binary_matmul(normed_rep, w_gate)            # (1, num_experts)×tile(seq_len, inter_dim)
    up_out   = binary_matmul(normed_rep, w_up)              # (1, num_experts)×tile(seq_len, inter_dim)

    # SiLU activation and element‑wise multiplication
    hidden = binary_mul(unary_silu(gate_out), up_out)       # (1, num_experts)×tile(seq_len, inter_dim)

    # Down‑projection back to hidden dimension
    down = binary_matmul(hidden, w_down)                    # (1, num_experts)×tile(seq_len, hidden_dim)

    # -----------------------------------------------------------------
    # 3️⃣ MoE – compute per‑token‑per‑expert scalar weights
    # -----------------------------------------------------------------
    # Convert the int one‑hot to float (allowed as a simple dtype cast)
    expert_onehot_f = tensors["expert_onehot"].float()

    # Load the one‑hot mask (token × position × expert)
    onehot = offchip_load(
        expert_onehot_f,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=2,                      # two top‑k positions
        tile_col=num_experts,
    )  # → stream(1, seq_len)×tile(2, num_experts)

    # Load the per‑token two weights (token × position)
    expert_weights = offchip_load(
        tensors["expert_weights"],
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=2,
    )  # → stream(1, seq_len)×tile(1, 2)

    # Multiply (2‑position) weights with the one‑hot to obtain a (seq_len × num_experts) matrix
    weight_mat = binary_matmul(expert_weights, onehot)      # (1, seq_len)×tile(1, num_experts)

    # Move the expert dimension from tile to stream, keeping token as a stream dim
    weight_split = retile_streamify(weight_mat, chunk=1, split_col=True)   # (1, seq_len * num_experts)×tile(1,1)
    weight = reshape_stream(weight_split, chunk_size=num_experts, rank=0)   # (1, seq_len, num_experts)×tile(1,1)

    # -----------------------------------------------------------------
    # 3️⃣ MoE – align dimensions between ↓ and weights
    # -----------------------------------------------------------------
    # Bring the token dimension from tile rows of `down` into the stream
    down_split = retile_streamify(down, chunk=1, split_row=True)          # (1, num_experts * seq_len)×tile(1, hidden_dim)
    down = reshape_stream(down_split, chunk_size=num_experts, rank=0)    # (1, seq_len, num_experts)×tile(1, hidden_dim)

    # Move the hidden dimension of `down` into the stream so it aligns with the expanded weights
    down = retile_streamify(down, chunk=1, split_col=True)               # (1, seq_len, num_experts * hidden_dim)×tile(1,1)

    # Expand the scalar weight across the hidden dimension:
    #   – promote to add a new stream dim,
    #   – repeat that dim `hidden_dim` times,
    #   – flatten the expert & hidden dimensions together.
    weight = promote(weight, rank=0)                                      # (1, seq_len, num_experts, 1,1)
    weight = repeat_static(weight, factor=hidden_dim)                    # (1, seq_len, num_experts, hidden_dim,1,1)
    weight = flatten(weight, min_rank=0, max_rank=1)                     # (1, seq_len, num_experts * hidden_dim)×tile(1,1)

    # Element‑wise multiply the per‑expert hidden contribution by its scalar weight
    weighted = binary_mul(down, weight)                                   # (1, seq_len, num_experts * hidden_dim)×tile(1,1)

    # Restore the hidden dimension as a tile column
    weighted = reshape_stream(weighted, chunk_size=hidden_dim, rank=0)     # (1, seq_len, num_experts, hidden_dim)×tile(1,1)
    weighted = accum_retile_col(weighted, rank=1)                         # (1, seq_len, num_experts)×tile(1, hidden_dim)

    # Sum over the expert stream dimension
    moe = accum_add(weighted, rank=1)                                     # (1, seq_len)×tile(1, hidden_dim)

    # -----------------------------------------------------------------
    # 4️⃣ Residual addition (original attention output)
    # -----------------------------------------------------------------
    # Turn the original attention output into the same stream shape as `moe`
    attn_stream = retile_streamify(attn, chunk=1, split_row=True)        # (1, seq_len)×tile(1, hidden_dim)

    final = binary_add(moe, attn_stream)                                 # (1, seq_len)×tile(1, hidden_dim)

    # -----------------------------------------------------------------
    # 5️⃣ Off‑chip write
    # -----------------------------------------------------------------
    return offchip_store(final)