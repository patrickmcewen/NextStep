# Compute scaled dot‑product attention directly with DSL ops.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # Q: stream(64,)×tile(16,32)
    # K: stream(64,)×tile(4,32)
    # V: stream(64,)×tile(4,32)

    # 1. Compute raw attention scores Q · Kᵀ → (16 × 4) tile.
    scores = binary_matmul(Q, K, weight_transposed=True)  # stream(64,)×tile(16,4)

    # 2. Scale by 1/√head_dim (head_dim = tile_col = 32).
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = unary_mul_imm(scores, scale)                # stream(64,)×tile(16,4)

    # 3. Softmax over the KV‑head dimension (tile‑col).
    exp_scores = unary_exp(scores)                       # stream(64,)×tile(16,4)
    sum_exp = unary_rowwise_sum(exp_scores)              # stream(64,)×tile(16,1)
    attn_weights = binary_div(exp_scores, sum_exp)       # stream(64,)×tile(16,4)

    # 4. Weighted sum of V values → final output (16 × 32) tile.
    attn = binary_matmul(attn_weights, V)                # stream(64,)×tile(16,32)

    return attn