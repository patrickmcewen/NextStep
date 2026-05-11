# per_head_norm: apply RMS (root‑mean‑square) normalization per head.
# For each tensor (Q and K) we compute:
#   1. Square the values (unary_square)
#   2. Sum over the last (head_dim) axis (unary_rowwise_sum)
#   3. Divide by the head dimension size → mean of squares (unary_mul_imm with 1/head_dim)
#   4. Add a small epsilon for numerical stability (unary_add_imm)
#   5. Take the reciprocal square‑root (unary_rsqrt) → 1/√(mean + eps)
#   6. Multiply the original tensor by this factor (binary_mul)
# Steps 1‑5 produce a (seq_len, heads, 1) scaling tensor that broadcasts over the
# head‑dimension when multiplied in step 6. The same sequence is performed for Q
# and K independently, preserving the required output shapes (64,16,32) and
# (64,4,32) and leaving the data on‑chip.
def per_head_norm(Q, K, *, out_shapes, out_perms=None):
    # ---- Normalize Q ----
    Q_sq = unary_square(Q)                                 # Q²
    Q_sum = unary_rowwise_sum(Q_sq)                        # Σ Q² over head_dim → (seq, heads, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / Q.shape[-1])       # mean = Σ Q² / head_dim
    Q_mean_eps = unary_add_imm(Q_mean, 1e-6)               # mean + ε
    Q_scale = unary_rsqrt(Q_mean_eps)                      # 1 / √(mean + ε)
    Q_norm = binary_mul(Q, Q_scale)                        # Q * scale (broadcast over head_dim)

    # ---- Normalize K ----
    K_sq = unary_square(K)                                 # K²
    K_sum = unary_rowwise_sum(K_sq)                        # Σ K² over head_dim → (seq, heads, 1)
    K_mean = unary_mul_imm(K_sum, 1.0 / K.shape[-1])       # mean = Σ K² / head_dim
    K_mean_eps = unary_add_imm(K_mean, 1e-6)               # mean + ε
    K_scale = unary_rsqrt(K_mean_eps)                      # 1 / √(mean + ε)
    K_norm = binary_mul(K, K_scale)                        # K * scale (broadcast)

    return Q_norm, K_norm