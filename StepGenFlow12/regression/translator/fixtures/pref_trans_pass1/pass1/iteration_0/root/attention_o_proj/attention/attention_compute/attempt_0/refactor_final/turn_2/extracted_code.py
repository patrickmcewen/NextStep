# attention_compute
# ----------------------------------------------------------------------
# Compute full‑sequence GQA attention using only DSL primitives.
#   • Qh, Kh, Vh are already on‑chip streams.
#   • Stable softmax is implemented by extracting the per‑row maximum,
#     subtracting it, exponentiating, then normalising.
#   • The final layout must be (seq_len, num_heads, head_dim) = (64, 16, 32).
#     To achieve the required ordering we:
#       1) Promote the attention‑output tile rows into a stream dimension.
#       2) Bufferize the three stream dimensions (kv, query‑per‑kv, seq).
#       3) Streamify the buffer with a custom stride that emits the
#          dimensions in the order (seq, kv, query‑per‑kv).
#       4) Flatten the inner two stream dimensions (kv, query‑per‑kv) into
#          the head dimension.
#       5) Absorb that head‑stream dimension back into the tile‑row size.
# ----------------------------------------------------------------------
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # 1️⃣ Broadcast K and V to the Qh stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # 2️⃣ Scores = Qh @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # ------------------------------------------------------------------
    # Stable soft‑max (row‑wise over the key dimension)
    # ------------------------------------------------------------------
    # a) Promote to expose a singleton innermost stream dim.
    scores_prom = promote(scores, rank=0)

    # b) Split the *column* tile (size 64) into a fresh stream dim.
    scores_split = retile_streamify(scores_prom, chunk=1, split_row=False)

    # c) Max‑reduce over that new stream dim → row_max (shape …×tile(64,1)).
    row_max = accum_max(scores_split, rank=1)

    # d) Subtract the max: scores – row_max
    row_max_neg = unary_mul_imm(row_max, -1.0)
    scores_centered = binary_add(scores, row_max_neg)

    # e) Exponential.
    e = unary_exp(scores_centered)

    # ------------------------------------------------------------------
    # Numerator = e @ V   ;   Denominator = Σₖ e   ;   Attention = num / denom
    # ------------------------------------------------------------------
    num   = binary_matmul(e, Vh_exp, weight_transposed=False)
    denom = unary_rowwise_sum(e)
    attn  = binary_div(num, denom)          # shape: stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Reshape to vanilla (seq_len, num_heads, head_dim) = (64,16,32)
    # ------------------------------------------------------------------
    # a) Promote then split the row tile (seq_len) into a stream dim.
    attn_prom   = promote(attn, rank=0)                         # (4,4,1)×tile(64,32)
    attn_split  = retile_streamify(attn_prom, chunk=1, split_row=True)
    #    → stream (kv, qp, seq)×tile (1,32)

    # b) Bufferize the three stream dims so we can reorder them.
    attn_buf = bufferize(attn_split, rank=3)                    # Buffer(kv,qp,seq)

    # c) Streamify with a stride that yields order (seq, kv, qp).
    #    Buffer shape is (kv, qp, seq) = (4, 4, 64).
    kv_dim   = attn_buf.shape[0]      # 4
    qp_dim   = attn_buf.shape[1]      # 4
    seq_dim  = attn_buf.shape[2]      # 64
    #    Linear index = kv * (qp*seq) + qp * seq + seq
    stride_seq = 1
    stride_kv  = qp_dim * seq_dim   # 4 * 64 = 256
    stride_qp  = seq_dim            # 64
    attn_ord = streamify(
        attn_buf,
        stride=[stride_seq, stride_kv, stride_qp],
        out_shape_tiled=(seq_dim, kv_dim, qp_dim)      # (64,4,4)
    )
    #    → stream (seq, kv, qp)×tile (1,32)

    # d) Merge the inner two stream dims (kv, qp) → heads.
    attn_flat = flatten(attn_ord, min_rank=0, max_rank=1)   # stream (seq, heads)

    # e) Absorb the heads stream dim into the tile‑row dimension.
    attn_out = accum_retile_row(attn_flat, rank=1)          # stream (seq)×tile (heads,32)

    return attn_out