# GQA full‑sequence attention with numerically‑stable softmax, expressed only with STeP DSL ops.
#   1. Broadcast the KV heads to match Q’s query‑per‑KV‑head dimension (expand_ref).
#   2. Compute raw scores = Q @ Kᵀ.
#   3. Stable softmax:
#        – Move the key‑dimension (tile columns) into a stream (retile_streamify with split_row=False).
#        – Split the merged stream so that the column index becomes its own stream dimension (reshape_stream).
#        – Reduce that column stream with accum_max to obtain the per‑row maximum.
#        – Subtract the max from the original scores (binary_add + unary_mul_imm) and exponentiate.
#   4. Multiply by V to get the numerator and sum the exponentials for the denominator.
#   5. Divide to obtain the attention tensor.
#   6. Collapse the two head‑stream dimensions into one (flatten).
#   7. Turn the sequence‑length tile rows into a stream dimension so that the final stream shape is
#        (seq_len, num_heads, head_dim) → (64, 16, 32) as required (retile_streamify with split_row=True).
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast KV tensors across the query‑per‑KV‑head dimension.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)  # (4,4,64,32)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)  # (4,4,64,32)

    # 2. Raw attention scores.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4,4,64,64)

    # 3. Stable softmax: compute max over the key (column) dimension.
    #    a) Split the column tiles into a stream (each column becomes a separate stream element).
    scores_split_step1 = retile_streamify(scores, chunk=1, split_row=False)  # (4,256,64,1)
    #    b) Separate the merged stream dimension into (head, column) streams.
    scores_split = reshape_stream(scores_split_step1, 64, rank=0)          # (4,4,64,64,1)
    #    c) Max‑reduce over the column stream.
    row_max = accum_max(scores_split, rank=1)                           # (4,4,64,1)

    # 4. Subtract max and exponentiate.
    scores_centered = binary_add(scores, unary_mul_imm(row_max, -1.0))   # (4,4,64,64)
    e = unary_exp(scores_centered)                                      # (4,4,64,64)

    # 5. Numerator and denominator.
    num = binary_matmul(e, Vh_exp)                                      # (4,4,64,32)
    denom = unary_rowwise_sum(e)                                        # (4,4,64,1)

    # 6. Final attention values.
    attn = binary_div(num, denom)                                       # (4,4,64,32)

    # 7. Merge the two head‑stream dimensions (4 * 4 → 16).
    attn_flat = flatten(attn, min_rank=0, max_rank=1)                   # (16,64,32)

    # 8. Move sequence length into the stream dimension to get the required shape (64,16,32).
    out = retile_streamify(attn_flat, chunk=16, split_row=True)        # (64,16,32)

    return out