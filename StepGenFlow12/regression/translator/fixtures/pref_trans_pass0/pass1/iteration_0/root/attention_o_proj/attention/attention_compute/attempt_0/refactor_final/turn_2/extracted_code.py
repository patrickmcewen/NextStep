# Implementation notes:
# - Qh, Kh, Vh are already on‑chip streams. We expand Kh and Vh along their
#   trailing singleton stream dimension so they share Qh's (kv_head, query_per_kvhead)
#   stream shape.
# - Stable softmax is implemented by subtracting the per‑row maximum before the
#   exponentiation. The row‑wise max is obtained by turning the tile‑column
#   dimension into a stream dimension (via `retile_streamify` + `reshape_stream`)
#   and then reducing with `accum_max`.
# - After the attention computation we reshape the result into the required
#   vanilla layout (seq_len=64, num_heads=16, head_dim=32) by:
#   1) Splitting each row into its own tile (`retile_streamify`).
#   2) Bufferizing and re‑ordering the stream with `streamify` (using a stride
#      that maps (seq_len, num_heads) onto the underlying tile grid).
#   3) Merging the singleton tile‑row dimension back into the stream with
#      `accum_retile_row`, producing the final shape (64, 16, 32).
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Expand KV tensors to match Qh's stream shape (kv_head, query_per_kvhead)
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # ----- GQA attention (stable softmax) -----
    # scores = Q @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # Compute per‑row max for numerical stability:
    # 1) Turn the tile‑column dimension into a stream dimension.
    scores_col_split = retile_streamify(scores, chunk=1, split_row=False)
    # 2) Reshape the combined stream dimension (query_per_kv * tile_c) into
    #    (query_per_kv, tile_c) so the column index becomes its own stream dim.
    tile_c = scores.shape[-1]            # = 64
    scores_reshaped = reshape_stream(scores_col_split,
                                     chunk_size=tile_c,
                                     rank=0)
    # 3) Reduce across the column‑index stream dimension to obtain the max.
    row_max = accum_max(scores_reshaped, rank=1)

    # Subtract the max (broadcast across tile columns) and exponentiate.
    centered_scores = binary_add(scores,
                                unary_mul_imm(row_max, -1.0))
    e = unary_exp(centered_scores)

    # Numerator = e @ V
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # Denominator = sum over the last tile dimension (softmax normalizer)
    denom = unary_rowwise_sum(e)

    # Attention = num / denom   (broadcasts denom across the tile‑col dim)
    attn = binary_div(num, denom)

    # ------------------------------------------------------------------
    # Reshape to the required vanilla layout (seq_len, num_heads, head_dim)
    # ------------------------------------------------------------------
    # Split each row of the attention matrix into a separate tile.
    attn_split = retile_streamify(attn, chunk=1, split_row=True)  # (4,256,1,32)

    # Bufferize so we can index into the stream with a custom stride.
    buf = bufferize(attn_split, rank=2)

    # Desired output shape (seq_len, num_heads, head_dim)
    target_seq, target_heads, target_dim = out_shapes[0]   # (64, 16, 32)

    # Mapping stride: linear_idx = seq * 1 + head * seq_len
    stride = [1, target_seq]                # [1, 64]
    out_shape_tiled = (target_seq, target_heads)  # (64, 16)

    # Reorder the stream dimensions using the stride mapping.
    reordered = streamify(buf,
                          stride,
                          out_shape_tiled)    # (64,16,1,32)

    # Merge the singleton tile‑row dimension into the stream axis,
    # producing tile shape (16,32) and stream shape (64).
    result = accum_retile_row(reordered)   # (64,16,32)

    return result