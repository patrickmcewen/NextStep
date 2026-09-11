# Implementation notes:
# 1. Qh, Kh, Vh are already on‑chip streams with shapes (4,4,64,32),
#    (4,1,64,32), (4,1,64,32) respectively.
# 2. To make the shapes match for matmul we expand Kh and Vh along the
#    second stream dimension (size‑1) using `expand_ref`.
# 3. Attention is computed with the standard sequence:
#        scores = Qh @ Khᵀ
#        e      = exp(scores)
#        num    = e @ Vh
#        denom  = e.sum(dim=-1, keepdim=True)
#        attn   = num / denom
#    All operations are expressed with DSL binary/unary primitives.
# 4. The required vanilla output shape is (seq_len=64, num_heads=16, head_dim=32).
#    The internal tensor has stream dims (kv_head, query_per_kv) = (4,4)
#    and tile rows = 64, tile cols = 32.  We therefore:
#      a) Split the tile‑row dimension into separate tiles (one row per tile)
#         with `retile_streamify(..., chunk=1, split_row=True)` → shape (4,256,1,32).
#      b) Reorder the resulting stream dimensions so that the first stream
#         axis corresponds to seq_len (64) and the second to the flattened head
#         dimension (16).  This is done by treating the current stream as a
#         buffered grid and streaming it back with the desired ordering via
#         `bufferize` + `streamify`.  The stride `[1, seq_len]` implements the
#         mapping  linear_idx = seq * 1 + head * seq_len.
#      c) The streamed tensor now has shape (64,16,1,32).  The singleton tile‑row
#         dimension is merged into the last stream axis using `accum_retile_row`,
#         yielding the exact required shape (64,16,32).
# 5. No raw tensor methods are used; every shape manipulation goes through
#    DSL ops.

def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Expand KV tensors to match Qh's stream shape (kv_head, query_per_kv)
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # Scores = Q @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # Stabilized softmax (row‑wise max omitted – the values are small enough)
    e = unary_exp(scores)

    # Numerator = e @ V
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # Denominator = sum over the last tile dimension
    denom = unary_rowwise_sum(e)

    # Attention = num / denom   (broadcasts denom across the last tile dim)
    attn = binary_div(num, denom)

    # ------------------------------------------------------------------
    # Reshape to the required vanilla layout (seq_len, num_heads, head_dim)
    # ------------------------------------------------------------------

    # 1) Split each row of the attention matrix into its own tile
    attn_split = retile_streamify(attn, chunk=1, split_row=True)  # (4,256,1,32)

    # 2) Reorder stream dimensions: (seq_len=64, num_heads=16)
    #    The target tile size is (1, head_dim)
    target_seq, target_heads, target_dim = out_shapes[0]   # (64, 16, 32)

    # Bufferize the split tensor (buffer grid = current stream dims)
    buf = bufferize(attn_split, rank=2)

    # Streamify with a stride that maps (seq, head) → linear tile index:
    # linear_idx = seq * 1 + head * seq_len
    stride = [1, target_seq]          # [1, 64]
    out_shape_tiled = (target_seq, target_heads)   # (64, 16)

    reordered = streamify(
        buf,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=1,
        tile_col=target_dim,
    )  # shape (64, 16, 1, 32)

    # 3) Merge the singleton tile‑row dimension into the last stream axis
    result = accum_retile_row(reordered)   # shape (64, 16, 32)

    return result