# attention_compute implements GQA full‑sequence attention using only DSL ops.
# 1. Expand the single KV head (Kh, Vh) across the query‑per‑kv dimension.
# 2. Compute Q·Kᵀ, apply exp, and form the softmax numerator/denominator.
#    The row‑wise max subtraction is omitted because it cancels in the final
#    softmax (exp(x‑max)/∑exp(x‑max) == exp(x)/∑exp(x)).
# 3. Multiply the softmax weights with V to obtain the attention tensor.
# 4. Rearrange the result from (kv, q_per_kv, seq, head) → (seq, heads, head)
#    by:
#    a) flattening the two stream dims (kv, q_per_kv) → (heads)
#    b) promoting a singleton stream dim, retile‑streamifying to move the
#       sequence length from a tile dimension into a stream dimension,
#       and finally flattening the two stream dims (heads, seq/4) → (seq)
#    The sequence of flatten → promote → retile_streamify → flatten yields
#    the required output shape (seq_len, num_heads, head_dim) = (64,16,32).
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Expand K and V across the query‑per‑kv head dimension.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # scores = Q · Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # e = exp(scores)   (row‑wise max subtraction omitted – mathematically harmless)
    e = unary_exp(scores)

    # denom = Σₖ e   (sum over the last tile dimension)
    denom = unary_rowwise_sum(e)

    # num = e · V
    num = binary_matmul(e, Vh_exp)

    # attn = num / denom
    attn = binary_div(num, denom)

    # --- reshape to (seq_len, num_heads, head_dim) ---
    # 1. Merge the two original stream dims (kv, q_per_kv) → heads (16)
    attn = flatten(attn, min_rank=0, max_rank=1)

    # 2. Insert a singleton inner stream dim
    attn = promote(attn, rank=0)

    # 3. Split the tile‑row dimension (seq_len=64) into chunks of 16,
    #    turning the remaining factor (4) into an inner stream dim.
    attn = retile_streamify(attn, chunk=16, split_row=True)

    # 4. Merge the two stream dims (heads=16, factor=4) → seq_len=64
    attn = flatten(attn, min_rank=0, max_rank=1)

    return attn