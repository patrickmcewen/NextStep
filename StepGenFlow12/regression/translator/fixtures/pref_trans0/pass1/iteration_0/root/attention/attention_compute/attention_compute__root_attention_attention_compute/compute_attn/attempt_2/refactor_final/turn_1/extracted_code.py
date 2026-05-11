# compute_attn:
#  * `e` and `Vh` arrive already on‑chip with stream shapes (4,4) and (4,1)
#    respectively.  For a batched matrix multiplication the stream shapes must
#    match, so we first broadcast `Vh` across the second stream dimension using
#    `expand_ref`.  `expand_ref(Vh, e, expand_rank=1)` expands the trailing
#    stream dim of size 1 in `Vh` to size 4, yielding a tensor of shape
#    (4,4,64,32).
#  * Weighted sum of values: `binary_matmul(e, Vh_exp)` performs the per‑tile
#    matrix multiply, producing a tensor of shape (4,4,64,32).
#  * Denominator: sum the attention scores over the key dimension while keeping
#    the dimensions (`unary_rowwise_sum`).  This yields shape (4,4,64,1).
#  * Final attention: element‑wise division (`binary_div`) broadcasts the
#    denominator across the output’s column dimension, giving the required
#    output shape (4,4,64,32).
def compute_attn(e, Vh, *, out_shapes, out_perms=None):
    # Broadcast Vh across the query‑per‑kv‑head dimension to match e's stream.
    Vh_exp = expand_ref(Vh, e, expand_rank=1)

    # Weighted sum of values (batched matrix multiplication).
    num = binary_matmul(e, Vh_exp)

    # Denominator: sum over the last tile dimension (keep dim → size 1).
    denom = unary_rowwise_sum(e)

    # Attention output: element‑wise division (broadcast over the column dim).
    attn = binary_div(num, denom)

    return attn