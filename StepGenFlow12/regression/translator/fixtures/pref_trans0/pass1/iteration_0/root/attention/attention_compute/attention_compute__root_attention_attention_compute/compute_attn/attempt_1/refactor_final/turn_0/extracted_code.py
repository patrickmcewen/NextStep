# Compute attention for a single head group.
#   - `e`   : (batch_heads, kv_heads_per_group, seq_len, seq_len)   → on‑chip stream
#   - `Vh`  : (batch_heads, 1, seq_len, head_dim)                  → on‑chip stream
#   Goal   : attn = (e @ Vh) / (e.sum(dim=-1, keepdim=True))
#   The stream shape of `Vh` (…,1) must be broadcast to match `e` (…,4) before the matmul.
#   This is done with `expand_ref` which expands trailing size‑1 stream dimensions.
#   Afterwards we use `binary_matmul` for the weighted sum, `unary_rowwise_sum` for the
#   denominator, and `binary_div` for the final division.
def compute_attn(e, Vh, *, out_shapes, out_perms=None):
    # Expand Vh's trailing stream dimension (size 1) to match e's second stream dim.
    Vh_exp = expand_ref(Vh, e, expand_rank=1)      # shape → (4, 4, 64, 32)

    # Numerator: matrix multiplication over the last two tile dimensions.
    num = binary_matmul(e, Vh_exp)                # shape → (4, 4, 64, 32)

    # Denominator: sum over the last tile dimension, keepdim → (4, 4, 64, 1)
    denom = unary_rowwise_sum(e)

    # Final attention output.
    attn = binary_div(num, denom)                # broadcast division → (4, 4, 64, 32)

    return attn