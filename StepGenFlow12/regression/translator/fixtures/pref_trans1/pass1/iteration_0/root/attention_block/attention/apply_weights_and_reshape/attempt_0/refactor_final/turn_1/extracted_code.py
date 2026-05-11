# The weighted sum is computed by a batched matrix multiplication between
# `attn_weights` and `Vh`.  `Vh` has a singleton second stream dimension,
# which we broadcast to match `attn_weights` using `expand_ref`.  The PyTorch
# reference then permutes the dimensions to move the sequence length to the
# leading (stream) axis and merges the KV‑head and query‑per‑KV‑head dimensions
# into the final `num_heads` axis.  We reproduce this exact permutation and
# reshape using PyTorch's `permute` and `reshape` (which are shape‑only
# operations) after the DSL‑only computation.

def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # Broadcast Vh across the second stream dimension of attn_weights.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)
    # Weighted sum over values: shape (kv, qpkv, seq_len, head_dim)
    weighted = binary_matmul(attn_weights, Vh_exp)
    # Permute to (seq_len, kv, qpkv, head_dim) and reshape
    # to (seq_len, num_heads, head_dim) as required.
    seq_len, num_heads, head_dim = out_shapes[0]
    attn = weighted.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
    return attn