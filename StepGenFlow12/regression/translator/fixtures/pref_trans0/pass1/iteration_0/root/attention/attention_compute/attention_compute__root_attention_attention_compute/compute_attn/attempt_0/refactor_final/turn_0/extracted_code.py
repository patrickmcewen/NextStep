# compute_attn:
#   - ``e`` is the attention‑score matrix (shape (Hkv, qpkv, S, S)).
#   - ``Vh`` holds the per‑head value vectors (shape (Hkv, 1, S, D)).
#   - The attention output is:
#       num   = e @ Vh                     # matrix multiply, yields (Hkv, qpkv, S, D)
#       denom = e.sum(dim=-1, keepdim=True) # row‑wise sum → (Hkv, qpkv, S, 1)
#       attn  = num / denom                 # broadcast division → (Hkv, qpkv, S, D)
#   All tensors are already on‑chip streams, so we can feed them directly to the
#   DSL compute ops.  ``unary_rowwise_sum`` produces the denominator, and the
#   division broadcasts the singleton column dimension automatically.
def compute_attn(e, Vh, *, out_shapes, out_perms=None):
    # numerator: batched matrix multiplication of scores with values
    num = binary_matmul(e, Vh)                     # shape (4, 4, 64, 32)

    # denominator: row‑wise sum of the scores (keep the column dim as 1)
    denom = unary_rowwise_sum(e)                   # shape (4, 4, 64, 1)

    # final attention output: element‑wise division (broadcast over the column dim)
    attn = binary_div(num, denom)                  # shape (4, 4, 64, 32)

    # No permutation required (out_perms is None), and the shape already matches out_shapes[0].
    return attn