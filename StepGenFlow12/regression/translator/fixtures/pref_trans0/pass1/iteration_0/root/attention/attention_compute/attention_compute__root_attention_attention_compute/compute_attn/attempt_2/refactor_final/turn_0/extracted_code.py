# compute_attn:
#  * `e` and `Vh` are already on‑chip stream tensors (no off‑chip load needed).
#  * Weighted sum of values: perform a batched matrix multiplication across the
#    last two tile dimensions.  `binary_matmul` does exactly this, yielding a
#    tensor of shape (Hkv, qpkv, S, D) → (4, 4, 64, 32).
#  * Denominator: sum the attention scores over the last tile dimension (the
#    key dimension).  `unary_rowwise_sum` reduces the final tile axis while
#    keeping the dimensions, producing shape (4, 4, 64, 1).
#  * Final attention: element‑wise division of the numerator by the denominator.
#    `binary_div` broadcasts the denominator’s last dim (size 1) across the
#    output’s column dimension, giving the required output shape.
#  * The function returns the attention stream directly; the parent will handle
#    any further reshaping or off‑chip storage.
def compute_attn(e, Vh, *, out_shapes, out_perms=None):
    # Weighted sum of values (batched matmul over the last two tile dims)
    num = binary_matmul(e, Vh)

    # Sum of exp scores along the key dimension (keepdim → last tile dim = 1)
    denom = unary_rowwise_sum(e)

    # Attention = numerator / denominator (broadcast division)
    attn = binary_div(num, denom)

    return attn