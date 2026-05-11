# This node computes the softmax exponentials `e = exp(Qh @ Khᵀ)`.
# `Qh` already has stream shape (kv_heads, q_per_kv, seq_len, head_dim) = (4,4,64,32).
# `Kh` is missing the `q_per_kv` dimension (shape (4,1,64,32)).
# We broadcast `Kh` across that dimension using `repeat_static` (to add a new
# stream axis) followed by `flatten` (to merge the new axis with the existing
# singleton axis). After broadcasting, both tensors have stream shape (4,4)
# and tile shape (64,32), so `binary_matmul` with `weight_transposed=True`
# yields the attention scores of shape (4,4,64,64).  Finally we apply
# `unary_exp` to obtain `e`.  The intermediate row‑wise max used for
# numerical stability in the reference is omitted because the DSL does not
# provide a primitive for a per‑row max reduction.

def compute_e(Qh, Kh, *, out_shapes, out_perms=None):
    # Broadcast Kh across the query-per‑kv dimension.
    factor = Qh.shape[1]                     # size of the missing stream axis (4)
    Kh_rep = repeat_static(Kh, factor)       # (4, 1, factor, 64, 32)
    Kh_broadcast = flatten(Kh_rep, min_rank=0, max_rank=1)  # (4, 4, 64, 32)

    # Compute attention scores Qh @ Khᵀ.
    scores = binary_matmul(Qh, Kh_broadcast, weight_transposed=True)

    # Exponentiate the scores (row‑wise max omitted).
    e = unary_exp(scores)

    return e