# This node computes the attention weighted sum and reshapes it to (seq_len, num_heads, head_dim).
# Steps:
#   1. Broadcast Vh across the second attention stream dimension so its shape matches attn_weights.
#   2. Perform a batched matrix multiplication to obtain the weighted values.
#   3. Split the sequence‑length tile (64) into chunks of size `num_heads` (16) using `retile_streamify`,
#      which moves the chunk index into a new stream dimension and merges it with the original last
#      stream dimension.
#   4. Flatten the two remaining stream dimensions (kv‑heads and the new chunk stream) into a single
#      stream dimension representing the sequence length.  The final tile stream has shape
#      (64, 16, 32), matching the required output shape.
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Vh to match attn_weights' stream shape (4, 4)
    Vh_exp = expand_ref(Vh, attn_weights, 1)

    # 2. Weighted sum over values
    weighted = binary_matmul(attn_weights, Vh_exp)

    # 3. Split the sequence‑length tile into chunks of size `num_heads` (the middle dimension of the target)
    #    `out_shapes[0]` is (seq_len, num_heads, head_dim); we take the second entry as the chunk size.
    chunk = out_shapes[0][1]  # expected to be 16
    split = retile_streamify(weighted, chunk=chunk, split_row=True)

    # 4. Merge the two stream dimensions (kv‑heads and the new chunk stream) into one,
    #    yielding a stream of length `seq_len` (64) and tile rows = num_heads (16).
    result = flatten(split, min_rank=0, max_rank=1)

    return result