# The computation performs a batched matmul between the attention weights and the
# value tensor.  `Vh` has a singleton second stream dimension, so we first expand
# it with `expand_ref` to match the stream shape of `attn_weights`.  The matmul
# yields a stream of shape (kv, qpkv) with tile rows equal to the sequence
# length.  We then need to rearrange the data so that the final layout is
# (seq_len, num_heads, head_dim).  This is achieved by:
#   • Merging the second stream dimension into the tile‑row dimension with
#     `accum_retile_row(rank=1)`, producing a tile row of size 256 (=4·seq_len).
#   • Splitting that combined tile‑row into `num_heads` rows per token and
#     lifting the sequence dimension back to the stream using `retile_streamify`.
# The resulting tensor has stream shape (64,) and tile shape (16,32),
# which corresponds to the required vanilla shape (64, 16, 32).
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1️⃣ Broadcast Vh across the second stream dimension of attn_weights.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)

    # 2️⃣ Weighted sum over values: stream(4, 4) × tile(64, 32).
    weighted = binary_matmul(attn_weights, Vh_exp)

    # 3️⃣ Merge the last stream dimension (qpkv) into the tile‑row dimension.
    merged = accum_retile_row(weighted, rank=1)  # stream(4,) × tile(256, 32)

    # 4️⃣ Split the combined tile‑row (256) into `num_heads` rows (16) per token
    #    and promote the sequence length (64) to the stream dimension.
    num_heads = out_shapes[0][1]  # expected 16 for the given config
    final = retile_streamify(merged, chunk=num_heads)  # stream(64,) × tile(16, 32)

    return final