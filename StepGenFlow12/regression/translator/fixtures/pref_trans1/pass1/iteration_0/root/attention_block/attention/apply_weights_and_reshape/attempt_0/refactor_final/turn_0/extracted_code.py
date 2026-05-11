# The node receives on‑chip tensors `attn_weights` (stream shape 4×4, tile 64×64)
# and `Vh` (stream shape 4×1, tile 64×32).  We first broadcast `Vh` over the
# second stream dimension so both inputs have identical stream shape (4, 4).
# A batched matmul (`binary_matmul`) then yields a tensor of shape
# (4, 4, 64, 32) representing the weighted sum over values.
#
# The two stream dimensions correspond to the KV‑head and query‑per‑KV‑head.
# To obtain the final layout (seq_len, num_heads, head_dim) = (64, 16, 32)
# we:
#   1. Merge one of the stream dimensions into the tile‑row dimension with
#      `accum_retile_row(rank=1)`, producing (4, 256, 32) where 256 = 4 × 64.
#   2. Use `retile_streamify(chunk=16)` to split the 256 tile rows into
#      `num_heads = 16` rows per token and promote the remaining factor
#      (4 × 16 = 64) to the stream dimension, resulting in the required shape
#      (64, 16, 32).
# The chunk size (the number of heads) is taken from the contract’s
# `out_shapes` (the middle dimension of the desired output).
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # Broadcast Vh across the second stream dimension of attn_weights.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)
    # Weighted sum: (4, 4, 64, 32)
    weighted = binary_matmul(attn_weights, Vh_exp)
    # Merge the last stream dimension into the tile‑row dimension.
    merged = accum_retile_row(weighted, rank=1)
    # Chunk size = number of heads per token (tile‑row in the final layout).
    chunk = out_shapes[0][1]  # expected to be 16 for the given configuration
    # Reshape to (seq_len, num_heads, head_dim) = (64, 16, 32).
    attn = retile_streamify(merged, chunk=chunk)
    return attn