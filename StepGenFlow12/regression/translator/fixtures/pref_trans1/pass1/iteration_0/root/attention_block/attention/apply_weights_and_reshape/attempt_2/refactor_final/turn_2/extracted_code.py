# The function receives on‑chip tensors:
#   attn_weights: shape (4, 4, 64, 64) → stream (4,4) tile (64,64)
#   Vh:          shape (4, 1, 64, 32) → stream (4,1) tile (64,32)
# Goal: compute `attn_weights @ Vh`, then reshape the result to the vanilla
# shape (seq_len, num_heads, head_dim) = (64, 16, 32).  This is achieved
# solely with DSL primitives:
#   1. Broadcast Vh across the second KV‑head dimension (`expand_ref`).
#   2. Batched matrix multiplication (`binary_matmul`).
#   3. Collapse the two head‑related stream dimensions into one (`flatten`).
#   4. Fold that stream dimension into the tile‑row dimension
#      (`accum_retile_row`), yielding a tile‑only tensor.
#   5. Add a leading singleton stream dimension (`promote` with rank = 0)
#      so the tile‑row can be split back into a stream (seq_len) and a tile‑row
#      (num_heads) (`retile_streamify`).
#   6. Return the resulting stream tensor, which has shape (64,16,32)
#      — exactly the required output.
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Vh across the second KV‑head stream dimension.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)

    # 2. Weighted sum over values (batched matmul).
    weighted = binary_matmul(attn_weights, Vh_exp)

    # 3. Merge the two head‑related stream dimensions into one.
    merged_heads = flatten(weighted, min_rank=0, max_rank=1)

    # 4. Fold that stream dimension into the tile‑row dimension, producing a tile‑only tensor.
    merged_tile = accum_retile_row(merged_heads, rank=1)

    # 5. Insert a leading singleton stream dim so we can split the tile‑row.
    merged_tile_promoted = promote(merged_tile, rank=0)

    # Number of heads = Hkv * Q_per_KV.
    num_heads = attn_weights.shape[0] * attn_weights.shape[1]

    # 6. Split the combined tile‑row (seq_len * num_heads) into a stream (seq_len)
    #    and a tile‑row (num_heads).
    result = retile_streamify(merged_tile_promoted, chunk=num_heads, split_row=True)

    return result