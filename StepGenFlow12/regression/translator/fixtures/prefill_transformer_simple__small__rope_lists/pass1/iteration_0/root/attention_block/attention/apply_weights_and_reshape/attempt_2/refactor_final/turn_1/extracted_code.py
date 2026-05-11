# The node receives on‑chip tensors:
#   attn_weights: shape (Hkv, Q_per_KV, S, S) → stream (4,4) tile (64,64)
#   Vh:          shape (Hkv, 1, S, D)   → stream (4,1) tile (64,32)
# We need to compute the weighted sum over values and then reshape the result
# to the vanilla shape (seq_len, num_heads, head_dim) = (64,16,32).
# The transformation is performed entirely with DSL primitives:
#   1. Broadcast `Vh` across the second KV‑head dimension using `expand_ref`.
#   2. Batched matmul (`binary_matmul`) gives shape (4,4,64,32).
#   3. Merge the two head‑related stream dimensions into one with `flatten`.
#   4. Fold that stream dimension into the tile‑row dimension using
#      `accum_retile_row`, yielding a tile‑only tensor (1024,32).
#   5. Insert a dummy leading stream dimension (`promote_outer`) so that the
#      tile‑row can be split back into a stream (seq_len) and a tile‑row
#      (num_heads) via `retile_streamify`.
#   6. Return the resulting stream tensor which has shape (64,16,32)
#      – exactly the required output.
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Vh across the second KV‑head stream dimension.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)

    # 2. Weighted sum over values (batched matmul).
    weighted = binary_matmul(attn_weights, Vh_exp)

    # 3. Collapse the two head‑related stream dims into one.
    merged_heads = flatten(weighted, min_rank=0, max_rank=1)

    # 4. Merge that stream dimension into the tile‑row dimension.
    merged_tile = accum_retile_row(merged_heads, rank=1)

    # 5. Promote a leading dummy stream dim so we can split the tile‑row.
    merged_tile_promoted = promote_outer(merged_tile)

    # Number of heads = Hkv * Q_per_KV.
    num_heads = attn_weights.shape[0] * attn_weights.shape[1]

    # 6. Split the (seq_len * num_heads) tile‑row into a stream (seq_len)
    #    and a tile‑row (num_heads).
    result = retile_streamify(merged_tile_promoted, chunk=num_heads, split_row=True)

    return result