# This node receives on‑chip tensors `attn_weights` (shape (Hkv, Q_per_KV, S, D)) and
# `Vh` (shape (Hkv, 1, S, D)).  We first broadcast `Vh` along the second KV‑head
# dimension, perform a batched matrix multiplication, then reshape the result to
# the required layout (seq_len, num_heads, head_dim) == (S, Hkv*Q_per_KV, D).
# The transformation uses only DSL primitives:
#   1. `expand_ref` to broadcast `Vh`.
#   2. `binary_matmul` for the weighted sum.
#   3. `flatten` to collapse the two head‑related stream dimensions.
#   4. `accum_retile_row` to merge the remaining head stream dimension into the
#      tile‑row dimension, producing a tensor with no stream dimensions.
#   5. `reshape_stream` (with `add_outer_dim=True`) to split the combined tile‑row
#      (seq_len * num_heads) back into a stream (seq_len) and a tile‑row (num_heads).
#   6. A final `flatten` merges the leading singleton stream dimension with the
#      sequence‑length stream dimension, yielding the exact output shape.
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Vh across the second KV‑head stream dimension.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)

    # 2. Compute the batched matrix multiplication (attention weight × values).
    weighted = binary_matmul(attn_weights, Vh_exp)

    # 3. Collapse the two head‑related stream dimensions (Hkv, Q_per_KV) into one.
    merged_heads = flatten(weighted, min_rank=0, max_rank=1)

    # 4. Merge the remaining head stream dimension into the tile‑row dimension,
    #    eliminating all stream dimensions.
    collapsed = accum_retile_row(merged_heads, rank=1)

    # 5. Split the combined tile‑row (seq_len * num_heads) into a new stream
    #    dimension (seq_len) and a tile‑row (num_heads).  The chunk size equals
    #    the total number of heads.
    num_heads = attn_weights.shape[0] * attn_weights.shape[1]
    split = reshape_stream(collapsed, chunk_size=num_heads, add_outer_dim=True)

    # 6. Remove the leading singleton stream dimension, yielding the final shape
    #    (seq_len, num_heads, head_dim).
    result = flatten(split, min_rank=0, max_rank=1)

    return result