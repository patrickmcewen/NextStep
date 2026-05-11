# The computation consists of three steps:
# 1️⃣ Broadcast `Vh` to the same stream shape as `attn_weights`.
# 2️⃣ Perform the batched matmul to get a tensor of shape
#    (kv, qpkv, seq_len, head_dim) → stream(4, 4)×tile(64, 32).
# 3️⃣ Re‑order the data so that the final layout is
#    (seq_len, num_heads, head_dim) = (64, 16, 32).
#    This is achieved without any Python‐level reshapes or permutes:
#    • `flatten` merges the two stream dimensions (kv, qpkv) into one
#      (size 16). Result: stream(16)×tile(64, 32).
#    • `retile_streamify(chunk=1)` collapses the tile‑row dimension into the
#      stream, turning the tensor into stream(1024)×tile(1, 32)
#      (1024 = 16 × 64).
#    • `reshape_stream(chunk_size=num_heads, rank=0)` splits that long stream
#      back into (seq_len, num_heads) while keeping tile rows = 1.
#      The final tensor has stream(64, 16)×tile(1, 32), which corresponds to the
#      required vanilla shape (64, 16, 32).
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1️⃣ Broadcast Vh across the second stream dimension of attn_weights.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)

    # 2️⃣ Weighted sum over values.
    weighted = binary_matmul(attn_weights, Vh_exp)  # stream(4,4)×tile(64,32)

    # 3️⃣ Merge the two stream dims (kv, qpkv) → stream(16)×tile(64,32)
    merged_stream = flatten(weighted, min_rank=0, max_rank=1)

    # 4️⃣ Collapse the tile‑row (seq_len) into the stream.
    lifted = retile_streamify(merged_stream, chunk=1)  # stream(1024)×tile(1,32)

    # 5️⃣ Split the long stream into (seq_len, num_heads).
    _, num_heads, _ = out_shapes[0]            # num_heads = 16 for the given config
    final = reshape_stream(lifted, chunk_size=num_heads, rank=0)

    return final