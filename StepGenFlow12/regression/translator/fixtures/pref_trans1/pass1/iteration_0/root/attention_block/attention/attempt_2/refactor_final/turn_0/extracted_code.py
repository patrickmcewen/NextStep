# Implementation reasoning:
# - Q, K, V arrive as on‑chip streams (already tiled), so they can be fed
#   directly into the child blackboxes without any loading or tensor
#   reshaping.
# - The shapes of the intermediate streams are derived from the input
#   dimensions:
#     * seq_len, num_heads, head_dim = Q.shape
#     * kv_heads = K.shape[1]                     (number of KV heads)
#     * heads_per_kv = num_heads // kv_heads       (heads per KV head)
# - `prepare_qkv` produces three streams with vanilla shapes:
#     Qh: (kv_heads, heads_per_kv, seq_len, head_dim)
#     Kh: (kv_heads, 1,            seq_len, head_dim)
#     Vh: (kv_heads, 1,            seq_len, head_dim)
#   We supply these shapes via `out_shapes`.
# - `attention_weights` consumes Qh and Kh and yields a weight stream
#   of shape (kv_heads, heads_per_kv, seq_len, seq_len).
# - `apply_weights_and_reshape` combines the weights with Vh and produces
#   the final attention output. Its output shape is exactly the requested
#   `out_shapes` (the vanilla shape of the model's attention result).
# - No tensor‑method transformations are used; all shape manipulations
#   are pure Python scalar computations, and all heavy lifting is done
#   by the child blackboxes.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # Extract vanilla dimensions from the on‑chip inputs.
    seq_len, num_heads, head_dim = Q.shape
    kv_heads = K.shape[1]
    heads_per_kv = num_heads // kv_heads

    # 1) Prepare Q, K, V heads.
    qh, kh, vh = prepare_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (kv_heads, heads_per_kv, seq_len, head_dim),  # Qh
            (kv_heads, 1,            seq_len, head_dim),  # Kh
            (kv_heads, 1,            seq_len, head_dim),  # Vh
        ),
    )

    # 2) Compute attention weights.
    attn_weights = attention_weights(
        qh,
        kh,
        out_shapes=((kv_heads, heads_per_kv, seq_len, seq_len),),
    )

    # 3) Apply weights to V and reshape to the final attention output.
    attn = apply_weights_and_reshape(
        attn_weights,
        vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn