# attention_block implements the two-step attention computation by delegating the
# heavy lifting to the provided child blackboxes.
#   1. `qkv_preprocess` converts the normalized activations into query, key and
#      value streams.  The expected tile‑stream shapes are derived from the model
#      dimensions: (batch=1, seq_len=64, num_q_heads=16, head_dim=32) for Q and
#      (1, 64, 4, 32) for K and V.  These shapes satisfy the DSL requirement of
#      rank ≥ 3 (at least one stream dimension plus the two tile dimensions).
#   2. `attention_o_proj` performs the attention weighting and the final output
#      projection, producing a stream whose shape matches the contract‑specified
#      `out_shapes` ((1, 64, 1, 512),).
# No off‑chip loads or tensor arithmetic are performed here; raw tensors are
# passed directly to the child blackboxes, which internally handle any necessary
# loading.
def attention_block(normed, q_proj, k_proj, v_proj, cos, sin, o_proj_weight, *, out_shapes, out_perms=None):
    # Shapes for the Q, K, V streams expected by the qkv_preprocess blackbox.
    q_shape = (1, 64, 16, 32)   # (batch, seq_len, q_heads, head_dim)
    k_shape = (1, 64, 4, 32)    # (batch, seq_len, kv_heads, head_dim)
    v_shape = (1, 64, 4, 32)    # (batch, seq_len, kv_heads, head_dim)

    # Produce Q, K, V streams.
    Q, K, V = qkv_preprocess(
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(q_shape, k_shape, v_shape),
        out_perms=(None, None, None),
    )

    # Final projection; forward the caller‑provided output shape/permutation.
    o_proj_out = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return o_proj_out