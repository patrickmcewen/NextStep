# This node simply wires the provided black‑box operators to implement the
# full attention block.  All inputs are RAW off‑chip tensors, which the
# black‑boxes can consume directly, so we do not issue any `offchip_load`
# here.  For each child we specify a stream shape (rank ≥ 3) that matches
# the expected vanilla shape:
#   - pre‑attention norm and the final O‑proj residual return (seq_len, 1,
#     hidden) → (64, 1, 512)
#   - Q has (seq_len, num_q_heads, head_dim) → (64, 16, 1, 32)
#   - K, V have (seq_len, num_kv_heads, head_dim) → (64, 4, 1, 32)
#   - The attention output has the same shape as Q.
# These shapes satisfy the stream‑shape invariant (≥ 1 stream dim + 2 tile
# dims) and are compatible with the downstream black‑boxes.
def attention_block(
    input_tensor,
    q_proj,
    k_proj,
    v_proj,
    cos,
    sin,
    o_proj_weight,
    *,
    out_shapes,
    out_perms=None,
):
    # 1️⃣ Pre‑attention RMSNorm
    normed = pre_attention_norm(
        input_tensor,
        out_shapes=((64, 1, 512),),  # stream: (seq_len, 1, hidden)
        out_perms=None,
    )

    # 2️⃣ QKV projection (produces Q, K, V streams)
    Q, K, V = qkv_projection(
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=(
            (64, 16, 1, 32),  # Q: (seq_len, n_q_heads, 1, head_dim)
            (64, 4, 1, 32),   # K: (seq_len, n_kv_heads, 1, head_dim)
            (64, 4, 1, 32),   # V: (seq_len, n_kv_heads, 1, head_dim)
        ),
        out_perms=None,
    )

    # 3️⃣ Per‑head RMSNorm
    Q, K = per_head_norm(
        Q,
        K,
        out_shapes=((64, 16, 1, 32), (64, 4, 1, 32)),
        out_perms=None,
    )

    # 4️⃣ Apply RoPE
    Q, K = apply_rope(
        Q,
        K,
        cos,
        sin,
        out_shapes=((64, 16, 1, 32), (64, 4, 1, 32)),
        out_perms=None,
    )

    # 5️⃣ GQA full‑sequence attention
    attn = attention(
        Q,
        K,
        V,
        out_shapes=((64, 16, 1, 32),),
        out_perms=None,
    )

    # 6️⃣ O‑projection + residual addition, produce final output stream
    out = o_proj_residual(
        attn,
        o_proj_weight,
        input_tensor,
        out_shapes=((64, 1, 512),),
        out_perms=None,
    )

    return out