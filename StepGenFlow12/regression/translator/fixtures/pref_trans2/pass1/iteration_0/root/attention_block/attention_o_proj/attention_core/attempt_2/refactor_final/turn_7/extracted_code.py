def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Transform Q to (kv_head, q_per_kv, seq_len, head_dim) layout
    Qh = restream(
        Q,
        stride=(128, 32, 512, 1),          # (kv_head, q_per_kv, seq_len, head_dim) strides
        out_shape_tiled=(4, 4, 64, 32),
    )

    # Transform K and V to (kv_head, 1, seq_len, head_dim) layout
    Kh = restream(
        K,
        stride=(32, 0, 128, 1),             # (kv_head, broadcast, seq_len, head_dim) strides
        out_shape_tiled=(4, 1, 64, 32),
    )
    Vh = restream(
        V,
        stride=(32, 0, 128, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Core attention (blackbox) – returns (kv_head, q_per_kv, seq_len, head_dim)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),
        out_perms=(None,),
    )

    # Inverse transformation back to (batch=1, seq_len)×tile(num_heads, head_dim)
    out = restream(
        attn,
        stride=(128, 32, 512, 1),
        out_shape_tiled=(1, 64, 16, 32),
    )

    return out