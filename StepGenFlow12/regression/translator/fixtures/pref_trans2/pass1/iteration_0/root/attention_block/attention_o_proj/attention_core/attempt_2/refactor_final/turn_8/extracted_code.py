def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Transform Q from (batch=1, seq_len)×tile(heads, dim) to
    # (kv_head=4, q_per_kv=4, seq_len, head_dim) → stream(4,4)×tile(64,32)
    Qh = restream(
        Q,
        stride=(128, 32, 512, 1),          # (kv_head, q_per_kv, seq_len, head_dim) strides
        out_shape_tiled=(4, 4, 64, 32),
    )

    # Transform K and V from (batch=1, seq_len)×tile(kv_heads, dim)
    # to (kv_head, 1, seq_len, head_dim) → stream(4,1)×tile(64,32)
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

    # Core attention computation (blackbox)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),      # child‑specific output shape
        out_perms=(None,),
    )

    # Inverse transform: (kv, q_per_kv)×tile(seq, dim) → (batch=1, seq_len)×tile(heads, dim)
    out = restream(
        attn,
        stride=(0, 32, 2048, 1),            # (batch, seq_len, heads, head_dim) strides
        out_shape_tiled=(1, 64, 16, 32),
    )

    return out