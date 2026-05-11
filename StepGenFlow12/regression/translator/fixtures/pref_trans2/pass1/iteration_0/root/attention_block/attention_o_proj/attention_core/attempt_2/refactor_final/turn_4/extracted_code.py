def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Reinterpret Q as (kv_head, q_per_kv, seq_len, head_dim)
    Qh = restream(
        Q,
        stride=(8192, 2048, 32, 1),          # row‑major mapping for (4,4,64,32)
        out_shape_tiled=(4, 4, 64, 32),
    )

    # Reinterpret K as (kv_head, 1, seq_len, head_dim)
    Kh = restream(
        K,
        stride=(2048, 0, 32, 1),             # broadcast q_per_kv dimension
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Reinterpret V as (kv_head, 1, seq_len, head_dim)
    Vh = restream(
        V,
        stride=(2048, 0, 32, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Core attention: ask the child to apply the required permutation
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=((2, 0, 1, 3),),  # corresponds to .permute(2,0,1,3) in the reference
    )

    return attn