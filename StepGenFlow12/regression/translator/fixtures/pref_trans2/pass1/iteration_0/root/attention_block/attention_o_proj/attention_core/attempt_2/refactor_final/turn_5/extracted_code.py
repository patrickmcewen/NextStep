def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Reinterpret Q as (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32)
    Qh = restream(
        Q,
        stride=(8192, 2048, 32, 1),          # (kv_head, q_per_kv, seq_len, head_dim) strides
        out_shape_tiled=(4, 4, 64, 32),
    )

    # Reinterpret K as (kv_head=4, 1, seq_len=64, head_dim=32)
    Kh = restream(
        K,
        stride=(2048, 0, 32, 1),             # broadcast over q_per_kv
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Reinterpret V as (kv_head=4, 1, seq_len=64, head_dim=32)
    Vh = restream(
        V,
        stride=(2048, 0, 32, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Core attention (blackbox) – produces stream (4,4)×tile(64,32)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),
        out_perms=(None,),
    )

    # Convert (4,4)×tile(64,32) → (1,64)×tile(16,32)
    attn_flat = flatten(attn, min_rank=0, max_rank=1)   # merge kv_head & q_per_kv → stream(16,)
    attn_outer = promote_outer(attn_flat)                # add leading batch dim → stream(1,16)
    out = retile_streamify(
        attn_outer,
        chunk=16,            # heads per token
        split_row=True,
    )                         # stream(1,64)×tile(16,32)

    return out