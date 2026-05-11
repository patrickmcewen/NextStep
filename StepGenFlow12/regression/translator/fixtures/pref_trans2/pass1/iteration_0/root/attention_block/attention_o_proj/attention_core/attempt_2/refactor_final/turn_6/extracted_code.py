def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Convert Q from (batch=1, seq_len)×tile(heads=16, head_dim=32)
    # to the GQA layout expected by the child:
    #   Qh : (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32) → stream(4,4)×tile(64,32)
    Qh = restream(
        Q,
        stride=(128, 32, 512, 1),          # (kv_head, q_per_kv, seq_len, head_dim) strides
        out_shape_tiled=(4, 4, 64, 32),
    )

    # Convert K and V to (kv_head=4, 1, seq_len=64, head_dim=32) → stream(4,1)×tile(64,32)
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

    # Core attention (blackbox) – returns stream(4,4)×tile(64,32)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),
        out_perms=(None,),
    )

    # Inverse transformation:
    #   1. Merge the two head‑related stream dimensions → stream(16)×tile(64,32)
    #   2. Add the leading batch dimension → stream(1,16)×tile(64,32)
    #   3. Split the tile‑row dimension (64) back into a stream dimension
    #      (seq_len) and tile rows (heads) → stream(1,64)×tile(16,32)
    attn_flat = flatten(attn, min_rank=0, max_rank=1)   # merge (4,4) → (16,)
    attn_outer = promote_outer(attn_flat)                # add batch dim → (1,16)
    out = retile_streamify(
        attn_outer,
        chunk=16,            # heads per token
        split_row=True,
    )                         # final shape: stream(1,64)×tile(16,32)

    return out