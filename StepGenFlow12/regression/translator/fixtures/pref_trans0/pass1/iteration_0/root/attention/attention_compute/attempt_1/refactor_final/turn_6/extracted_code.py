# Convert Q, K, V into the GQA layout expected by the heavy‑attention
# blackbox, invoke it, and reshape the result back to the contract’s shape.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh : (Hkv, q per kv, S, D) -----
    # Start: stream(64,)×tile(16,32)
    # 1. Move the head‑rows (tile rows) into the stream.
    q_tmp = retile_streamify(Q, chunk=1, split_row=True)   # → stream(1024,)×tile(1,32)
    # 2. Split the stream (1024) into (16, 64) so that the 64‑length part
    #    will become the token dimension.
    q_tmp = reshape_stream(q_tmp, chunk_size=64, rank=0)    # → stream(16,64)×tile(1,32)
    # 3. Merge the 64‑stream dim with the tile‑row dim (currently 1) to obtain
    #    tile rows = token length.
    q_tmp = accum_retile_row(q_tmp, rank=1)                # → stream(16,)×tile(64,32)
    # 4. Split the remaining stream (16) into the two GQA axes.
    Qh = reshape_stream(q_tmp, chunk_size=4, rank=0)       # → stream(4,4)×tile(64,32)

    # ----- K → Kh : (Hkv, 1, S, D) -----
    # Start: stream(64,)×tile(4,32)
    k_tmp = retile_streamify(K, chunk=1, split_row=True)   # → stream(256,)×tile(1,32)
    k_tmp = reshape_stream(k_tmp, chunk_size=64, rank=0)    # → stream(4,64)×tile(1,32)
    k_tmp = accum_retile_row(k_tmp, rank=1)                # → stream(4,)×tile(64,32)
    Kh = reshape_stream(k_tmp, chunk_size=1, rank=0)        # → stream(4,1)×tile(64,32)

    # ----- V → Vh : (Hkv, 1, S, D) -----
    v_tmp = retile_streamify(V, chunk=1, split_row=True)   # → stream(256,)×tile(1,32)
    v_tmp = reshape_stream(v_tmp, chunk_size=64, rank=0)    # → stream(4,64)×tile(1,32)
    v_tmp = accum_retile_row(v_tmp, rank=1)                # → stream(4,)×tile(64,32)
    Vh = reshape_stream(v_tmp, chunk_size=1, rank=0)        # → stream(4,1)×tile(64,32)

    # ----- Heavy attention computation (child) -----
    # The child’s native output shape is (4, 4, 64, 32); request that explicitly.
    child_out_shapes = ((4, 4, 64, 32),)
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh,
        out_shapes=child_out_shapes,
        out_perms=(None,),
    )

    # ----- Convert (Hkv, q per kv, S, D) → (S, H, D) -----
    # Merge the two GQA stream dimensions.
    attn = flatten(attn, min_rank=0, max_rank=1)           # → stream(16,)×tile(64,32)
    # Restore the token dimension as the stream and the heads as the tile rows.
    attn = retile_streamify(attn, chunk=16, split_row=True)  # → stream(64,)×tile(16,32)

    return attn