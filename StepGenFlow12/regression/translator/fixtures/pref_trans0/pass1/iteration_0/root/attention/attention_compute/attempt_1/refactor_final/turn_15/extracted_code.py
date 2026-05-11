# Transform Q, K, V into the exact layout expected by the heavy‑attention
# blackbox, then reshape the result back to the contract’s shape.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # -------------------- Q → Qh : (Hkv, Q_per_KV, S, D) --------------------
    # 1. Split the head dimension (16) into groups of 4 and merge those groups
    #    with the token stream, producing a combined stream of size 256.
    q_tmp = retile_streamify(Q, chunk=4, split_row=True)   # → stream(256,)×tile(4,32)
    # 2. Separate the combined stream into (kv_head, token) streams.
    q_tmp = reshape_stream(q_tmp, chunk_size=64, rank=0)   # → stream(4,64)×tile(4,32)
    # 3. Move the query‑per‑kv dimension (currently tile rows = 4) into the
    #    stream as an additional dimension.
    q_tmp = retile_streamify(q_tmp, chunk=1, split_row=True)   # → stream(4,256)×tile(1,32)
    # 4. Split this stream back into (q_per_kv, token) streams.
    q_tmp = reshape_stream(q_tmp, chunk_size=64, rank=0)   # → stream(4,4,64)×tile(1,32)
    # 5. Finally merge the token stream into the tile rows to obtain the
    #    required tile layout (S × D) while keeping the two GQA stream axes.
    Qh = accum_retile_row(q_tmp, rank=1)                   # → stream(4,4)×tile(64,32)

    # -------------------- K → Kh : (Hkv, 1, S, D) --------------------
    kh = reshape_stream(K, chunk_size=4, rank=0)           # → stream(16,4)×tile(4,32)
    kh = accum_retile_row(kh, rank=1)                     # → stream(16,)×tile(16,32)
    kh = reshape_stream(kh, chunk_size=4, rank=0)         # → stream(4,4)×tile(16,32)
    kh = accum_retile_row(kh, rank=1)                     # → stream(4,)×tile(64,32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)         # → stream(4,1)×tile(64,32)

    # -------------------- V → Vh : (Hkv, 1, S, D) --------------------
    vh = reshape_stream(V, chunk_size=4, rank=0)           # → stream(16,4)×tile(4,32)
    vh = accum_retile_row(vh, rank=1)                     # → stream(16,)×tile(16,32)
    vh = reshape_stream(vh, chunk_size=4, rank=0)         # → stream(4,4)×tile(16,32)
    vh = accum_retile_row(vh, rank=1)                     # → stream(4,)×tile(64,32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)         # → stream(4,1)×tile(64,32)

    # -------------------- Heavy‑attention computation (child) --------------------
    # The child expects tensors of the above shapes and will emit a tensor of
    # vanilla shape (4, 4, 64, 32).  We request that shape explicitly.
    child_out_shapes = ((4, 4, 64, 32),)
    attn = attention_compute__root_attention_attention_compute(
        Qh, kh, vh,
        out_shapes=child_out_shapes,
        out_perms=(None,),
    )

    # -------------------- Convert back to the contract’s shape (S, H, D) -----
    # Merge the two GQA stream dimensions and move the token dimension back
    # to the tile, yielding stream(64,)×tile(16,32) → vanilla (64, 16, 32).
    attn = flatten(attn, min_rank=0, max_rank=1)           # → stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)  # → stream(64,)×tile(16,32)

    return attn