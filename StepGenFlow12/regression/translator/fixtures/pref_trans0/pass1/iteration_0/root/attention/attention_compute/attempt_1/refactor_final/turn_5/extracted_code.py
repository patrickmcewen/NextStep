# Transform Q, K, V into the GQA layout expected by the heavy‑attention
# blackbox, invoke it, and reshape the result back to the contract’s shape.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh : (Hkv, q per kv, S, D) -----
    # Q: stream(64,)×tile(16,32) → Qh: stream(4,4)×tile(64,32)
    qh = reshape_stream(Q, chunk_size=4, rank=0)          # → stream(16,4)×tile(16,32)
    qh = accum_retile_row(qh, rank=1)                    # → stream(16,)×tile(64,32)
    qh = reshape_stream(qh, chunk_size=4, rank=0)        # → stream(4,4)×tile(64,32)

    # ----- K → Kh : (Hkv, 1, S, D) -----
    # K: stream(64,)×tile(4,32) → Kh: stream(4,1)×tile(64,32)
    kh = reshape_stream(K, chunk_size=4, rank=0)          # → stream(16,4)×tile(4,32)
    kh = accum_retile_row(kh, rank=1)                    # → stream(16,)×tile(16,32)
    kh = reshape_stream(kh, chunk_size=4, rank=0)        # → stream(4,4)×tile(16,32)
    kh = accum_retile_row(kh, rank=1)                    # → stream(4,)×tile(64,32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)        # → stream(4,1)×tile(64,32)

    # ----- V → Vh : (Hkv, 1, S, D) -----
    # V: stream(64,)×tile(4,32) → Vh: stream(4,1)×tile(64,32)
    vh = reshape_stream(V, chunk_size=4, rank=0)          # → stream(16,4)×tile(4,32)
    vh = accum_retile_row(vh, rank=1)                    # → stream(16,)×tile(16,32)
    vh = reshape_stream(vh, chunk_size=4, rank=0)        # → stream(4,4)×tile(16,32)
    vh = accum_retile_row(vh, rank=1)                    # → stream(4,)×tile(64,32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)        # → stream(4,1)×tile(64,32)

    # ----- Heavy attention computation -----
    # Ask the child to emit its native shape (4, 4, 64, 32).
    child_out_shapes = ((4, 4, 64, 32),)
    attn = attention_compute__root_attention_attention_compute(
        qh, kh, vh,
        out_shapes=child_out_shapes,
        out_perms=(None,),
    )

    # ----- Convert back to (S, H, D) = (64, 16, 32) -----
    attn = flatten(attn, min_rank=0, max_rank=1)           # → stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)  # → stream(64,)×tile(16,32)

    return attn