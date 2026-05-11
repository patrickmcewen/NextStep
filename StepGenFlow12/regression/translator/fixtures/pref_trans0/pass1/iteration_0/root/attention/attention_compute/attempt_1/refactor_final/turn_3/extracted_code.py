# Convert Q, K, V into the GQA layout expected by the heavy‑attention
# blackbox, invoke it, and return the result directly (the child is asked
# to produce the final contract shape).  All shape manipulations use DSL
# primitives only; no raw tensor operations appear.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh : (Hkv, q per kv, S, D) -----
    # Q: stream(64,)×tile(16,32)  →  Qh: stream(4,4)×tile(64,32)
    qh = reshape_stream(Q, chunk_size=4, rank=0)          # stream(16,4)×tile(16,32)
    qh = accum_retile_row(qh, rank=1)                    # stream(16,)×tile(64,32)
    qh = reshape_stream(qh, chunk_size=4, rank=0)        # stream(4,4)×tile(64,32)

    # ----- K → Kh : (Hkv, 1, S, D) -----
    # K: stream(64,)×tile(4,32)   →  Kh: stream(4,1)×tile(64,32)
    kh = reshape_stream(K, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(16,)×tile(16,32)
    kh = reshape_stream(kh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(4,)×tile(64,32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ----- V → Vh : (Hkv, 1, S, D) -----
    # V: stream(64,)×tile(4,32)   →  Vh: stream(4,1)×tile(64,32)
    vh = reshape_stream(V, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(16,)×tile(16,32)
    vh = reshape_stream(vh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(4,)×tile(64,32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ----- Heavy attention computation -----
    # Request the child to emit the final contract shape directly.
    attn = attention_compute__root_attention_attention_compute(
        qh, kh, vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn