# NOTE: The attention_compute node must reshape the on‑chip inputs Q, K, V into the
# canonical GQA layout expected by the heavy‑attention blackbox, invoke that
# blackbox, and finally reshape the result back to the contract’s vanilla shape.
# All shape manipulations are expressed using DSL primitives only.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh : (Hkv, q per kv, S, D) -----
    #   Q  : stream(64,)×tile(16,32)
    #   Goal: stream(4,4)×tile(64,32)
    #   1. Split the token stream into groups of 4.
    qh = reshape_stream(Q, chunk_size=4, rank=0)          # stream(16,4)×tile(16,32)
    # 2. Merge the inner token subgroup with the head dimension.
    qh = accum_retile_row(qh, rank=1)                    # stream(16,)×tile(64,32)
    # 3. Split the remaining stream (16) into the two GQA axes.
    qh = reshape_stream(qh, chunk_size=4, rank=0)        # stream(4,4)×tile(64,32)

    # ----- K → Kh : (Hkv, 1, S, D) -----
    #   K : stream(64,)×tile(4,32)
    #   Goal: stream(4,1)×tile(64,32)
    kh = reshape_stream(K, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(16,)×tile(16,32)
    kh = reshape_stream(kh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(4,)×tile(64,32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ----- V → Vh : (Hkv, 1, S, D) -----
    #   V : stream(64,)×tile(4,32)
    #   Goal: stream(4,1)×tile(64,32)
    vh = reshape_stream(V, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(16,)×tile(16,32)
    vh = reshape_stream(vh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(4,)×tile(64,32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ----- Heavy attention computation (child) -----
    # Request the child’s native output shape, then reshape back.
    attn = attention_compute__root_attention_attention_compute(
        qh, kh, vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # ----- Convert (Hkv, qpkv, S, D) → (S, H, D) -----
    # 1. Merge the two GQA stream dimensions.
    attn = flatten(attn, min_rank=0, max_rank=1)           # stream(16,)×tile(64,32)
    # 2. Promote the token dimension to the stream.
    attn = retile_streamify(attn, chunk=16, split_row=True)  # stream(64,)×tile(16,32)

    return attn