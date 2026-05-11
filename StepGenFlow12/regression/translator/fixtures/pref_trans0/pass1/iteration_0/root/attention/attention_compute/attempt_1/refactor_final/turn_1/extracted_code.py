# Transform Q, K, V into the GQA layout expected by the heavy‑attention
# blackbox, invoke the child, then reshape the result back to (seq_len,
# num_heads, head_dim) = (64, 16, 32).  All shape changes use DSL ops; no raw
# tensor arithmetic or indexing is used.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh : (Hkv, qpkv, S, D)  -----
    # Q is stream(64,)×tile(16,32)
    qh = reshape_stream(Q, chunk_size=4, rank=0)          # stream(16,4)×tile(16,32)
    qh = accum_retile_row(qh, rank=1)                    # stream(16,)×tile(64,32)
    qh = reshape_stream(qh, chunk_size=4, rank=0)        # stream(4,4)×tile(64,32)

    # ----- K → Kh : (Hkv, 1, S, D)  -----
    # K is stream(64,)×tile(4,32)
    kh = reshape_stream(K, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(16,)×tile(16,32)
    kh = reshape_stream(kh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(4,)×tile(64,32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ----- V → Vh : (Hkv, 1, S, D)  -----
    # V is stream(64,)×tile(4,32)
    vh = reshape_stream(V, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(16,)×tile(16,32)
    vh = reshape_stream(vh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(4,)×tile(64,32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ----- Heavy attention computation -----
    # The child expects Qh/K h/Vh in its native layout and will emit a
    # tensor of vanilla shape (4, 4, 64, 32).  We explicitly request that
    # shape via `out_shapes`.
    child_out_shapes = ((4, 4, 64, 32),)
    attn = attention_compute__root_attention_attention_compute(
        qh, kh, vh,
        out_shapes=child_out_shapes,
        out_perms=(None,),
    )

    # ----- Convert back to contract shape (64, 16, 32) -----
    # attn now has stream(4,4)×tile(64,32).  Merge the two stream dims,
    # then split the tile‑row dimension (64) back into a stream dim of 64
    # and a tile‑row of 16.
    attn = flatten(attn, min_rank=0, max_rank=1)           # stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)  # stream(64,)×tile(16,32)

    return attn