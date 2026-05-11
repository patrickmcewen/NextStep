# The child attention_compute__root_* expects its own output layout (Hkv, Q_per_KV, S, D) = (4, 4, 64, 32).  We therefore reshape the inputs into that layout, invoke the child with the correct expected output shape, and finally reshape the result back to the contract’s vanilla shape (64, 16, 32).  All shape changes use DSL primitives only.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ---- Q → Qh : (Hkv, Q_per_KV, S, D) ----
    qh = reshape_stream(Q, chunk_size=4, rank=0)          # stream(16,4)×tile(16,32)
    qh = accum_retile_row(qh, rank=1)                    # stream(16,)×tile(64,32)
    qh = reshape_stream(qh, chunk_size=4, rank=0)        # stream(4,4)×tile(64,32)

    # ---- K → Kh : (Hkv, 1, S, D) ----
    kh = reshape_stream(K, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(16,)×tile(16,32)
    kh = reshape_stream(kh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    kh = accum_retile_row(kh, rank=1)                    # stream(4,)×tile(64,32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ---- V → Vh : (Hkv, 1, S, D) ----
    vh = reshape_stream(V, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(16,)×tile(16,32)
    vh = reshape_stream(vh, chunk_size=4, rank=0)        # stream(4,4)×tile(16,32)
    vh = accum_retile_row(vh, rank=1)                    # stream(4,)×tile(64,32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)        # stream(4,1)×tile(64,32)

    # ---- Heavy‑attention computation (child) ----
    # The child’s native output shape is (4, 4, 64, 32).
    child_out_shapes = ((4, 4, 64, 32),)
    attn = attention_compute__root_attention_attention_compute(
        qh, kh, vh,
        out_shapes=child_out_shapes,
        out_perms=(None,),
    )

    # ---- Convert back to the contract’s shape (S, H, D) ----
    # Merge the two GQA stream dimensions, then move the token dimension back to the tile.
    attn = flatten(attn, min_rank=0, max_rank=1)           # stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)  # stream(64,)×tile(16,32)

    return attn