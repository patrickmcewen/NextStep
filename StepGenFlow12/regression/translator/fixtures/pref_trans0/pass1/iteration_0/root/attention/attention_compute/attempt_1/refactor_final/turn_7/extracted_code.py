# Convert the on‑chip inputs Q, K, V into the GQA layout expected by the heavy‑attention
# blackbox, invoke the blackbox, then reshape the result back to the contract’s vanilla
# shape (64, 16, 32).  All shape changes use DSL primitives only.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # -------------------- Q → Qh : (Hkv, Q_per_KV, S, D) --------------------
    # 1. Split the head dimension (16) into kv‑heads (4) and queries‑per‑kv (4);
    #    move kv‑heads into the stream dimension.
    qh0 = retile_streamify(Q, chunk=4, split_row=True)          # stream(256,)×tile(4,32)
    # 2. Split the combined stream (token × kv) back into token and kv dimensions.
    qh1 = reshape_stream(qh0, chunk_size=4, rank=0)            # stream(64,4)×tile(4,32)
    # 3. Merge the two stream dimensions into one (token × kv) to prepare a swap.
    qh2 = flatten(qh1, min_rank=0, max_rank=1)                 # stream(256,)×tile(4,32)
    # 4. Reshape so that kv becomes the outer stream dimension and token the inner.
    qh3 = reshape_stream(qh2, chunk_size=64, rank=0)           # stream(4,64)×tile(4,32)
    # 5. Merge the token dimension (inner stream) with the tile‑row dimension (q).
    qh4 = accum_retile_row(qh3, rank=1)                        # stream(4,)×tile(256,32)
    # 6. Split the big tile‑row (256 = q × S) back into token (S) and q dimensions,
    #    merging q into the stream.
    qh5 = retile_streamify(qh4, chunk=64, split_row=True)      # stream(16,)×tile(64,32)
    # 7. Finally split the combined stream (kv × q) into separate kv and q streams.
    Qh = reshape_stream(qh5, chunk_size=4, rank=0)             # stream(4,4)×tile(64,32)

    # -------------------- K → Kh : (Hkv, 1, S, D) --------------------
    # Split the (single) head dimension (4) into kv‑heads (4) and a singleton q.
    kh0 = retile_streamify(K, chunk=1, split_row=True)          # stream(256,)×tile(1,32)
    kh1 = reshape_stream(kh0, chunk_size=4, rank=0)            # stream(64,4)×tile(1,32)
    kh2 = flatten(kh1, min_rank=0, max_rank=1)                 # stream(256,)×tile(1,32)
    kh3 = reshape_stream(kh2, chunk_size=64, rank=0)           # stream(4,64)×tile(1,32)
    kh4 = accum_retile_row(kh3, rank=1)                        # stream(4,)×tile(64,32)
    Kh = reshape_stream(kh4, chunk_size=1, rank=0)             # stream(4,1)×tile(64,32)

    # -------------------- V → Vh : (Hkv, 1, S, D) --------------------
    vh0 = retile_streamify(V, chunk=1, split_row=True)          # stream(256,)×tile(1,32)
    vh1 = reshape_stream(vh0, chunk_size=4, rank=0)            # stream(64,4)×tile(1,32)
    vh2 = flatten(vh1, min_rank=0, max_rank=1)                 # stream(256,)×tile(1,32)
    vh3 = reshape_stream(vh2, chunk_size=64, rank=0)           # stream(4,64)×tile(1,32)
    vh4 = accum_retile_row(vh3, rank=1)                        # stream(4,)×tile(64,32)
    Vh = reshape_stream(vh4, chunk_size=1, rank=0)             # stream(4,1)×tile(64,32)

    # -------------------- Heavy‑attention computation --------------------
    # Ask the child to emit its native layout.
    child_out_shapes = ((4, 4, 64, 32),)
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh,
        out_shapes=child_out_shapes,
        out_perms=(None,),
    )

    # -------------------- Convert back to vanilla (S, H, D) --------------------
    # Merge the two GQA stream dimensions into one, then move the token dimension
    # from tile rows into the stream (producing stream token, tile rows heads).
    attn = flatten(attn, min_rank=0, max_rank=1)               # stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)    # stream(64,16)×tile(16,32)

    return attn