# Transform Q, K, V into the shapes expected by the heavy‑attention
# blackbox, then invoke it and return its result directly (the child is
# asked to emit the final contract shape).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh : (Hkv, Q_per_KV, S, D) -----
    # Swap the token stream (64) with the head tile rows (16) using a
    # merge‑then‑retile pattern, then split the resulting head stream
    # into the two GQA axes.
    q_tmp = accum_retile_row(Q, rank=1)                     # → stream()×tile(1024,32)
    q_tmp = promote_outer(q_tmp)                           # → stream(1,)×tile(1024,32)
    q_tmp = retile_streamify(q_tmp, chunk=64, split_row=True)  # → stream(16,)×tile(64,32)
    Qh = reshape_stream(q_tmp, chunk_size=4, rank=0)       # → stream(4,4)×tile(64,32)

    # ----- K → Kh : (Hkv, 1, S, D) -----
    # Swap token and KV‑head dimensions, then add a singleton stream dim.
    k_tmp = accum_retile_row(K, rank=1)                     # → stream()×tile(256,32)
    k_tmp = promote_outer(k_tmp)                           # → stream(1,)×tile(256,32)
    k_tmp = retile_streamify(k_tmp, chunk=64, split_row=True)  # → stream(4,)×tile(64,32)
    Kh = repeat_static(k_tmp, factor=1)                    # → stream(4,1)×tile(64,32)

    # ----- V → Vh : (Hkv, 1, S, D) -----
    v_tmp = accum_retile_row(V, rank=1)                     # → stream()×tile(256,32)
    v_tmp = promote_outer(v_tmp)                           # → stream(1,)×tile(256,32)
    v_tmp = retile_streamify(v_tmp, chunk=64, split_row=True)  # → stream(4,)×tile(64,32)
    Vh = repeat_static(v_tmp, factor=1)                    # → stream(4,1)×tile(64,32)

    # ----- Heavy attention computation -----
    # The child is asked to emit the contract’s final shape directly.
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn