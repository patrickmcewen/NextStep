# Transform the on‑chip streams Q, K, V into the GQA layout expected by the
# `attention_compute` blackbox, invoke it, and then undo the layout changes.
# All shape manipulations use DSL primitives; no raw tensor methods appear.
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------ Q → Qh ---------------------------------
    # Q: (1, 64, 16, 32)  →  stream (1, 64), tile (16,32)
    Qr = retile_streamify(Q, chunk=1, split_row=True)               # (1,1024,1,32)
    Qs = reshape_stream(Qr, chunk_size=64, rank=0)                  # (1,16,64,1,32)
    Qt = reshape_stream(Qs, chunk_size=4, rank=1)                   # (1,4,4,64,1,32)
    Qh = flatten(Qt, min_rank=2, max_rank=3)                       # (4,4,64,1,32)

    # ------------------------ K → Kh --------------------------------
    # K: (1, 64, 4, 32)  →  stream (1, 64), tile (4,32)
    Kr = retile_streamify(K, chunk=1, split_row=True)               # (1,256,1,32)
    Ks = reshape_stream(Kr, chunk_size=64, rank=0)                  # (1,4,64,1,32)
    Kf = flatten(Ks, min_rank=1, max_rank=2)                       # (4,64,1,32)
    Kh = repeat_static(Kf, factor=1)                               # (4,1,64,1,32)

    # ------------------------ V → Vh --------------------------------
    Vr = retile_streamify(V, chunk=1, split_row=True)               # (1,256,1,32)
    Vs = reshape_stream(Vr, chunk_size=64, rank=0)                  # (1,4,64,1,32)
    Vf = flatten(Vs, min_rank=1, max_rank=2)                       # (4,64,1,32)
    Vh = repeat_static(Vf, factor=1)                               # (4,1,64,1,32)

    # ------------------------ Core attention ------------------------
    # Child expects a stream whose tile row size is 1 and tile col size is 32.
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 1, 32),),   # stream shape matching vanilla (4,4,64,32)
        out_perms=out_perms,
    )                                   # → (4,4,64,1,32)

    # ------------------------ Inverse transform --------------------
    a0 = promote_outer(attn)                                     # (1,4,4,64,1,32)
    a1 = flatten(a0, min_rank=1, max_rank=2)                     # (1,16,64,1,32)
    a2 = flatten(a1, min_rank=0, max_rank=1)                     # (1,1024,1,32)
    a3 = reshape_stream(a2, chunk_size=16, rank=0)               # (1,64,16,1,32)
    out = accum_retile_row(a3, rank=1)                           # (1,64,16,32)

    # Verify contract compliance
    assert out.shape == out_shapes[0], (
        f"attention_core: output shape {out.shape} does not match expected "
        f"{out_shapes[0]}"
    )
    return out