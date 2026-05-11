# Transform Q, K, V into the GQA layout required by the `attention_compute`
# blackbox, invoke it, and then undo the layout to produce the contract‑
# expected output (1, 64, 16, 32).  All shape manipulations use DSL primitives;
# no raw tensor methods appear.
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Q → Qh : (4, 4, 64) stream, tile row = 1, tile col = 32
    # ------------------------------------------------------------------
    Qr = retile_streamify(Q, chunk=1, split_row=True)               # (1,1024,1,32)
    Qs = reshape_stream(Qr, chunk_size=64, rank=0)                  # (1,16,64,1,32)
    Qt = reshape_stream(Qs, chunk_size=4, rank=1)                   # (1,4,4,64,1,32)
    Qh = flatten(Qt, min_rank=2, max_rank=3)                       # (4,4,64,1,32)

    # ------------------------------------------------------------------
    # K → Kh : need a transpose from (seq, kv) → (kv, seq)
    #   1) flatten the (1, seq*kv) stream to a single stream dimension
    #   2) `parallelize` splits it round‑robin into `num_kv_heads` streams,
    #      each now ordered KV‑major (kv‑head i contains all seq positions)
    #   3) `eager_merge` concatenates those streams in KV order,
    #      yielding a flat stream with KV‑major layout.
    #   4) `reshape_stream` splits the flat stream into (kv, seq)
    #   5) `promote` inserts the required singleton dimension after kv.
    # ------------------------------------------------------------------
    Kr = retile_streamify(K, chunk=1, split_row=True)                # (1,256,1,32)
    K_flat = flatten(Kr, min_rank=0, max_rank=1)                     # (256,1,32)
    K_parts = parallelize(K_flat, n=4)                               # 4 × (64,1,32)
    K_cat, _ = eager_merge(K_parts)                                 # (256,1,32)
    K_resh = reshape_stream(K_cat, chunk_size=64, rank=0)            # (4,64,1,32)
    Kh = promote(K_resh, rank=0)                                    # (4,1,64,1,32)

    # ------------------------------------------------------------------
    # V → Vh : exactly the same pattern as K
    # ------------------------------------------------------------------
    Vr = retile_streamify(V, chunk=1, split_row=True)                # (1,256,1,32)
    V_flat = flatten(Vr, min_rank=0, max_rank=1)                     # (256,1,32)
    V_parts = parallelize(V_flat, n=4)                               # 4 × (64,1,32)
    V_cat, _ = eager_merge(V_parts)                                 # (256,1,32)
    V_resh = reshape_stream(V_cat, chunk_size=64, rank=0)            # (4,64,1,32)
    Vh = promote(V_resh, rank=0)                                    # (4,1,64,1,32)

    # ------------------------------------------------------------------
    # Core attention (blackbox).  The child expects streams whose tile‑row
    # size is 1 and tile‑col size is 32.
    # ------------------------------------------------------------------
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )  # → (4,4,64,1,32) – stream shape (4,4,64)

    # ------------------------------------------------------------------
    # Inverse transform back to (1, seq_len=64, num_heads=16, head_dim=32)
    # ------------------------------------------------------------------
    a0 = promote_outer(attn)                                          # (1,4,4,64,1,32)
    a1 = flatten(a0, min_rank=1, max_rank=2)                          # (1,16,64,1,32)
    a2 = flatten(a1, min_rank=0, max_rank=1)                          # (1,1024,1,32)
    a3 = reshape_stream(a2, chunk_size=16, rank=0)                    # (1,64,16,1,32)
    out = accum_retile_row(a3, rank=1)                                # (1,64,16,32)

    # ------------------------------------------------------------------
    # Contract verification
    # ------------------------------------------------------------------
    assert out.shape == out_shapes[0], (
        f"attention_core: output shape {out.shape} does not match expected "
        f"{out_shapes[0]}"
    )
    return out