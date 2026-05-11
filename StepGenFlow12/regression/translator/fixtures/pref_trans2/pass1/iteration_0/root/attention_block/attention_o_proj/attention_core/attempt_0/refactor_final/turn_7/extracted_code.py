# Build the GQA‑layout tensors expected by `attention_compute`,
# invoke the child, and return its result directly.
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Q → Qh  (num_kv_heads=4, query_per_kvhead=4, seq_len=64)
    #   Q: (1,64,16,32)  →  stream (1,64), tile (16,32)
    # ------------------------------------------------------------------
    Qr = retile_streamify(Q, chunk=1, split_row=True)               # (1,1024,1,32)
    Qs = reshape_stream(Qr, chunk_size=64, rank=0)                  # (1,16,64,1,32)
    Qt = reshape_stream(Qs, chunk_size=4, rank=1)                   # (1,4,4,64,1,32)
    Qh = flatten(Qt, min_rank=2, max_rank=3)                       # (4,4,64,1,32)

    # ------------------------------------------------------------------
    # K → Kh  (num_kv_heads=4, seq_len=64)
    # ------------------------------------------------------------------
    Kr = retile_streamify(K, chunk=1, split_row=True)               # (1,256,1,32)
    Ks = reshape_stream(Kr, chunk_size=64, rank=0)                  # (1,4,64,1,32)
    K_flat = flatten(Ks, min_rank=1, max_rank=2)                    # (4,64,1,32)
    Kh = reshape_stream(K_flat, chunk_size=1, rank=1)               # (4,1,64,1,32)

    # ------------------------------------------------------------------
    # V → Vh  (num_kv_heads=4, seq_len=64)
    # ------------------------------------------------------------------
    Vr = retile_streamify(V, chunk=1, split_row=True)               # (1,256,1,32)
    Vs = reshape_stream(Vr, chunk_size=64, rank=0)                  # (1,4,64,1,32)
    V_flat = flatten(Vs, min_rank=1, max_rank=2)                    # (4,64,1,32)
    Vh = reshape_stream(V_flat, chunk_size=1, rank=1)               # (4,1,64,1,32)

    # ------------------------------------------------------------------
    # Core attention (blackbox).  The child will reshape its vanilla
    # output (4,4,64,32) to the stream shape requested via `out_shapes`.
    # ------------------------------------------------------------------
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )   # expected shape: (1,64,16,32)

    # The child already yields the contract‑required layout, so simply return it.
    assert attn.shape == out_shapes[0], (
        f"attention_core: output shape {attn.shape} does not match expected "
        f"{out_shapes[0]}"
    )
    return attn