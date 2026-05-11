# Convert the (seq_len, num_heads, head_dim) streams into the GQA layout
# expected by the `attention_compute` blackbox, run the attention, then
# reverse the layout back to (seq_len, num_heads, head_dim).  All shape
# manipulations use DSL operators; no raw Tensor methods appear.
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Q -> Qh : (1,64,16,32)  ->  (1,4,4,64,1,32)
    #   1) split tile‑rows (16) fully into a stream dimension
    #   2) reshape that stream (1024) into (16,64)
    #   3) reshape the 16 into (4,4)
    # ------------------------------------------------------------------
    Qr = retile_streamify(Q, chunk=1, split_row=True)            # (1,1024,1,32)
    Qs = reshape_stream(Qr, chunk_size=64, rank=0)               # (1,16,64,1,32)
    Qh = reshape_stream(Qs, chunk_size=4, rank=1)                # (1,4,4,64,1,32)

    # ------------------------------------------------------------------
    # K -> Kh : (1,64,4,32)  ->  (4,64,1,32)  (vanilla (4,1,64,32))
    #   1) split tile‑rows (4) fully into a stream dimension
    #   2) reshape that stream (256) into (4,64)
    #   3) merge the leading singleton with the first 4 to obtain
    #      a stream of size 4 (the GQA KV‑head dimension)
    # ------------------------------------------------------------------
    Kr = retile_streamify(K, chunk=1, split_row=True)            # (1,256,1,32)
    Ks = reshape_stream(Kr, chunk_size=64, rank=0)               # (1,4,64,1,32)
    Kh = flatten(Ks, min_rank=1, max_rank=2)                     # (4,64,1,32)

    # ------------------------------------------------------------------
    # V -> Vh : same logic as K
    # ------------------------------------------------------------------
    Vr = retile_streamify(V, chunk=1, split_row=True)            # (1,256,1,32)
    Vs = reshape_stream(Vr, chunk_size=64, rank=0)               # (1,4,64,1,32)
    Vh = flatten(Vs, min_rank=1, max_rank=2)                     # (4,64,1,32)

    # ------------------------------------------------------------------
    # Core attention (blackbox).  The output vanilla shape is (4,4,64,32),
    # so we request a matching stream layout.
    # ------------------------------------------------------------------
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((1, 4, 4, 64, 1, 32),),   # streamed equivalent of (4,4,64,32)
        out_perms=None,
    )

    # ------------------------------------------------------------------
    # Inverse the layout transformation:
    #   attn : (1,4,4,64,1,32)
    #   1) merge the two leading 4‑dims into 16
    #   2) merge 16 with 64 → 1024
    #   3) split 1024 into (64,16) where 16 becomes the tile‑row size
    #   4) retile the 16 back into the tile‑row dimension
    # ------------------------------------------------------------------
    a1 = flatten(attn, min_rank=1, max_rank=2)       # (1,16,64,1,32)
    a2 = flatten(a1, min_rank=0, max_rank=1)         # (1,1024,1,32)
    a3 = reshape_stream(a2, chunk_size=16, rank=0)   # (1,64,16,1,32)
    out = accum_retile_row(a3, rank=1)               # (1,64,16,32)

    # Verify that the produced shape matches the contract
    assert out.shape == out_shapes[0], (
        f"attention_core: output shape {out.shape} does not match expected {out_shapes[0]}"
    )
    return out