# The parent supplies on‑chip streams Q, K, V with shapes
#   Q: (1, 64, 16, 32)   K/V: (1, 64,  4, 32)
# We must reshape them into the GQA layout expected by
# `attention_compute`:
#   Qh → (4, 4, 64, 1, 32)   (i.e. vanilla (4,4,64,32) with a singleton tile‑row)
#   Kh → (4, 1, 64, 1, 32)
#   Vh → (4, 1, 64, 1, 32)
# All manipulations use DSL primitives; no raw tensor methods appear.
# After the attention blackbox we invert the reshapes to obtain the
# contract‑required output stream shape (1, 64, 16, 32).
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh -------------------------------------------------------
    # Split the tile‑row dimension (16) into stream dims (4,4)
    Q1 = retile_streamify(Q, chunk=1, split_row=True)                # (1,1024,1,32)
    Q2 = reshape_stream(Q1, chunk_size=64, rank=0)                   # (1,16,64,1,32)
    Q3 = reshape_stream(Q2, chunk_size=4, rank=1)                    # (1,4,4,64,1,32)
    Qh = flatten(Q3, min_rank=2, max_rank=3)                        # (4,4,64,1,32)

    # ----- K → Kh -------------------------------------------------------
    # Move the tile‑row (4) into the stream, then introduce the required
    # singleton KV‑head dimension.
    K1 = retile_streamify(K, chunk=1, split_row=True)                # (1,256,1,32)
    K2 = reshape_stream(K1, chunk_size=64, rank=0)                   # (1,4,64,1,32)
    K3 = reshape_stream(K2, chunk_size=1, rank=1)                    # (1,4,1,64,1,32)
    Kh = flatten(K3, min_rank=2, max_rank=3)                        # (4,1,64,1,32)

    # ----- V → Vh -------------------------------------------------------
    V1 = retile_streamify(V, chunk=1, split_row=True)                # (1,256,1,32)
    V2 = reshape_stream(V1, chunk_size=64, rank=0)                   # (1,4,64,1,32)
    V3 = reshape_stream(V2, chunk_size=1, rank=1)                    # (1,4,1,64,1,32)
    Vh = flatten(V3, min_rank=2, max_rank=3)                        # (4,1,64,1,32)

    # ----- Core attention (blackbox) ------------------------------------
    # The child expects a stream whose tile‑row is 1 and tile‑col is 32.
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 1, 32),),   # stream shape for vanilla (4,4,64,32)
        out_perms=out_perms,
    )  # → (4, 4, 64, 1, 32)

    # ----- Inverse transform back to (1, 64, 16, 32) --------------------
    # Merge the two head dimensions (4,4) → 16
    a1 = flatten(attn, min_rank=1, max_rank=2)          # (16, 64, 1, 32)
    a2 = promote_outer(a1)                              # (1, 16, 64, 1, 32)
    a3 = flatten(a2, min_rank=0, max_rank=1)           # (1, 1024, 1, 32)
    a4 = reshape_stream(a3, chunk_size=16, rank=0)      # (1, 64, 16, 1, 32)
    out = accum_retile_row(a4, rank=1)                  # (1, 64, 16, 32)

    # Verify that we honour the contract
    assert out.shape == out_shapes[0], (
        f"attention_core: output shape {out.shape} does not match expected {out_shapes[0]}"
    )
    return out