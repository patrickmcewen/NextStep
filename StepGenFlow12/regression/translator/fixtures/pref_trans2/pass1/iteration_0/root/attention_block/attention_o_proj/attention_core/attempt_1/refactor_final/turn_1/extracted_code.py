# ----------------------------------------------------------------------
# Implementation reasoning
# ----------------------------------------------------------------------
# Q, K, V arrive as on‑chip streams:
#   Q : (1, 64) × tile (16, 32)   – 64 time steps, 16 heads, 32‑dim
#   K : (1, 64) × tile ( 4, 32)   – 64 time steps, 4 KV‑heads, 32‑dim
#   V : (1, 64) × tile ( 4, 32)   – same layout as K
#
# The child `attention_compute` expects its inputs in the *vanilla* layout
# (kv, heads_per_kv, seq, dim).  In DSL terms this means:
#   stream dims  = (kv, heads_per_kv)   (batch may be a leading 1)
#   tile rows   = seq_len (64)
#   tile cols   = head_dim (32)
#
# To achieve that we:
#   1. Move the original head dimension (the tile rows) into the stream
#      using `retile_streamify(chunk=1)`.  This yields a stream of size
#      1 × 1024 (or 1 × 256 for K/V) and tile rows = 1.
#   2. Split that large stream dimension into the desired (kv, heads_per_kv,
#      seq) factors with two successive `reshape_stream` calls.
#   3. Merge the innermost stream dimension (seq_len) back into the tile
#      rows via `accum_retile_row`, so the tile rows become 64.
#   4. For K and V we still lack the `heads_per_kv` stream dimension; we
#      broadcast them across that axis with `repeat_static(factor=4)`.
#   5. The three tensors now share the identical stream shape
#      (1, 4, 4) and tile shape (64, 32), satisfying the binary ops used
#      inside `attention_compute`.
#   6. Finally we call the child, forwarding the output shape/permutation
#      requested by the parent.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # --------------------------------------------------------------
    # Transform Q -> stream (1, 4, 4) × tile (64, 32)
    # --------------------------------------------------------------
    Qh = retile_streamify(Q, chunk=1, split_row=True)          # (1,1024)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=256, rank=0)            # (1,4,256)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=64, rank=0)             # (1,4,4,64)×tile(1,32)
    Qh = accum_retile_row(Qh, rank=1)                         # (1,4,4)×tile(64,32)

    # --------------------------------------------------------------
    # Transform K -> stream (1, 4) × tile (64, 32), then broadcast
    # --------------------------------------------------------------
    Kh = retile_streamify(K, chunk=1, split_row=True)          # (1,256)×tile(1,32)
    Kh = reshape_stream(Kh, chunk_size=64, rank=0)             # (1,4,64)×tile(1,32)
    Kh = accum_retile_row(Kh, rank=1)                         # (1,4)×tile(64,32)
    Kh = repeat_static(Kh, factor=4)                          # (1,4,4)×tile(64,32)

    # --------------------------------------------------------------
    # Transform V analogous to K
    # --------------------------------------------------------------
    Vh = retile_streamify(V, chunk=1, split_row=True)          # (1,256)×tile(1,32)
    Vh = reshape_stream(Vh, chunk_size=64, rank=0)             # (1,4,64)×tile(1,32)
    Vh = accum_retile_row(Vh, rank=1)                         # (1,4)×tile(64,32)
    Vh = repeat_static(Vh, factor=4)                          # (1,4,4)×tile(64,32)

    # --------------------------------------------------------------
    # Core attention computation (blackbox)
    # --------------------------------------------------------------
    attn = attention_compute(Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms)

    return attn