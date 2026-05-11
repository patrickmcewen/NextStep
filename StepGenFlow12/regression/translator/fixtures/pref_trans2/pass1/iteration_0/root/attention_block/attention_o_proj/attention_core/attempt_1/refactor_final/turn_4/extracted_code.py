# ----------------------------------------------------------------------
# Implementation reasoning
# ----------------------------------------------------------------------
# The parent supplies on‑chip streams:
#   Q : (1, 64) × tile (16, 32)   # seq_len=64, num_heads=16, head_dim=32
#   K : (1, 64) × tile ( 4, 32)   # seq_len=64, num_kv_heads=4, head_dim=32
#   V : (1, 64) × tile ( 4, 32)
#
# The child `attention_compute` expects its inputs in *vanilla* shape:
#   Qh : (4, 4, 64, 32)   # (kv_heads, heads_per_kv, seq_len, head_dim)
#   Kh : (4, 1, 64, 32)
#   Vh : (4, 1, 64, 32)
#
# We convert the streams to those layouts using only DSL primitives:
#   1. `retile_streamify(..., split_row=True)` moves the tile‑row dimension
#      into the stream, leaving tile rows = 1.
#   2. `reshape_stream` splits the resulting large stream dimension into the
#      required (kv, heads_per_kv, seq) factors.
#   3. `accum_retile_row` folds the final stream dimension back into the
#      tile‑row, giving tile rows = seq_len (64).
#
# After these steps we have:
#   Qh : stream(1,4,4) × tile(64,32)   → flattened => (4,4,64,32)
#   Kh : stream(1,4)   × tile(64,32)   → flattened => (4,1,64,32)
#   Vh : stream(1,4)   × tile(64,32)   → flattened => (4,1,64,32)
#
# Finally we forward the tensors to `attention_compute`, propagating the
# output shape/permutation requested by the parent.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Q -> (kv, heads_per_kv, seq) with tile rows = 64
    Qh = retile_streamify(Q, chunk=1, split_row=True)                 # (1,1024)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=256, rank=0)                   # (1,4,256)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=64, rank=0)                    # (1,4,4,64)×tile(1,32)
    Qh = accum_retile_row(Qh, rank=1)                                # (1,4,4)×tile(64,32)

    # K -> (kv, 1, seq) with tile rows = 64
    Kh = retile_streamify(K, chunk=1, split_row=True)                 # (1,256)×tile(1,32)
    Kh = reshape_stream(Kh, chunk_size=64, rank=0)                    # (1,4,64)×tile(1,32)
    Kh = accum_retile_row(Kh, rank=1)                                # (1,4)×tile(64,32)

    # V -> (kv, 1, seq) with tile rows = 64
    Vh = retile_streamify(V, chunk=1, split_row=True)                 # (1,256)×tile(1,32)
    Vh = reshape_stream(Vh, chunk_size=64, rank=0)                    # (1,4,64)×tile(1,32)
    Vh = accum_retile_row(Vh, rank=1)                                # (1,4)×tile(64,32)

    # Core attention (blackbox)
    attn = attention_compute(
        Qh, Kh, Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn