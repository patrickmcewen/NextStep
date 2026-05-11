# ----------------------------------------------------------------------
# Implementation reasoning
# ----------------------------------------------------------------------
# * Q, K, V arrive as on‑chip streams:
#     Q : (1, 64) × tile (16, 32)   # seq_len=64, num_heads=16, head_dim=32
#     K : (1, 64) × tile ( 4, 32)   # seq_len=64, num_kv_heads=4, head_dim=32
#     V : (1, 64) × tile ( 4, 32)
#
# * The child `attention_compute` expects its arguments in *vanilla* shape:
#     Qh : (4, 4, 64, 32)   # (kv_heads, heads_per_kv, seq_len, head_dim)
#     Kh : (4, 1, 64, 32)
#     Vh : (4, 1, 64, 32)
#
# * We reshape the on‑chip streams into stream tensors that, when flattened,
#   match those vanilla layouts.  The steps are:
#   – `retile_streamify(..., split_row=True)` moves tile‑row elements into the
#     stream dimension, leaving tile rows = 1.
#   – `reshape_stream` splits the resulting large stream dimension into the
#     required (kv, heads_per_kv, seq) factors.
#   – `accum_retile_row` folds the final stream dimension back into the tile‑row
#     so the tile rows become the original sequence length (64).
#   This yields:
#       Qh : stream(1,4,4) × tile(64,32)   → 32768 elements
#       Kh : stream(1,4)   × tile(64,32)   →  8192 elements
#       Vh : stream(1,4)   × tile(64,32)   →  8192 elements
#
# * The blackbox stub will flatten each argument and reshape it to the
#   vanilla shape listed above, so the extra leading singleton dimension
#   is harmless.
#
# * Finally we invoke the child, passing the arguments by name (keyword
#   arguments) to avoid any ordering ambiguity, and forward the output
#   shape/permutation information supplied by the parent.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # -------------------- Q -> (kv, heads_per_kv, seq) --------------------
    Qh = retile_streamify(Q, chunk=1, split_row=True)                # (1,1024)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=256, rank=0)                  # (1,4,256)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=64, rank=0)                   # (1,4,4,64)×tile(1,32)
    Qh = accum_retile_row(Qh, rank=1)                               # (1,4,4)×tile(64,32)

    # -------------------- K -> (kv, 1, seq) ------------------------------
    Kh = retile_streamify(K, chunk=1, split_row=True)                # (1,256)×tile(1,32)
    Kh = reshape_stream(Kh, chunk_size=64, rank=0)                   # (1,4,64)×tile(1,32)
    Kh = accum_retile_row(Kh, rank=1)                               # (1,4)×tile(64,32)

    # -------------------- V -> (kv, 1, seq) ------------------------------
    Vh = retile_streamify(V, chunk=1, split_row=True)                # (1,256)×tile(1,32)
    Vh = reshape_stream(Vh, chunk_size=64, rank=0)                   # (1,4,64)×tile(1,32)
    Vh = accum_retile_row(Vh, rank=1)                               # (1,4)×tile(64,32)

    # -------------------- Core attention (blackbox) ----------------------
    attn = attention_compute(
        Qh=Qh,
        Kh=Kh,
        Vh=Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn