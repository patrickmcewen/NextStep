# ----------------------------------------------------------------------
# Implementation notes
# ----------------------------------------------------------------------
# Q, K, V arrive as on‑chip streams:
#   Q : (1, 64) × tile (16, 32)   # seq_len=64, 16 heads, head_dim=32
#   K : (1, 64) × tile ( 4, 32)   # seq_len=64, 4 KV‑heads, head_dim=32
#   V : (1, 64) × tile ( 4, 32)
#
# To match the vanilla layout expected by the child `attention_compute`
#   Qh : (4, 4, 64, 32)   # (kv, heads_per_kv, seq, dim)
#   Kh : (4, 1, 64, 32)
#   Vh : (4, 1, 64, 32)
#
# The on‑chip tensors still carry a leading singleton stream dimension.
# We first collapse that singleton together with the sequence dimension
# using `flatten`, obtaining streams of shape (64) × tile(...).
# Then we employ `offchip_load` with an appropriate stride and
# `out_shape_tiled` so that the flat tile index reproduces the required
# (kv, heads_per_kv, seq) ordering.  Finally `accum_retile_row` folds the
# sequence dimension back into the tile rows, yielding the exact tiled
# shape needed by the blackbox.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ---------- Q -> (kv, heads_per_kv, seq) -----------------------
    # Merge the leading singleton stream dim with the seq stream dim.
    Q_flat = flatten(Q, min_rank=0, max_rank=1)                 # stream(64) × tile(16,32)

    # offchip_load will reorder the 16 head rows into (kv, heads_per_kv, seq)
    Qh = offchip_load(
        Q_flat,
        stride=(4, 1, 16),          # kv stride=4, heads_per_kv stride=1, seq stride=16
        out_shape_tiled=(4, 4, 64),
        tile_row=1,
        tile_col=32,
    )
    Qh = accum_retile_row(Qh, rank=1)                         # stream(1,4,4) × tile(64,32)

    # ---------- K -> (kv, 1, seq) ---------------------------------
    K_flat = flatten(K, min_rank=0, max_rank=1)                 # stream(64) × tile(4,32)

    Kh = offchip_load(
        K_flat,
        stride=(1, 4),               # kv stride=1, seq stride=4
        out_shape_tiled=(4, 64),
        tile_row=1,
        tile_col=32,
    )
    Kh = accum_retile_row(Kh, rank=1)                          # stream(1,4) × tile(64,32)

    # ---------- V -> (kv, 1, seq) ---------------------------------
    V_flat = flatten(V, min_rank=0, max_rank=1)                 # stream(64) × tile(4,32)

    Vh = offchip_load(
        V_flat,
        stride=(1, 4),               # kv stride=1, seq stride=4
        out_shape_tiled=(4, 64),
        tile_row=1,
        tile_col=32,
    )
    Vh = accum_retile_row(Vh, rank=1)                          # stream(1,4) × tile(64,32)

    # ---------------- Core attention (blackbox) --------------------
    attn = attention_compute(
        Qh, Kh, Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn