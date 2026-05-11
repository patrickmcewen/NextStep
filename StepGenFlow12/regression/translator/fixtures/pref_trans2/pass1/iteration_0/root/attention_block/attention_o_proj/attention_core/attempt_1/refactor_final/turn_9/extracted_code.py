# ----------------------------------------------------------------------
# Implementation notes
# ----------------------------------------------------------------------
# Q, K, V are on‑chip streams:
#   Q : (1, 64) × tile (16, 32)   # seq_len=64, 16 heads, head_dim=32
#   K : (1, 64) × tile ( 4, 32)   # seq_len=64, 4 KV‑heads, head_dim=32
#   V : (1, 64) × tile ( 4, 32)
#
# The child `attention_compute` expects its inputs in vanilla shape:
#   Qh : (4, 4, 64, 32)   # (kv_heads, heads_per_kv, seq_len, head_dim)
#   Kh : (4, 1, 64, 32)
#   Vh : (4, 1, 64, 32)
#
# We obtain those tensors with `offchip_load`, providing a stride that
# reproduces the view‑permute logic of the reference code, and then
# fold the streaming `seq` dimension back into the tile rows with
# `accum_retile_row`.  Finally we forward the tensors to the blackbox,
# passing the parent‑supplied `out_shapes`/`out_perms`.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Q → (kv=4, heads_per_kv=4, seq=64)  × tile (1, 32)
    Qh = offchip_load(
        Q,
        stride=(4, 1, 16),                 # kv stride=4, heads_per_kv stride=1, seq stride=16
        out_shape_tiled=(4, 4, 64),
        tile_row=1,
        tile_col=32,
    )
    Qh = accum_retile_row(Qh, rank=1)      # (1,4,4) × tile (64,32)

    # K → (kv=4, seq=64) × tile (1, 32)
    Kh = offchip_load(
        K,
        stride=(1, 4),                     # kv stride=1, seq stride=4
        out_shape_tiled=(4, 64),
        tile_row=1,
        tile_col=32,
    )
    Kh = accum_retile_row(Kh, rank=1)      # (1,4) × tile (64,32)

    # V → (kv=4, seq=64) × tile (1, 32)
    Vh = offchip_load(
        V,
        stride=(1, 4),                     # kv stride=1, seq stride=4
        out_shape_tiled=(4, 64),
        tile_row=1,
        tile_col=32,
    )
    Vh = accum_retile_row(Vh, rank=1)      # (1,4) × tile (64,32)

    # Core attention (blackbox)
    attn = attention_compute(
        Qh, Kh, Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn