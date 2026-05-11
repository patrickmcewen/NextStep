# The core attention receives Q, K, V already as on‑chip streams:
#   Q : (1, 64, 16, 32)   – (batch, seq_len, num_heads, head_dim)
#   K/V : (1, 64, 4, 32) – (batch, seq_len, num_kv_heads, head_dim)
# For group‑query attention we must reshape them to the per‑head layout
# expected by `attention_compute`:
#   Qh : (4, 4, 64, 32)   – (kv_head, q_per_kv, seq_len, head_dim)
#   Kh/Vh : (4, 1, 64, 32) – (kv_head, 1, seq_len, head_dim)
# This permutation is performed with `restream`, which can arbitrarily
# reorder stream vs. tile dimensions via `promote`, `retile_streamify`,
# `bufferize`, `streamify`, and the subsequent `accum_retile_*` merges.
# The required `stride` values encode a row‑major flattening that maps the
# target dimensions onto the underlying buffer grid.
# After the child computation we invert the transformation to obtain the
# original layout (1, 64, 16, 32).

def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Q → (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32)
    Qh = restream(
        Q,
        stride=(8192, 2048, 32, 1),          # i0*8192 + i1*2048 + i2*32 + i3*1
        out_shape_tiled=(4, 4, 64, 32),
    )

    # K → (kv_head=4, 1, seq_len=64, head_dim=32)
    Kh = restream(
        K,
        stride=(2048, 0, 32, 1),             # i0*2048 + i1*0 + i2*32 + i3*1
        out_shape_tiled=(4, 1, 64, 32),
    )

    # V → (kv_head=4, 1, seq_len=64, head_dim=32)
    Vh = restream(
        V,
        stride=(2048, 0, 32, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Core attention (blackbox)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),
        out_perms=(None,),
    )

    # Restore original layout: (batch=1, seq_len=64, num_heads=16, head_dim=32)
    out = restream(
        attn,
        stride=(0, 32, 2048, 1),            # batch broadcast, seq_len*head_dim, heads, head_dim
        out_shape_tiled=(1, 64, 16, 32),
    )

    return out