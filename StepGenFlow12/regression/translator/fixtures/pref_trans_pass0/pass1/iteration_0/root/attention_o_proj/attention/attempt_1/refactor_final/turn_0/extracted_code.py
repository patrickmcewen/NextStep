# The `attention` node receives on‑chip streams `Q`, `K`, `V`.  
# It first expands those into the three head‑specific streams expected by the model  
# (`Qh`, `Kh`, `Vh`) via the `compute_qkv` blackbox.  The vanilla shapes of those
# outputs are:
#   * Qh : (4, 4, 64, 32)   – 4 “group” heads, 4 heads per group, seq_len=64, head_dim=32  
#   * Kh : (4, 1, 64, 32)   – 4 groups, 1 KV‑head per group, seq_len=64, head_dim=32  
#   * Vh : (4, 1, 64, 32)   – same layout as `Kh`  
# For each output we pass a stream shape that mirrors its vanilla shape; the last two
# dimensions are the tile dimensions, so the shapes are valid tile‑streams (rank ≥ 3).  
# The `out_perms` entries are `None` because no permutation is needed.
#  
# The three streams are then fed to `attention_compute`, which produces the final
# attention output with the shape requested by the caller (`out_shapes`, typically
# `(64, 16, 32)`).  We simply forward the caller‑provided `out_shapes` and `out_perms`
# to the child.  The resulting stream is returned directly – the root node will
# handle the off‑chip store.  

def attention(Q, K, V, *, out_shapes, out_perms=None):
    # Compute Q, K, V projections per head.
    Qh, Kh, Vh = compute_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),  # Qh stream shape
            (4, 1, 64, 32),  # Kh stream shape
            (4, 1, 64, 32),  # Vh stream shape
        ),
        out_perms=(None, None, None),
    )
    # Perform attention on the projected heads.
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn