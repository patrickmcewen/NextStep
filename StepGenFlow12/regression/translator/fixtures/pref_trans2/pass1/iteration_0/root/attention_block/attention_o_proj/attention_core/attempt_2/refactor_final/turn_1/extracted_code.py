# Implementation notes:
# - Q, K, V are already on‑chip streams with shapes:
#     Q: stream(1, 64)×tile(16, 32)
#     K, V: stream(1, 64)×tile(4, 32)
# - To feed the child `attention_compute` we need the GQA layout:
#     Qh: (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32)
#     Kh, Vh: (kv_head=4, 1, seq_len=64, head_dim=32)
#   This is performed by the `restream` helper with the appropriate strides.
# - After the core attention produces a tensor of shape (4, 4)×tile(64, 32),
#   we must invert the earlier permutation.  We do this without any
#   tensor‑method calls:
#     • `flatten` merges the two leading stream dimensions (kv_head, q_per_kv)
#       into a single dimension of size 16.
#     • `promote_outer` inserts a leading batch dimension (size 1).
#     • `retile_streamify` splits the tile‑row dimension (64) into a new
#       stream dimension (seq_len=64) and a tile‑row dimension of size 16
#       (the original number of heads).  The result is exactly the shape
#       required by the parent: stream(1, 64)×tile(16, 32).
#
# The child `attention_compute` is called with explicit `out_shapes` and
# `out_perms` as required by the DSL.

def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Transform Q into Qh: (kv_head, q_per_kv, seq_len, head_dim)
    Qh = restream(
        Q,
        stride=(8192, 2048, 32, 1),          # (kv_head, q_per_kv, seq_len, head_dim) strides
        out_shape_tiled=(4, 4, 64, 32),
    )

    # Transform K into Kh: (kv_head, 1, seq_len, head_dim)
    Kh = restream(
        K,
        stride=(2048, 0, 32, 1),             # kv_head stride, no broadcast on q_per_kv
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Transform V into Vh: (kv_head, 1, seq_len, head_dim)
    Vh = restream(
        V,
        stride=(2048, 0, 32, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Core attention computation (blackbox)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),
        out_perms=(None,),
    )

    # Inverse transform: (4,4)×tile(64,32) → (1,64)×tile(16,32)
    # 1. Merge the two head‑related stream dimensions.
    attn_flat = flatten(attn, min_rank=0, max_rank=1)   # stream(16)×tile(64,32)

    # 2. Add the outer batch dimension required by the contract.
    attn_outer = promote_outer(attn_flat)                # stream(1,16)×tile(64,32)

    # 3. Split the tile‑row dimension (64) into (seq_len=64) stream and
    #    tile rows = 16 (number of heads).
    out = retile_streamify(
        attn_outer,
        chunk=16,            # head count per token
        split_row=True,
    )                         # stream(1,64)×tile(16,32)

    return out