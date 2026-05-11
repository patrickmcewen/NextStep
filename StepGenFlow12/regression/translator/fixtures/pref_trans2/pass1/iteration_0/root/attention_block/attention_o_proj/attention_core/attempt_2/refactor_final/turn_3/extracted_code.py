# The inputs Q, K, V are already on‑chip streams:
#   Q : (1, 64)×tile(16, 32)   – (batch, seq_len) stream, heads as tile rows
#   K, V : (1, 64)×tile(4, 32) – (batch, seq_len) stream, kv‑heads as tile rows
#
# We first reshape them to the layout expected by the child
# `attention_compute`:
#   Qh → (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32)  → stream(4,4)×tile(64,32)
#   Kh, Vh → (kv_head=4, 1, seq_len=64, head_dim=32)      → stream(4,1)×tile(64,32)
#
# This is done with `restream`, whose strides encode the permutation
# (kv_head, q_per_kv, seq_len, head_dim) → (batch, seq_len, head, dim).
#
# The child returns a tensor of vanilla shape (4,4,64,32).  The reference
# implementation then does:
#     attn.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
# To reproduce this with the DSL we ask the child to apply the same
# permutation before reshaping to the required output stream shape
# (1, 64, 16, 32).  The parent passes `out_perms=None`, but we can supply
# our own permutation to the child – the contract only requires that the
# *returned* tensor matches the `out_shapes` provided to us.
#
# Finally we return the child’s result directly (no extra DSL ops needed).

def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Q → (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32)
    Qh = restream(
        Q,
        stride=(128, 32, 512, 1),          # kv_head stride, q_per_kv stride, seq_len stride, head_dim stride
        out_shape_tiled=(4, 4, 64, 32),
    )

    # K → (kv_head=4, 1, seq_len=64, head_dim=32)
    Kh = restream(
        K,
        stride=(32, 0, 128, 1),             # kv_head stride, broadcast dim, seq_len stride, head_dim stride
        out_shape_tiled=(4, 1, 64, 32),
    )

    # V → (kv_head=4, 1, seq_len=64, head_dim=32)
    Vh = restream(
        V,
        stride=(32, 0, 128, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Ask the child to produce the final layout directly.
    # The reference implementation permutes axes (2, 0, 1, 3) before reshaping;
    # we pass that permutation to the child so that its output matches
    # the required stream shape (1, 64, 16, 32).
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=((2, 0, 1, 3),),  # corresponds to .permute(2,0,1,3) in the reference
    )

    return attn