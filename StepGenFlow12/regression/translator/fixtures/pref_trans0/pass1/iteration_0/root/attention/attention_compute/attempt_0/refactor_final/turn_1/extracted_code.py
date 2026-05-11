# The node receives on‑chip streams Q, K, V with shapes:
#   Q: stream(64,)×tile(16,32)   → (seq_len, num_heads, head_dim)
#   K: stream(64,)×tile(4,32)    → (seq_len, num_kv_heads, head_dim)
#   V: stream(64,)×tile(4,32)    → (seq_len, num_kv_heads, head_dim)
#
# The heavy attention work is delegated to the child blackbox
# `attention_compute__root_attention_attention_compute`, which expects
# the GQA‑reshaped tensors Qh, Kh, Vh (vanilla shape
# (num_kv_heads, query_per_kv, seq_len, head_dim)).  The stub inside the
# blackbox will reinterpret the incoming streams accordingly, so we can
# pass Q, K, V directly.
#
# The blackbox returns a stream representing the vanilla shape
# (num_kv_heads, query_per_kv, seq_len, head_dim).  After flattening
# (performed by the stub) this becomes stream(num_heads,)×tile(seq_len,
# head_dim).  To restore the model‑expected layout (seq_len, num_heads,
# head_dim) we:
#   1. Split the tile‑row dimension (seq_len) into a stream dimension
#      using `retile_streamify(..., chunk=1, split_row=True)`.
#      This yields stream(num_heads * seq_len,)×tile(1, head_dim).
#   2. Split that combined stream back into two stream dimensions
#      (seq_len, num_heads) with `reshape_stream(..., chunk_size=num_heads)`.
#
# The resulting tensor has stream shape (seq_len, num_heads) and tile
# shape (1, head_dim), which aligns with the required vanilla shape
# (64, 16, 32) after the final off‑chip store in the root.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # Extract problem dimensions from the on‑chip streams
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # Call the heavyweight attention blackbox.
    # Its stub will reshape Q/K/V to the GQA layout internally.
    attn = attention_compute__root_attention_attention_compute(
        Q,
        K,
        V,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),  # Qh/Kh/Vh vanilla shape
        ),
        out_perms=(None,),
    )
    # `attn` now has stream shape (num_heads,) and tile rows = seq_len.

    # 1) Turn the tile‑row (seq_len) into a stream dimension.
    attn_split = retile_streamify(attn, chunk=1, split_row=True)

    # 2) Split the merged stream back into (seq_len, num_heads).
    out = reshape_stream(attn_split, chunk_size=num_heads, rank=0)

    return out