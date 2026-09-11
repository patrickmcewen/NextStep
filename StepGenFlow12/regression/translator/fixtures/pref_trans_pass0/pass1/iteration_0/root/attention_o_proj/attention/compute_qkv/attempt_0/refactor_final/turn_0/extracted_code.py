# Implementation reasoning:
# Q, K, V are already on‑chip streams. Their shapes are:
#   Q: (seq_len=64, num_heads=16, head_dim=32)   → stream dim = num_heads = 16
#   K: (seq_len=64, num_kv_heads=4, head_dim=32) → stream dim = num_kv_heads = 4
#   V: same as K.
# The required outputs are:
#   Qh: split Q's stream dimension (16) into (num_kv_heads, query_per_kvhead)
#       where query_per_kvhead = 16 // 4 = 4, yielding shape (4,4,64,32).
#   Kh, Vh: split K/V's stream dimension (4) into (4,1), yielding shape (4,1,64,32).
# This is achieved by using the DSL `reshape_stream`, which factors a stream
# dimension into two stream dimensions (new_count, chunk).  No off‑chip loads
# are needed because the inputs are already on‑chip.  The computed shapes are
# optionally verified against the caller‑provided `out_shapes`.
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Determine the number of KV heads from K (same for V).
    kv_heads = K.shape[-2]                     # integer, e.g. 4
    # Each query head corresponds to `query_per_kvhead` heads in Q.
    query_per_kvhead = Q.shape[-2] // kv_heads  # 16 // 4 = 4

    # Split Q's stream dimension (num_heads) into (kv_heads, query_per_kvhead).
    Qh = reshape_stream(Q, chunk_size=query_per_kvhead, rank=0)

    # Split K and V's stream dimension (kv_heads) into (kv_heads, 1).
    Kh = reshape_stream(K, chunk_size=1, rank=0)
    Vh = reshape_stream(V, chunk_size=1, rank=0)

    # Optional sanity checks against the expected shapes from the parent.
    assert Qh.shape == out_shapes[0], f"Qh shape mismatch: {Qh.shape} != {out_shapes[0]}"
    assert Kh.shape == out_shapes[1], f"Kh shape mismatch: {Kh.shape} != {out_shapes[1]}"
    assert Vh.shape == out_shapes[2], f"Vh shape mismatch: {Vh.shape} != {out_shapes[2]}"

    return Qh, Kh, Vh