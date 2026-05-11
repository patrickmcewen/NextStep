# Implementation reasoning:
# Q, K, V are on‑chip tile streams with shapes:
#   Q: (seq_len=64, heads=16, head_dim=32) → stream(64,)×tile(16,32)
#   K, V: (seq_len=64, kv_heads=4, head_dim=32) → stream(64,)×tile(4,32)
#
# Desired outputs:
#   Qh → (kv_heads=4, query_per_kvhead=4, seq_len=64, head_dim=32) → stream(4,4)×tile(64,32)
#   Kh, Vh → (kv_heads=4, 1, seq_len=64, head_dim=32) → stream(4,1)×tile(64,32)
#
# The transformation swaps the original stream dimension (seq_len) with the tile‑row
# dimension (heads) and then splits the merged dimension into the required
# per‑KV‑head streams.  This can be expressed with the following DSL ops:
#   1. `accum_retile_row` merges the stream dimension into the tile‑row dimension.
#   2. `promote` inserts a dummy leading stream dim of size 1 so that we can
#      use `retile_streamify`.
#   3. `retile_streamify` splits the enlarged tile‑row dimension back into a
#      new stream dimension while restoring the original seq_len as the tile‑row.
#   4. `reshape_stream` finally splits the remaining stream dimension into
#      (kv_heads, query_per_kvhead) for Q, or (kv_heads, 1) for K/V.
#
# This sequence uses only DSL primitives; no raw tensor methods or arithmetic
# appear between the inputs and the outputs.

def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Basic scalar parameters derived from input shapes
    seq_len = Q.shape[0]                     # 64
    kv_heads = K.shape[-2]                   # 4
    query_per_kvhead = Q.shape[-2] // kv_heads  # 16 // 4 = 4

    # -------- Q transformation -------------------------------------------------
    # Merge stream (seq_len) into tile rows (heads) → tile rows = seq_len * heads
    Q_merged = accum_retile_row(Q, rank=1)
    # Insert a dummy leading stream dimension so we can retile the huge tile rows
    Q_promoted = promote(Q_merged, rank=0)
    # Split the merged tile rows back into a stream dimension, restoring seq_len as tile rows
    Q_retile = retile_streamify(Q_promoted, chunk=seq_len, split_row=True)
    # Split the remaining stream (kv_heads * query_per_kvhead) into the two desired streams
    Qh = reshape_stream(Q_retile, chunk_size=query_per_kvhead, rank=0)

    # -------- K transformation -------------------------------------------------
    K_merged = accum_retile_row(K, rank=1)
    K_promoted = promote(K_merged, rank=0)
    K_retile = retile_streamify(K_promoted, chunk=seq_len, split_row=True)
    Kh = reshape_stream(K_retile, chunk_size=1, rank=0)  # creates the extra singleton stream dim

    # -------- V transformation -------------------------------------------------
    V_merged = accum_retile_row(V, rank=1)
    V_promoted = promote(V_merged, rank=0)
    V_retile = retile_streamify(V_promoted, chunk=seq_len, split_row=True)
    Vh = reshape_stream(V_retile, chunk_size=1, rank=0)

    # Sanity‑check against the contracts supplied by the parent node
    assert Qh.shape == out_shapes[0], f"Qh shape mismatch: {Qh.shape} != {out_shapes[0]}"
    assert Kh.shape == out_shapes[1], f"Kh shape mismatch: {Kh.shape} != {out_shapes[1]}"
    assert Vh.shape == out_shapes[2], f"Vh shape mismatch: {Vh.shape} != {out_shapes[2]}"

    return Qh, Kh, Vh