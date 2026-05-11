# The reference implementation first permutes Q to (heads, seq_len, dim) and then
# reshapes heads (16) into two stream dimensions (4, 4), leaving seq_len as the
# tile‑row dimension.  It also permutes K/V to (kv_heads, 1, seq_len, dim).
# In the DSL we express these permutations without any raw tensor ops:
#   * `retile_streamify(..., chunk=1)` moves a tile‑row dimension into the stream.
#   * `reshape_stream` splits a stream dimension into two.
#   * `accum_retile_row` merges a singleton tile‑row back into the preceding
#     stream dimension, turning that stream dimension into the final tile‑row.
#   * A final `reshape_stream` with `chunk_size=1` creates the required singleton
#     stream axis for K and V.
#
# This sequence reproduces the exact output shapes required by the contract:
#   Qh → (4, 4, 64, 32)   Kh/Vh → (4, 1, 64, 32)
#
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ---- Q ----
    # Move the original heads (tile‑row) into the stream, flattening it.
    q0 = retile_streamify(Q, chunk=1)                     # (1024, 1, 32)
    # Split the combined stream (1024) into (16, 64):
    q1 = reshape_stream(q0, chunk_size=64, rank=0)        # (16, 64, 1, 32)
    # Split the 16 into (4, 4):
    q2 = reshape_stream(q1, chunk_size=4, rank=1)         # (4, 4, 64, 1, 32)
    # Merge the singleton tile‑row (1) into the 64‑stream dimension,
    # turning it into the final tile‑row.
    Qh = accum_retile_row(q2, rank=1)                     # (4, 4, 64, 32)

    # ---- K ----
    k0 = retile_streamify(K, chunk=1)                     # (256, 1, 32)
    k1 = reshape_stream(k0, chunk_size=64, rank=0)        # (4, 64, 1, 32)
    k2 = accum_retile_row(k1, rank=1)                     # (4, 64, 32)
    # Insert a singleton stream dimension after kv_heads.
    Kh = reshape_stream(k2, chunk_size=1, rank=0)          # (4, 1, 64, 32)

    # ---- V ---- (identical to K)
    v0 = retile_streamify(V, chunk=1)                     # (256, 1, 32)
    v1 = reshape_stream(v0, chunk_size=64, rank=0)        # (4, 64, 1, 32)
    v2 = accum_retile_row(v1, rank=1)                     # (4, 64, 32)
    Vh = reshape_stream(v2, chunk_size=1, rank=0)          # (4, 1, 64, 32)

    return Qh, Kh, Vh