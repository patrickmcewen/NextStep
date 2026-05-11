# The reference implementation reshapes Q from (S, N_q, D) to (Hkv, qpkv, S, D) and
# permutes K/V from (S, N_kv, D) to (Hkv, 1, S, D).
# In the DSL we cannot use torch.view/permute directly, but the same effect can be
# achieved by:
#   1. Merging the stream dimension (S) with the tile‑row dimension (N_q or N_kv)
#      using `accum_retile_row`, producing a single large tile row.
#   2. Adding a dummy leading stream dimension with `promote` so that the tensor
#      once again has a stream rank ≥ 1.
#   3. Splitting the large tile row into the original sequence length (S) with
#      `retile_streamify` (chunk = seq_len).  This yields a stream of size
#      Hkv × qpkv (or Hkv for K/V) and a tile row equal to the sequence length.
#   4. Finally, turning the remaining stream size into the required extra stream
#      dimensions with `reshape_stream`.  For Q we split size 16 into (4, 4);
#      for K/V we split size 4 into (4, 1) by using a chunk size of 1.
# The `out_shapes` tuple supplies the concrete numbers (Hkv, qpkv, seq_len, D)
# so we can drive the transformations without hard‑coding dimensions.
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Q -> (Hkv, qpkv, S, D)
    Qh = accum_retile_row(Q, rank=1)                       # (S*N_q, D)
    Qh = promote(Qh, rank=0)                               # (1, S*N_q, D)
    Qh = retile_streamify(Qh, chunk=out_shapes[0][2], split_row=True)  # (Hkv*qpkv, S, D)
    Qh = reshape_stream(Qh, chunk_size=out_shapes[0][0], rank=0)       # (Hkv, qpkv, S, D)

    # K -> (Hkv, 1, S, D)
    Kh = accum_retile_row(K, rank=1)                       # (S*N_kv, D)
    Kh = promote(Kh, rank=0)                               # (1, S*N_kv, D)
    Kh = retile_streamify(Kh, chunk=out_shapes[1][2], split_row=True)  # (Hkv, S, D)
    Kh = reshape_stream(Kh, chunk_size=out_shapes[1][1], rank=0)       # (Hkv, 1, S, D)

    # V -> (Hkv, 1, S, D)  (same pattern as K)
    Vh = accum_retile_row(V, rank=1)                       # (S*N_kv, D)
    Vh = promote(Vh, rank=0)                               # (1, S*N_kv, D)
    Vh = retile_streamify(Vh, chunk=out_shapes[2][2], split_row=True)  # (Hkv, S, D)
    Vh = reshape_stream(Vh, chunk_size=out_shapes[2][1], rank=0)       # (Hkv, 1, S, D)

    return Qh, Kh, Vh