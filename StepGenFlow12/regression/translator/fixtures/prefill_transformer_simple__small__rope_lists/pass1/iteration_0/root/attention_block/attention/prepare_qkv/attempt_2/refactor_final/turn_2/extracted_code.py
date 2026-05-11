# QKV preparation using only DSL ops.
#  - Q: (64,16,32) → (4,4,64,32)  (kv, q_per_kv, seq, dim)
#  - K/V: (64,4,32) → (4,1,64,32) (kv, 1, seq, dim)
# The transformation is expressed as a series of stream‑shape manipulations
# (retile → reshape → flatten → reshape → row‑wise merge).

def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # --------------------
    # Q: split heads (16) into (kv=4, q=4) and move seq_len (64) into tile rows.
    # Step 1: turn each head row into its own stream element.
    q0 = retile_streamify(Q, chunk=1)                     # (1024, 1, 32)
    # Step 2: create the q‑dimension (size 4) from the tile‑row chunks.
    q1 = reshape_stream(q0, chunk_size=4, rank=0)          # (256, 4, 1, 32)
    # Step 3: split the combined (kv·seq) dimension into kv and seq.
    q2 = reshape_stream(q1, chunk_size=64, rank=1)         # (4, 64, 4, 1, 32)
    # Step 4: merge seq_len and q into a single stream dimension.
    q3 = flatten(q2, min_rank=0, max_rank=1)               # (4, 256, 1, 32)
    # Step 5: split that merged dimension back into (q, seq_len).
    q4 = reshape_stream(q3, chunk_size=64, rank=0)         # (4, 4, 64, 1, 32)
    # Step 6: promote the seq_len stream dimension into the tile‑row axis.
    Qh = accum_retile_row(q4, rank=1)                      # (4, 4, 64, 32)

    # --------------------
    # K (and V) share the same pattern: move seq_len into tile rows
    # while keeping the kv‑head dimension as a stream.
    # K
    k0 = retile_streamify(K, chunk=1)                     # (256, 1, 32)
    k1 = reshape_stream(k0, chunk_size=64, rank=0)         # (4, 64, 1, 32)
    k2 = reshape_stream(k1, chunk_size=1, rank=1)          # (4, 1, 64, 1, 32)
    Kh = accum_retile_row(k2, rank=1)                      # (4, 1, 64, 32)

    # V (identical to K)
    v0 = retile_streamify(V, chunk=1)                     # (256, 1, 32)
    v1 = reshape_stream(v0, chunk_size=64, rank=0)         # (4, 64, 1, 32)
    v2 = reshape_stream(v1, chunk_size=1, rank=1)          # (4, 1, 64, 1, 32)
    Vh = accum_retile_row(v2, rank=1)                      # (4, 1, 64, 32)

    return Qh, Kh, Vh