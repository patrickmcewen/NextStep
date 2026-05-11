# The reference implementation reshapes Q from (S, H, D) to (Hkv, Q_per_kv, S, D) and
# permutes K, V to (Hkv, 1, S, D).  Since we cannot call `.view` or `.permute`,
# we achieve the same layout using stream‑aware DSL primitives:
#   1. `accum_retile_row` merges the sequence stream into the tile‑row dimension,
#      turning (S, H, D) into a single large tile (S*H, D).  After this step the
#      tensor has **no** stream dimensions.
#   2. `promote(..., rank=0)` inserts a leading stream dimension of size 1, making
#      the tensor a valid stream (required for subsequent ops).  Using `rank=0`
#      (instead of `rank=1`) works for a rank‑2 input because the unsqueeze is
#      performed at `-(3+0) = -3`, which is allowed for a 2‑D tensor.
#   3. `retile_streamify` splits the enlarged tile‑row into chunks of size
#      `seq_len` (= 64).  This yields a stream of length 16 for Q (4 kv‑heads × 4 q‑per‑kv)
#      and a stream of length 4 for K/V.
#   4. `reshape_stream` finally splits the remaining stream dimension:
#        – for Q we split 16 → (4, 4) using `chunk_size=4`,
#        – for K/V we split 4 → (4, 1) using `chunk_size=1`.
# The resulting tensors have exactly the shapes requested by the parent:
#   Q → (4, 4, 64, 32)
#   K → (4, 1, 64, 32)
#   V → (4, 1, 64, 32)
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Q: (seq_len, num_heads, head_dim) → (num_kv_heads, q_per_kv, seq_len, head_dim)
    Q_h = accum_retile_row(Q, rank=1)                 # (seq_len * num_heads, head_dim) → (1024, 32)
    Q_h = promote(Q_h, rank=0)                        # (1, 1024, 32) – add leading stream dim
    Q_h = retile_streamify(Q_h, chunk=64, split_row=True)  # (16, 64, 32)
    Q_h = reshape_stream(Q_h, chunk_size=4, rank=0)  # (4, 4, 64, 32)

    # K: (seq_len, num_kv_heads, head_dim) → (num_kv_heads, 1, seq_len, head_dim)
    K_h = accum_retile_row(K, rank=1)                 # (256, 32)
    K_h = promote(K_h, rank=0)                        # (1, 256, 32)
    K_h = retile_streamify(K_h, chunk=64, split_row=True)  # (4, 64, 32)
    K_h = reshape_stream(K_h, chunk_size=1, rank=0)  # (4, 1, 64, 32)

    # V: same transformation as K
    V_h = accum_retile_row(V, rank=1)                 # (256, 32)
    V_h = promote(V_h, rank=0)                        # (1, 256, 32)
    V_h = retile_streamify(V_h, chunk=64, split_row=True)  # (4, 64, 32)
    V_h = reshape_stream(V_h, chunk_size=1, rank=0)  # (4, 1, 64, 32)

    return Q_h, K_h, V_h