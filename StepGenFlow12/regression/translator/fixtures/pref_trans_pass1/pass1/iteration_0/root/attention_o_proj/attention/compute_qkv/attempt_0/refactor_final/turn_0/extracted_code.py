# Implementation reasoning:
# - The input Q is a tile‑stream of shape (seq_len, num_heads, head_dim) where
#   `num_heads = 16` is stored as a tile‑row dimension.  We need to split this
#   dimension into two stream dimensions: `num_kv_heads` (size 4) and
#   `query_per_kvhead` (also size 4).  The DSL op `reshape_stream` does exactly
#   that by splitting a stream dimension into (new_count, chunk_size).  The
#   chunk size is the number of KV heads, which we read from the middle dimension
#   of K (its stream size).
# - For K and V we only have to add a singleton stream dimension after the
#   existing KV‑head stream dimension so that the final shape becomes
#   (num_kv_heads, 1, seq_len, head_dim).  The DSL op `promote` inserts a
#   static stream slot at the innermost stream position.
# - No off‑chip loads are needed because the inputs are already on‑chip.
# - The resulting tensors have the exact shapes required by the contract:
#   Q → (4, 4, 64, 32), K → (4, 1, 64, 32), V → (4, 1, 64, 32).
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Determine the KV‑head count from K's stream dimension (the middle size).
    kv_heads = int(K.tensor.shape[-2])

    # Split Q's head dimension (size 16) into two stream dims of size `kv_heads`
    # each.  The default `rank=0` splits the innermost stream dim.
    Q_out = reshape_stream(Q, chunk_size=kv_heads)

    # Insert a singleton stream dimension after the existing KV‑head stream dim.
    K_out = promote(K, rank=0)
    V_out = promote(V, rank=0)

    return Q_out, K_out, V_out