# compute_qkv implements the view/permute logic from the reference model
# using only DSL primitives.
#   Q : (seq_len, num_heads, head_dim)   → (Hkv, qpkv, seq_len, head_dim)
#   K,V : (seq_len, num_kv_heads, head_dim) → (num_kv_heads, 1, seq_len, head_dim)
#
# The transformation is performed by:
#   1. retile_streamify(chunk=1)  – moves the head dimension from tile rows
#      into the stream (tile rows become 1).
#   2. reshape_stream(chunk_size=seq_len) – splits the enlarged stream into
#      (heads, seq_len).
#   3. accum_retile_row(rank=1) – absorbs seq_len back into tile rows,
#      yielding (heads, seq_len, head_dim).
#   4. For Q we further split the combined head dimension (Hkv*qpkv) into two
#      stream dimensions using reshape_stream again.
#   5. For K,V we simply insert a singleton stream dimension with promote
#      to obtain the required (num_kv_heads, 1, seq_len, head_dim) layout.
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------------
    # Q → (Hkv, qpkv, S, D)
    # -----------------------------------------------------------------------
    seq_len = Q.tensor.shape[0]                     # static sequence length (S)
    # Move the head axis (tile rows) into the stream and make tile rows = 1
    q_tmp = retile_streamify(Q, chunk=1)
    # Split the expanded stream (S * H) into (H, S)
    q_tmp = reshape_stream(q_tmp, chunk_size=seq_len, rank=0)
    # Absorb the sequence length back into the tile rows (now tile rows = S)
    q_tmp = accum_retile_row(q_tmp, rank=1)
    # Split the combined head dimension (Hkv * qpkv) into two stream dims
    head_split = out_shapes[0][1]                  # expected = 4
    Qh = reshape_stream(q_tmp, chunk_size=head_split, rank=0)

    # -----------------------------------------------------------------------
    # K → (Hkv, 1, S, D)
    # -----------------------------------------------------------------------
    k_tmp = retile_streamify(K, chunk=1)
    k_tmp = reshape_stream(k_tmp, chunk_size=seq_len, rank=0)
    k_tmp = accum_retile_row(k_tmp, rank=1)
    Kh = promote(k_tmp, rank=0)

    # -----------------------------------------------------------------------
    # V → (Hkv, 1, S, D)
    # -----------------------------------------------------------------------
    v_tmp = retile_streamify(V, chunk=1)
    v_tmp = reshape_stream(v_tmp, chunk_size=seq_len, rank=0)
    v_tmp = accum_retile_row(v_tmp, rank=1)
    Vh = promote(v_tmp, rank=0)

    return (Qh, Kh, Vh)