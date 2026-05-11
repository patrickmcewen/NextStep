# attention_compute ---------------------------------------------------------
# Convert Q, K, V (already on‑chip) into the GQA layout required by the
# heavy‑weight child, call the child, and let the child’s stub apply the
# permutation (2,0,1,3) and reshape to the contract‑required shape
# (seq_len, num_heads, head_dim).  No further tensor manipulation is needed.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0️⃣  Scalars (plain Python ints – allowed)
    # ------------------------------------------------------------------
    seq_len          = Q.shape[0]                     # 64
    num_heads        = Q.shape[1]                     # 16
    head_dim         = Q.shape[2]                     # 32
    num_kv_heads     = K.shape[1]                     # 4
    query_per_kvhead = num_heads // num_kv_heads      # 4

    # ------------------------------------------------------------------
    # Helper: reshape a (seq, heads, dim) stream into GQA layout:
    #   stream (num_kv_heads, query_per_kv) × tile (seq_len, head_dim)
    # ------------------------------------------------------------------
    def _to_gqa(tensor, q_per_kv):
        # 1) split the head dimension into chunks of size q_per_kv
        x = retile_streamify(tensor, chunk=q_per_kv, split_row=True)

        # 2) split the combined stream (seq_len * (heads/q_per_kv)) into
        #    (num_kv_heads, seq_len) while keeping tile rows = q_per_kv
        x = reshape_stream(x, chunk_size=seq_len, rank=0)

        # 3) merge the seq_len stream dimension into the tile rows
        x = accum_retile_row(x)

        # 4) split the enlarged tile rows back into seq‑sized chunks,
        #    promoting q_per_kv to a new stream dimension
        x = retile_streamify(x, chunk=seq_len, split_row=True)

        # 5) finally split the merged stream (num_kv_heads * q_per_kv)
        #    into the two desired stream dimensions
        x = reshape_stream(x, chunk_size=q_per_kv, rank=0)
        return x

    # ------------------------------------------------------------------
    # 1️⃣  Build Qh, Kh, Vh in the layout the child expects.
    # ------------------------------------------------------------------
    Qh = _to_gqa(Q, query_per_kvhead)   # (num_kv_heads, query_per_kv) × tile (seq_len, head_dim)
    Kh = _to_gqa(K, 1)                  # (num_kv_heads, 1)               × tile (seq_len, head_dim)
    Vh = _to_gqa(V, 1)                  # (num_kv_heads, 1)               × tile (seq_len, head_dim)

    # ------------------------------------------------------------------
    # 2️⃣  Heavy‑weight attention child.
    #     The child’s vanilla output is (kv, q_per_kv, seq, dim);
    #     we request the final layout (seq, heads, dim) by supplying the
    #     contract’s out_shapes and the permutation that matches the
    #     reference code: (2, 0, 1, 3).
    # ------------------------------------------------------------------
    child_perm = ((2, 0, 1, 3),)  # permute (kv, qpkv, seq, dim) → (seq, kv, qpkv, dim)
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=out_shapes, out_perms=child_perm
    )

    # ------------------------------------------------------------------
    # 3️⃣  Return the stream tensor that already matches the contract.
    # ------------------------------------------------------------------
    return attn