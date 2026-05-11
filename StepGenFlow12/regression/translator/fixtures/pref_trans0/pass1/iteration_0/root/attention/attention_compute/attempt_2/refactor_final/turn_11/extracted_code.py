# attention_compute ---------------------------------------------------------
# Transform the on‑chip inputs Q, K, V into the GQA layout expected by the
# heavy‑weight child, invoke the child, and let the child’s stub perform the
# final permutation/reshape (via out_perms) to match the contract’s
# vanilla shape (seq_len, num_heads, head_dim).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0️⃣  Scalars (plain Python ints)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Helper: reshape a (seq, heads) stream into GQA layout:
    #   (num_kv_heads, query_per_kv) × tile(seq, dim)
    # ------------------------------------------------------------------
    def _to_gqa(tensor, q_per_kv):
        # 1) split the tile‑row (heads) into chunks of size q_per_kv
        x = retile_streamify(tensor, chunk=q_per_kv, split_row=True)

        # 2) split the combined stream (seq * (heads/q_per_kv))
        #    into (num_kv_heads, seq) while keeping tile rows = q_per_kv
        x = reshape_stream(x, chunk_size=seq_len, rank=0)

        # 3) merge the seq stream dimension into the tile rows
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
    Qh = _to_gqa(Q, query_per_kvhead)   # (num_kv_heads, query_per_kv) × tile(seq, dim)
    Kh = _to_gqa(K, 1)                  # (num_kv_heads, 1)               × tile(seq, dim)
    Vh = _to_gqa(V, 1)                  # (num_kv_heads, 1)               × tile(seq, dim)

    # ------------------------------------------------------------------
    # 2️⃣  Heavy‑weight attention (the child).  We ask the child to emit the
    #     final (seq, num_heads, dim) layout by supplying the contract’s
    #     out_shapes and the permutation that matches the reference code:
    #       permute(2, 0, 1, 3)  →  (seq, kv, q_per_kv, dim) → reshape → (seq, heads, dim)
    # ------------------------------------------------------------------
    child_perm = ((2, 0, 1, 3),) if out_perms is None else out_perms
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=out_shapes, out_perms=child_perm
    )

    # ------------------------------------------------------------------
    # 3️⃣  Return the child’s output directly (it already matches the
    #     contract’s vanilla shape (seq_len, num_heads, head_dim)).
    # ------------------------------------------------------------------
    return attn