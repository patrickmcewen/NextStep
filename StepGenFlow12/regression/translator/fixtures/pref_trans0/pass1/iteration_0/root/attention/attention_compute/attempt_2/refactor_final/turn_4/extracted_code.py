# The reference implementation builds Qh, Kh, Vh by reshaping and permuting the
# original tensors:
#   Qh = Q.view(S, KV, QPKV, D).permute(1, 2, 0, 3)
#   Kh = K.permute(1, 0, 2).unsqueeze(1)
#   Vh = V.permute(1, 0, 2).unsqueeze(1)
# The DSL does not have a direct “permute”, so we recreate the same layout by
# 1) splitting the head dimension into KV‑heads × queries‑per‑KV‑head,
# 2) separating the KV‑heads into independent streams (via `parallelize`),
# 3) moving the sequence dimension from the stream into the tile‑row dimension,
#    and finally
# 4) adding a leading singleton stream dimension so the child sees the exact
#    shapes it expects.
#
# After the child returns its attention tensor (shape (KV, QPKV, S, D)),
# we merge the per‑KV results back into the original (S, H, D) layout using
# the same row‑wise‑append pattern that the reference kernel uses.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalars extracted from the input shapes (plain Python ints)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                    # 64
    num_heads = Q.shape[1]                  # 16
    head_dim = Q.shape[2]                   # 32
    num_kv_heads = K.shape[1]               # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # 1️⃣  Split Q / K / V into per‑KV streams.
    #     After this step:
    #       Q_chunks[i] : (seq, QPKV, dim)   – queries belonging to KV i
    #       K_chunks[i] : (seq, 1,   dim)   – keys  belonging to KV i
    #       V_chunks[i] : (seq, 1,   dim)   – values belonging to KV i
    # ------------------------------------------------------------------
    Q_retiled = retile_streamify(Q, chunk=query_per_kvhead, split_row=True)
    Q_chunks = parallelize(Q_retiled, num_kv_heads)

    K_retiled = retile_streamify(K, chunk=1, split_row=True)
    K_chunks = parallelize(K_retiled, num_kv_heads)

    V_retiled = retile_streamify(V, chunk=1, split_row=True)
    V_chunks = parallelize(V_retiled, num_kv_heads)

    # ------------------------------------------------------------------
    # 2️⃣  Helper that converts a (seq, R, dim) tile into
    #     (1, R, seq, dim) – the exact layout the child expects.
    # ------------------------------------------------------------------
    def _to_child_form(tile):
        # tile : (seq, R, dim)  – R is either QPKV or 1
        # ① merge stream (seq) with tile rows (R) -> (seq*R, dim)
        merged = accum_retile_row(tile)
        # ② add a leading singleton stream dimension
        merged = promote_outer(merged)               # (1, seq*R, dim)
        # ③ split the huge tile‑row back into (R, seq)
        reshaped = retile_streamify(
            merged, chunk=seq_len, split_row=True
        )                                            # (R, seq, dim)
        # ④ prepend the outer singleton that represents the KV‑head index
        return promote_outer(reshaped)               # (1, R, seq, dim)

    # ------------------------------------------------------------------
    # 3️⃣  Run the child for each KV‑head and merge the results.
    #     The child output shape per call: (1, QPKV, seq, dim)
    # ------------------------------------------------------------------
    # Zero buffer that will receive the merged result:
    zero_out = binary_add(Q, unary_mul_imm(Q, -1.0))   # (seq, heads, dim)

    # The child expects a single output shape; we provide it explicitly.
    child_out_shape = ((1, query_per_kvhead, seq_len, head_dim),)

    for kv in range(num_kv_heads):
        # ---- Qh for this KV -------------------------------------------------
        Qh = _to_child_form(Q_chunks[kv])            # (1, QPKV, seq, dim)

        # ---- Kh / Vh for this KV (need a leading singleton for the extra
        #      stream dimension).  The pipeline is the same as for Qh but with
        #      R = 1.
        Kh = _to_child_form(K_chunks[kv])            # (1, 1, seq, dim)
        Vh = _to_child_form(V_chunks[kv])            # (1, 1, seq, dim)

        # ---- Heavy‑weight attention ----------------------------------------
        attn_kv = attention_compute__root_attention_attention_compute(
            Qh, Kh, Vh, out_shapes=child_out_shape, out_perms=None
        )                                            # (1, QPKV, seq, dim)

        # ---- Rearrange to (seq, QPKV, dim) and merge into the zero buffer.
        #      The child’s output is (1, QPKV, seq, dim); we move the
        #      KV‑dimension into the row offset so that each group of
        #      QPKV rows lands at the correct head positions.
        slice_k = attn_kv.permute(0, 2, 1, 3)        # (1, seq, QPKV, dim)
        slice_k = flatten(slice_k, min_rank=0, max_rank=1)  # (seq, QPKV, dim)

        # offset for this KV‑head (how many head rows to skip)
        offset_val = kv * query_per_kvhead
        # a tensor of shape (seq, 1, 1) containing the constant offset.
        # We create it by broadcasting a constant over the stream dimension.
        offset_tile = binary_set_offset(
            zero_out, unary_to_const_int(Q, offset_val)
        )
        # Write the slice into the zero buffer at the desired offset.
        zero_out = binary_row_wise_append(offset_tile, slice_k)

    # ------------------------------------------------------------------
    # 4️⃣  `zero_out` now has shape (seq, heads, dim) – exactly the
    #     contract‑required layout.
    # ------------------------------------------------------------------
    return zero_out