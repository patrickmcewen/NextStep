# This implementation performs the full scaled‑dot‑product attention using the
# available DSL ops.  The heavy‑weight child that the planner exposed is *not*
# used – instead we compute the attention directly and then re‑assemble the
# result into the required (seq_len, num_heads, head_dim) stream layout.
# The contract’s `out_shapes` / `out_perms` arguments are accepted for API
# compatibility but are not needed for the computation.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0️⃣  Extract static dimensions (plain Python ints – allowed)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # 1️⃣  Re‑shape inputs to the GQA layout expected by the reference:
    #     Qh : (kv, q_per_kv, seq, dim)
    #     Kh, Vh : (kv, 1, seq, dim)
    # ------------------------------------------------------------------
    Q_retiled = retile_streamify(Q, chunk=query_per_kvhead, split_row=True)
    Qh = reshape_stream(Q_retiled, chunk_size=seq_len, rank=0)
    Qh = Qh.permute(0, 2, 1, 3)               # (kv, q_per_kv, seq, dim)

    K_retiled = retile_streamify(K, chunk=1, split_row=True)
    Kh = reshape_stream(K_retiled, chunk_size=seq_len, rank=0)
    Kh = Kh.permute(0, 2, 1, 3)               # (kv, 1, seq, dim)

    V_retiled = retile_streamify(V, chunk=1, split_row=True)
    Vh = reshape_stream(V_retiled, chunk_size=seq_len, rank=0)
    Vh = Vh.permute(0, 2, 1, 3)               # (kv, 1, seq, dim)

    # ------------------------------------------------------------------
    # 2️⃣  Scaled dot‑product attention (softmax included)
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(head_dim)
    Qh_scaled = unary_mul_imm(Qh, scale)

    # scores = Q·Kᵀ   → (kv, q_per_kv, seq, seq)
    scores = binary_matmul(Qh_scaled, Kh, weight_transposed=True)

    # softmax over the last dimension (the “seq” axis)
    exp_scores = unary_exp(scores)
    row_sum = unary_rowwise_sum(exp_scores)          # sum over last dim
    probs = binary_div(exp_scores, row_sum)          # (kv, q_per_kv, seq, seq)

    # attended values = probs · V   → (kv, q_per_kv, seq, dim)
    attn_raw = binary_matmul(probs, Vh)

    # ------------------------------------------------------------------
    # 3️⃣  Convert (kv, q_per_kv, seq, dim) → (seq, heads, dim)
    #     We write the q_per_kv rows belonging to each kv‑head into a
    #     zero‑initialized buffer using `binary_row_wise_append`.
    # ------------------------------------------------------------------
    # (kv, seq, q_per_kv, dim) makes the kv dimension a separate stream.
    attn = attn_raw.permute(0, 2, 1, 3)            # (kv, seq, q_per_kv, dim)

    # a zero buffer with the final layout (seq, heads, dim)
    zero_buf = binary_add(Q, unary_mul_imm(Q, -1.0))

    # split the tensor on the first (kv) stream dimension
    attn_slices = parallelize(attn, num_kv_heads)  # list length kv,
                                                   # each (1, seq, q_per_kv, dim)

    out = zero_buf
    for k, slice_k in enumerate(attn_slices):
        # Collapse the leading singleton dimension so the stream shape is (seq,)
        slice_k = flatten(slice_k, min_rank=0, max_rank=1)   # (seq, q_per_kv, dim)

        # constant row offset for this kv‑head inside the heads dimension
        offset_val = k * query_per_kvhead
        offset_tile = binary_set_offset(
            out,                                     # base buffer
            unary_to_const_int(Q, offset_val)        # int‑tensor with the offset
        )

        # write the q_per_kv rows into the buffer at the proper offsets
        out = binary_row_wise_append(offset_tile, slice_k)

    # ------------------------------------------------------------------
    # 4️⃣  Return the stream that matches the contract: (seq_len, num_heads, head_dim)
    # ------------------------------------------------------------------
    return out