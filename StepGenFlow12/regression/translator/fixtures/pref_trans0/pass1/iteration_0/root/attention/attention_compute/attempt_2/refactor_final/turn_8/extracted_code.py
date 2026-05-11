# attention_compute ---------------------------------------------------------
# Implements the reference logic using only DSL ops.  The heavy attention
# computation is delegated to the child `attention_compute__root_attention_attention_compute`,
# but we first reshape the inputs into the exact GQA layout that the child
# expects (stream dimensions = (num_kv_heads, query_per_kvhead), tile rows = seq_len,
# tile cols = head_dim).  The reshaping is performed with retile/parallelize
# primitives – no tensor‐method calls are used.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0️⃣  Scalar constants (plain Python ints)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # 1️⃣  Split the head dimension into groups of size `query_per_kvhead`
    #     and then parallelize over the KV‑heads.  After this step we
    #     have three lists (one per KV‑head) of tensors shaped
    #         stream(seq_len) × tile(query_per_kvhead, head_dim)   – Q
    #         stream(seq_len) × tile(1, head_dim)                – K, V
    # ------------------------------------------------------------------
    Q_retiled = retile_streamify(Q, chunk=query_per_kvhead, split_row=True)
    Q_chunks = parallelize(Q_retiled, num_kv_heads)

    K_retiled = retile_streamify(K, chunk=1, split_row=True)
    K_chunks = parallelize(K_retiled, num_kv_heads)

    V_retiled = retile_streamify(V, chunk=1, split_row=True)
    V_chunks = parallelize(V_retiled, num_kv_heads)

    # ------------------------------------------------------------------
    # 2️⃣  For each KV‑head, convert the chunk into the child‑call layout:
    #       Qh : stream(1, query_per_kvhead) × tile(seq_len, head_dim)
    #       Kh/Vh : stream(1, query_per_kvhead) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    child_out_shape = ((1, query_per_kvhead, seq_len, head_dim),)
    attn_chunks = []                 # will hold the per‑KV attention results

    for kv in range(num_kv_heads):
        # ------------------------------------------------------------------
        # Qh for this KV‑head
        # ------------------------------------------------------------------
        Qk = Q_chunks[kv]                          # stream(seq) × tile(qpkv, dim)
        Qk = accum_retile_row(Qk)                  # stream()  × tile(seq*qpkv, dim)
        Qk = retile_streamify(Qk, chunk=seq_len, split_row=True)  # stream(qpkv) × tile(seq, dim)
        Qk = promote_outer(Qk)                     # stream(1, qpkv) × tile(seq, dim)

        # ------------------------------------------------------------------
        # Kh for this KV‑head (need to broadcast the single key row to all
        # query‑per‑KV rows)
        # ------------------------------------------------------------------
        Kk = K_chunks[kv]                          # stream(seq) × tile(1, dim)
        Kk = accum_retile_row(Kk)                  # stream()  × tile(seq, dim)
        Kk = retile_streamify(Kk, chunk=seq_len, split_row=True)  # stream(1) × tile(seq, dim)
        Kk = promote_outer(Kk)                     # stream(1,1) × tile(seq, dim)
        Kk = repeat_static(Kk, query_per_kvhead)   # stream(1, qpkv, 1) × tile(seq, dim)
        Kk = flatten(Kk, min_rank=0, max_rank=1)   # stream(1, qpkv) × tile(seq, dim)

        # ------------------------------------------------------------------
        # Vh – identical broadcasting logic as Kh
        # ------------------------------------------------------------------
        Vk = V_chunks[kv]                          # stream(seq) × tile(1, dim)
        Vk = accum_retile_row(Vk)                  # stream()  × tile(seq, dim)
        Vk = retile_streamify(Vk, chunk=seq_len, split_row=True)  # stream(1) × tile(seq, dim)
        Vk = promote_outer(Vk)                     # stream(1,1) × tile(seq, dim)
        Vk = repeat_static(Vk, query_per_kvhead)   # stream(1, qpkv, 1) × tile(seq, dim)
        Vk = flatten(Vk, min_rank=0, max_rank=1)   # stream(1, qpkv) × tile(seq, dim)

        # ------------------------------------------------------------------
        # 3️⃣  Call the heavy‑weight child on the prepared tensors.
        # ------------------------------------------------------------------
        attn_k = attention_compute__root_attention_attention_compute(
            Qk, Kk, Vk, out_shapes=child_out_shape, out_perms=None
        )                                          # stream(1, qpkv) × tile(seq, dim)

        attn_chunks.append(attn_k)

    # ------------------------------------------------------------------
    # 4️⃣  Merge the per‑KV results.  `eager_merge` concatenates along the
    #     first stream dimension, yielding a tensor of shape
    #     (num_kv_heads, query_per_kvhead, seq_len, head_dim).
    # ------------------------------------------------------------------
    merged = eager_merge(attn_chunks)[0]          # (kv, qpkv) × tile(seq, dim)

    # ------------------------------------------------------------------
    # 5️⃣  Collapse the two GQA stream dimensions back to the original
    #     “heads” stream, then swap that stream with the tile‑row dimension
    #     to obtain the contract‑required layout: stream(seq_len) ×
    #     tile(num_heads, head_dim).
    # ------------------------------------------------------------------
    flat = flatten(merged, min_rank=0, max_rank=1)      # stream(num_heads) × tile(seq, dim)
    out = retile_streamify(flat, chunk=num_heads, split_row=True)  # stream(seq) × tile(num_heads, dim)

    # ------------------------------------------------------------------
    # 6️⃣  Return the final stream tensor.
    # ------------------------------------------------------------------
    return out