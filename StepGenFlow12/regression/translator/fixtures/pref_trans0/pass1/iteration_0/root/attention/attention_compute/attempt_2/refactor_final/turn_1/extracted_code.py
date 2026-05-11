# The parent supplies Q, K, V already as on‑chip streams with shape
# (seq_len, num_heads, head_dim) → (64, 16, 32).
# The heavy attention work lives in the child
# `attention_compute__root_attention_attention_compute`, which expects the
# GQA‑re‑shaped tensors Qh, Kh, Vh with vanilla shape
# (num_kv_heads, query_per_kvhead, seq_len, head_dim) =
# (4, 4, 64, 32).  We therefore:
#   1. Split the head dimension (16) into the two GQA dimensions
#      (num_kv_heads=4, query_per_kvhead=4) using `retile_streamify`.
#   2. Pull the sequence dimension out of the stream and make it a tile
#      dimension using `reshape_stream`.
#   3. Apply a permute to get the exact ordering the child expects.
#      (Permute is applied *after* the DSL ops, i.e. on the tensor that will
#      be handed to the child; the contract permits this because the tensor
#      is already a stream produced by DSL ops.)
#   4. Call the child, requesting the final output shape ((64, 16, 32),).
#   5. The child already returns the correctly‑shaped result, so we return it
#      directly—no further reshaping is needed.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Constants derived from the input shapes (plain Python ints)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # 1️⃣  Q → Qh : (seq_len, num_heads, head_dim) → (kv, q_per_kv, seq, dim)
    # ------------------------------------------------------------------
    Q_retiled = retile_streamify(Q, chunk=query_per_kvhead, split_row=True)
    # Q_retiled : stream(256,)×tile(4,32)   → 256 = 64 * (16/4)
    Qh = reshape_stream(Q_retiled, chunk_size=seq_len, rank=0)
    # Qh : stream(4, 64)×tile(4,32)   →  (kv, seq, q_per_kv, dim)
    Qh = Qh.permute(0, 2, 1, 3)      # (kv, q_per_kv, seq, dim)

    # ------------------------------------------------------------------
    # 2️⃣  K → Kh : (seq_len, num_kv_heads, head_dim) → (kv, 1, seq, dim)
    # ------------------------------------------------------------------
    K_retiled = retile_streamify(K, chunk=1, split_row=True)
    Kh = reshape_stream(K_retiled, chunk_size=seq_len, rank=0)
    Kh = Kh.permute(0, 2, 1, 3)      # (kv, 1, seq, dim)

    # ------------------------------------------------------------------
    # 3️⃣  V → Vh : same pattern as K
    # ------------------------------------------------------------------
    V_retiled = retile_streamify(V, chunk=1, split_row=True)
    Vh = reshape_stream(V_retiled, chunk_size=seq_len, rank=0)
    Vh = Vh.permute(0, 2, 1, 3)      # (kv, 1, seq, dim)

    # ------------------------------------------------------------------
    # 4️⃣  Heavy attention: delegate to the child blackbox.
    #     The child receives Qh, Kh, Vh in the exact layout it expects,
    #     and we ask it to emit the final (seq_len, num_heads, head_dim)
    #     layout via `out_shapes`.
    # ------------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms
    )

    # ------------------------------------------------------------------
    # 5️⃣  The child's output already matches the contract, so we return it.
    # ------------------------------------------------------------------
    return attn