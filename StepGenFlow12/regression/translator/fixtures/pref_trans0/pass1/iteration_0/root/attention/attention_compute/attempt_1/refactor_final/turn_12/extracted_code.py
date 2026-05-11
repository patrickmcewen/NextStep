# Compute GQA attention using pure DSL ops, ensuring the KV‑head dimension is
# correctly aligned with the query‑per‑KV dimension by expanding K and V with
# `expand_ref` instead of `repeat_static`.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----- Q → Qh : (Hkv, Q_per_KV, S, D) -----
    qh = reshape_stream(Q, chunk_size=4, rank=0)          # → stream(16,4)×tile(16,32)
    qh = accum_retile_row(qh, rank=1)                    # → stream(16,)×tile(64,32)
    qh = reshape_stream(qh, chunk_size=4, rank=0)        # → stream(4,4)×tile(64,32)

    # ----- K → Kh : (Hkv, 1, S, D) -----
    kh = retile_streamify(K, chunk=1, split_row=True)    # → stream(256,)×tile(1,32)
    kh = reshape_stream(kh, chunk_size=64, rank=0)        # → stream(4,64)×tile(1,32)
    kh = accum_retile_row(kh, rank=1)                    # → stream(4,)×tile(64,32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)        # → stream(4,1)×tile(64,32)
    # expand the singleton dimension so it matches Qh’s (Hkv, Q_per_KV) stream shape
    kh = expand_ref(kh, qh, expand_rank=1)               # → stream(4,4)×tile(64,32)

    # ----- V → Vh : (Hkv, 1, S, D) -----
    vh = retile_streamify(V, chunk=1, split_row=True)    # → stream(256,)×tile(1,32)
    vh = reshape_stream(vh, chunk_size=64, rank=0)        # → stream(4,64)×tile(1,32)
    vh = accum_retile_row(vh, rank=1)                    # → stream(4,)×tile(64,32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)        # → stream(4,1)×tile(64,32)
    vh = expand_ref(vh, qh, expand_rank=1)               # → stream(4,4)×tile(64,32)

    # ----- Scaled dot‑product attention -----
    scale = 1.0 / math.sqrt(Q.shape[-1])                 # head_dim = 32
    scores = binary_matmul(qh, kh, weight_transposed=True)   # → stream(4,4)×tile(64,64)
    scores = unary_mul_imm(scores, scale)                # scale
    exp_scores = unary_exp(scores)                       # e^{scores}
    sum_exp = unary_rowwise_sum(exp_scores)              # sum over keys (tile‑col)
    probs = binary_div(exp_scores, sum_exp)              # softmax
    attn = binary_matmul(probs, vh)                      # → stream(4,4)×tile(64,32)

    # ----- Convert back to contract shape (S, H, D) -----
    attn = flatten(attn, min_rank=0, max_rank=1)         # → stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)  # → stream(64,)×tile(16,32)

    return attn