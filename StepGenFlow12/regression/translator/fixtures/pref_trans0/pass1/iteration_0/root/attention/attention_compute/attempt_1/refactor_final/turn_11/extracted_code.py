# Compute GQA‑scaled dot‑product attention using pure DSL ops.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Reshape Q, K, V so that the token dimension becomes a tile dimension
    # and the head dimensions become stream dimensions.
    # ------------------------------------------------------------------
    # Q: (S=64, H=16, D=32) → stream(4,4)×tile(64,32)
    q = retile_streamify(Q, chunk=1, split_row=True)          # → stream(1024,)×tile(1,32)
    q = reshape_stream(q, chunk_size=64, rank=0)              # → stream(16,64)×tile(1,32)
    q = accum_retile_row(q, rank=1)                           # → stream(16,)×tile(64,32)
    q = reshape_stream(q, chunk_size=4, rank=0)               # → stream(4,4)×tile(64,32)

    # K: (S=64, Hkv=4, D=32) → stream(4,4)×tile(64,32) (duplicated over Q‑per‑KV)
    k = retile_streamify(K, chunk=1, split_row=True)          # → stream(256,)×tile(1,32)
    k = reshape_stream(k, chunk_size=64, rank=0)              # → stream(4,64)×tile(1,32)
    k = accum_retile_row(k, rank=1)                           # → stream(4,)×tile(64,32)
    k = repeat_static(k, factor=4)                            # → stream(4,4)×tile(64,32)

    # V: (S=64, Hkv=4, D=32) → stream(4,4)×tile(64,32) (duplicated over Q‑per‑KV)
    v = retile_streamify(V, chunk=1, split_row=True)          # → stream(256,)×tile(1,32)
    v = reshape_stream(v, chunk_size=64, rank=0)              # → stream(4,64)×tile(1,32)
    v = accum_retile_row(v, rank=1)                           # → stream(4,)×tile(64,32)
    v = repeat_static(v, factor=4)                            # → stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Scaled dot‑product attention:
    #   scores = Q @ Kᵀ / sqrt(D)
    #   probs  = softmax(scores, dim=-1)
    #   out    = probs @ V
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(Q.shape[-1])                     # D = 32
    scores = binary_matmul(q, k, weight_transposed=True)      # → stream(4,4)×tile(64,64)
    scores = unary_mul_imm(scores, scale)                    # scaling
    exp_scores = unary_exp(scores)                           # e^{scores}
    sum_exp = unary_rowwise_sum(exp_scores)                  # Σₖ e^{scores} over key tokens
    probs = binary_div(exp_scores, sum_exp)                  # softmax
    attn = binary_matmul(probs, v)                           # → stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Merge the two GQA stream dimensions and move the token dimension
    # back to the stream rank, yielding the contract’s expected shape.
    # ------------------------------------------------------------------
    attn = flatten(attn, min_rank=0, max_rank=1)               # stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)   # stream(64,)×tile(16,32)

    return attn