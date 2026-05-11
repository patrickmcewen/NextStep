# Implementation reasoning:
# - Qh, Kh, Vh are on‑chip streams:
#     Qh : (kv_head=4, query_per_kvhead=4, seq_len=64, head_dim=32)
#     Kh : (kv_head=4, 1, seq_len=64, head_dim=32)
#     Vh : (kv_head=4, 1, seq_len=64, head_dim=32)
# - We first broadcast Kh and Vh across the query‑per‑kv dimension (`expand_ref`).
# - Perform GQA attention using DSL primitives:
#       scores = Qh @ Khᵀ
#       e      = exp(scores)
#       denom  = row‑wise sum of e
#       num    = e @ Vh
#       attn   = num / denom
#   (the stable‑softmax max subtraction is omitted – it does not change the result.)
# - The result after the division has shape  stream(4,4)×tile(64,32).
#   We need to reshape it to the contract‑requested stream shape
#   (seq_len=64) with tile shape (num_heads=16, head_dim=32).
#   Steps:
#   1. `retile_streamify(..., chunk=1, split_row=True)` moves the sequence
#      dimension into the stream, yielding stream(4,4,64)×tile(1,32).
#   2. `flatten(..., min_rank=0, max_rank=1)` collapses the first two stream
#      dimensions (kv_head and query_per_kvhead) into one, producing a single
#      stream dimension of size 4·4·64 = 1024.
#   3. `reshape_stream(..., chunk_size=num_heads, rank=0)` splits that flat
#      dimension into (seq_len, num_heads) → stream(64,16)×tile(1,32).
#   4. `accum_retile_row(..., rank=1)` merges the num_heads stream dimension
#      into the tile‑row dimension, giving the final shape
#      stream(64)×tile(16,32), i.e. exactly the required out_shape
#      (64, 16, 32).
#
# The function returns the correctly shaped stream; the parent will handle the
# off‑chip store.

def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # unpack the expected vanilla output shape
    seq_len, num_heads, head_dim = out_shapes[0]

    # -----------------------------------------------------------------
    # 1. Broadcast KV across the query‑per‑kv dimension
    # -----------------------------------------------------------------
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)  # (kv_head, query_per_kvhead, seq_len, head_dim)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # -----------------------------------------------------------------
    # 2. GQA attention (stable softmax – max subtraction omitted)
    # -----------------------------------------------------------------
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4,4)×tile(64,64)
    e = unary_exp(scores)                                      # (4,4)×tile(64,64)
    denom = unary_rowwise_sum(e)                               # (4,4)×tile(64,1)
    num = binary_matmul(e, Vh_exp, weight_transposed=False)    # (4,4)×tile(64,32)
    attn = binary_div(num, denom)                              # (4,4)×tile(64,32)

    # -----------------------------------------------------------------
    # 3. Reshape to (seq_len) stream with (num_heads, head_dim) tile
    # -----------------------------------------------------------------
    # Move the sequence length out of the tile into the stream
    attn = retile_streamify(attn, chunk=1, split_row=True)    # stream(4,4,64)×tile(1,32)

    # Collapse the kv‑head dimensions into one stream dimension
    attn = flatten(attn, min_rank=0, max_rank=1)              # stream(1024,)×tile(1,32)

    # Split that flat dimension into (seq_len, num_heads)
    attn = reshape_stream(attn, chunk_size=num_heads, rank=0) # stream(64,16)×tile(1,32)

    # Merge the num_heads stream dimension into the tile‑row dimension
    attn = accum_retile_row(attn, rank=1)                     # stream(64)×tile(16,32)

    return attn