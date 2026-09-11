# Implementation reasoning:
# Qh, Kh, Vh are already on‑chip streams:
#   Qh : (kv_head, query_per_kvhead, seq_len, head_dim) = (4,4,64,32)
#   Kh : (kv_head, 1, seq_len, head_dim)                = (4,1,64,32)
#   Vh : (kv_head, 1, seq_len, head_dim)                = (4,1,64,32)
#
# 1. Broadcast the KV tensors across the query‑per‑KV dimension with `expand_ref`.
# 2. Compute attention using the usual GQA formulation:
#        scores = Qh @ Khᵀ
#        e      = exp(scores)            (stable max subtraction omitted)
#        denom  = Σⱼ eᵢⱼ   (row‑wise sum)
#        num    = e @ Vh
#        attn   = num / denom
#    All ops are expressed with DSL binary/unary primitives.
# 3. Convert the (seq_len, head_dim) tile into stream dimensions:
#    - `retile_streamify(..., chunk=1, split_row=True)` moves each row of the
#      64‑element sequence into its own stream token, yielding tile shape (1,32).
#    - `flatten` collapses the two remaining stream dimensions (kv_head, token)
#      into one.
#    - `reshape_stream` splits that flat dimension into (seq_len, num_heads)
#      using the `num_heads` value taken from the contract’s `out_shapes`.
# 4. Move the head‑dimension (32) out of the tile into the stream:
#    - `retile_streamify(..., chunk=1, split_row=False)` splits the tile‑column
#      (size 32) into a stream factor, giving tile shape (1,1) and enlarging the
#      second stream dimension to `num_heads * head_dim`.
#    - `reshape_stream` finally splits that combined dimension into the
#      separate `num_heads` and `head_dim` stream axes, again using values from
#      `out_shapes`.
# The resulting tensor has stream shape (seq_len, num_heads, head_dim) and tile
# shape (1,1), i.e. a shape compatible with the declared vanilla output
# `(64, 16, 32)`. No off‑chip loads/stores are performed here; the parent stub
# will handle the off‑chip write.

def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # unpack expected vanilla output shape
    seq_len, num_heads, head_dim = out_shapes[0]

    # -----------------------------------------------------------------
    # 1. Broadcast KV across the query‑per‑KV dimension
    # -----------------------------------------------------------------
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)  # (kv_head, query_per_kvhead, seq_len, head_dim)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # -----------------------------------------------------------------
    # 2. GQA attention (stable softmax – max subtraction omitted)
    # -----------------------------------------------------------------
    # scores = Qh @ Khᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # e = exp(scores)
    e = unary_exp(scores)

    # denom = row‑wise sum of e  (keeps a singleton column dimension)
    denom = unary_rowwise_sum(e)

    # num = e @ Vh
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # attn = num / denom   (broadcast division over the singleton column)
    attn = binary_div(num, denom)

    # -----------------------------------------------------------------
    # 3. Turn the (seq_len, head_dim) tile into stream dimensions
    # -----------------------------------------------------------------
    # split the 64‑row tile into individual tokens (tile rows -> stream)
    attn = retile_streamify(attn, chunk=1, split_row=True)   # → stream(... ) × tile(1, head_dim)

    # collapse the two remaining stream dimensions (kv_head, token) into one
    attn = flatten(attn, min_rank=0, max_rank=1)            # → stream(1024,) × tile(1, head_dim)

    # reshape the flat stream into (seq_len, num_heads) stream dims
    attn = reshape_stream(attn, chunk_size=num_heads, rank=0)  # → stream(seq_len, num_heads) × tile(1, head_dim)

    # -----------------------------------------------------------------
    # 4. Move head_dim out of the tile and make it a stream axis
    # -----------------------------------------------------------------
    # split the tile‑column (size head_dim) into a stream factor of size head_dim
    attn = retile_streamify(attn, chunk=1, split_row=False)   # → stream(seq_len, num_heads*head_dim) × tile(1,1)

    # finally split that combined dimension back into (num_heads, head_dim)
    attn = reshape_stream(attn, chunk_size=head_dim, rank=0)   # → stream(seq_len, num_heads, head_dim) × tile(1,1)

    return attn