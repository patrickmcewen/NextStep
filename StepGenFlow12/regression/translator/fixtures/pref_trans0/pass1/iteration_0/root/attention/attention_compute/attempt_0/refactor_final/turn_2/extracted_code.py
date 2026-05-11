# We receive on‑chip streams Q, K, V with vanilla shape (S, H, D):
#   Q: stream(S)×tile(H, D)   where S=64, H=16, D=32
#   K, V: stream(S)×tile(Hkv, D) where Hkv=4.
# The heavy attention computation is delegated to the child
# `attention_compute__root_attention_attention_compute`, which expects the
# inputs in GQA layout (num_kv_heads, query_per_kv, seq_len, head_dim) and
# returns a stream with shape (num_kv_heads, query_per_kv) and tile rows
# = seq_len.  We pass the original Q, K, V directly – the child stub will
# reinterpret them correctly – and specify the output vanilla shape.
#
# The child’s output stream is (kv, q_per_kv)×tile(seq_len, head_dim)
# i.e. stream(4,4)×tile(64,32).  To match the contract of this node we need
# a stream of shape (seq_len,)×tile(num_heads, head_dim) = (64,)×tile(16,32).
# This is achieved by:
#   1) flattening the two stream dimensions (4,4) into one (size 16);
#   2) retile_streamify to split the tile‑row dimension (64) into chunks of
#      size num_heads (16), merging the chunk count (4) back into the stream.
#
# The resulting tensor has the required shape (64,16,32).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # Extract problem dimensions from the incoming streams
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # Call the heavyweight attention blackbox.
    # The stub will reinterpret Q/K/V as Qh/Kh/Vh internally.
    attn = attention_compute__root_attention_attention_compute(
        Q,
        K,
        V,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),  # vanilla shape of attn
        ),
        out_perms=(None,),
    )

    # 1) Merge the two stream dimensions (kv, q_per_kv) → single stream dim = 16
    attn_flat = flatten(attn, min_rank=0, max_rank=1)

    # 2) Split tile‑row (seq_len) into chunks of size `num_heads` (16),
    #    moving the chunk count into the stream dimension.
    out = retile_streamify(attn_flat, chunk=num_heads, split_row=True)

    return out