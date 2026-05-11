# Implementation notes:
# - The heavy attention work is delegated to the blackbox
#   `attention_compute__root_attention_attention_compute`.  It expects the
#   inputs in the GQA‑reshaped layout (Qh, Kh, Vh) with vanilla shape
#   (num_kv_heads, query_per_kv, seq_len, head_dim).  The parent already
#   provides the raw Q, K, V streams (seq_len, num_heads, head_dim); we can
#   forward them directly – the stub inside the blackbox will perform the
#   necessary view/permute on‑chip.
# - The blackbox returns a stream with shape (num_kv_heads, query_per_kv,
#   seq_len, head_dim) i.e. stream dims (kv, qperkv) and tile rows = seq_len.
# - The required output of this node is (seq_len, num_heads, head_dim) which
#   swaps the original stream dimensions with the tile‑row dimension.
# - This swap can be expressed with two DSL primitives:
#     1) `retile_streamify(..., chunk=num_heads, split_row=True)` splits the
#        tile‑row dimension (seq_len) into chunks of size `num_heads` and merges
#        the chunk count into the innermost stream dimension, yielding stream
#        shape (num_kv_heads, num_heads * query_per_kv) and tile rows = num_heads.
#     2) `flatten(..., min_rank=0, max_rank=1)` merges the two stream dimensions
#        back into a single stream dimension (seq_len), giving the final shape
#        (seq_len, num_heads, head_dim).
# - All shape arithmetic is done with plain Python scalars; no prohibited
#   tensor methods are used.

def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # Extract dimensions from the input streams
    seq_len = Q.shape[0]
    num_heads = Q.shape[1]
    head_dim = Q.shape[2]

    num_kv_heads = K.shape[1]
    query_per_kvhead = num_heads // num_kv_heads

    # Call the heavy‑weight attention blackbox.  Its output vanilla shape is:
    # (num_kv_heads, query_per_kvhead, seq_len, head_dim)
    attn = attention_compute__root_attention_attention_compute(
        Q, K, V,
        out_shapes=((num_kv_heads, query_per_kvhead, seq_len, head_dim),),
        out_perms=(None,),
    )

    # Swap the stream dimensions with the tile‑row dimension.
    # 1) Split the tile‑row (seq_len) into chunks of size `num_heads`
    #    and merge the chunk count into the stream.
    attn_retiled = retile_streamify(attn, chunk=num_heads, split_row=True)

    # 2) Merge the two resulting stream dimensions into a single one.
    out = flatten(attn_retiled, min_rank=0, max_rank=1)

    return out