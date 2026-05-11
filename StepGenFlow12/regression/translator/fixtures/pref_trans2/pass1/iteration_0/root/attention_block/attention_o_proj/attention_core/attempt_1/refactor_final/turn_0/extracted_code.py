# ----------------------------------------------------------------------
# Implementation notes:
#   * Q, K, V arrive as on‑chip streams:
#       Q : (1, 64, 16, 32)   – tile rows = 16 (num_heads)
#       K : (1, 64,  4, 32)   – tile rows = 4  (num_kv_heads)
#       V : (1, 64,  4, 32)   – same layout as K
#   * The child `attention_compute` expects its inputs in vanilla shape:
#       Qh : (4, 4, 64, 32)   – (kv_heads, q_per_kv, seq_len, head_dim)
#       Kh : (4, 1, 64, 32)
#       Vh : (4, 1, 64, 32)
#   * We convert the on‑chip streams to the required layout using only DSL
#     ops:
#       – `retile_streamify(..., chunk=1, split_row=True)` moves all tile‑row
#         elements into the stream dimension, leaving tile rows = 1.
#       – `reshape_stream` splits a stream dimension into two stream dimensions.
#         By applying it twice we obtain the (kv, q_per_kv, seq) ordering needed
#         for Q, and (kv, seq) for K/V.
#   * After the conversion we call `attention_compute`, forwarding the
#     `out_shapes`/`out_perms` that the parent supplied.  The requested output
#     shape is [(1, 64, 16, 32)], which matches the child’s ability to map its
#     vanilla result (4,4,64,32) back into that tiled form.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Q -> (kv, q_per_kv, seq) with tile rows = 1
    Qh = retile_streamify(Q, chunk=1, split_row=True)        # (1,1024,1,32)
    Qh = reshape_stream(Qh, chunk_size=64, rank=0)           # (1,16,64,1,32)
    Qh = reshape_stream(Qh, chunk_size=4, rank=1)            # (1,4,4,64,1,32)

    # K -> (kv, seq) with tile rows = 1
    Kh = retile_streamify(K, chunk=1, split_row=True)        # (1,256,1,32)
    Kh = reshape_stream(Kh, chunk_size=64, rank=0)           # (1,4,64,1,32)

    # V -> (kv, seq) with tile rows = 1
    Vh = retile_streamify(V, chunk=1, split_row=True)        # (1,256,1,32)
    Vh = reshape_stream(Vh, chunk_size=64, rank=0)           # (1,4,64,1,32)

    # Core attention computation; output shape/permutation are dictated by the parent.
    attn = attention_compute(Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms)

    return attn