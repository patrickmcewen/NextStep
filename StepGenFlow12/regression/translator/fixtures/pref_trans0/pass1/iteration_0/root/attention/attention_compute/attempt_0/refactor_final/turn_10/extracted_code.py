def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # The parent already supplies Q, K, V as on‑chip streams.
    # We can delegate the entire heavy computation to the blackbox.
    # The blackbox expects the GQA layout (kv, qp, seq, dim).  Its stub
    # internally performs the required view/permute on the inputs and then
    # computes attention.  To match the contract of this node we ask the
    # blackbox to emit the final layout directly:
    #   (seq_len, num_heads, head_dim)  ←  permute(2,0,1,3) + reshape
    seq_len = Q.shape[0]          # 64
    num_heads = Q.shape[1]        # 16
    head_dim = Q.shape[2]         # 32

    # Request the blackbox to produce the correctly permuted output.
    attn = attention_compute__root_attention_attention_compute(
        Q,
        K,
        V,
        out_shapes=((seq_len, num_heads, head_dim),),      # vanilla shape expected by parent
        out_perms=((2, 0, 1, 3),),                         # permute to (seq, heads, dim)
    )
    return attn