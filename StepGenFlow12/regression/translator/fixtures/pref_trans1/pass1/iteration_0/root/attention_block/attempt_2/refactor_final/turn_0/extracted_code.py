# The attention block is built by chaining the provided blackboxes.
# Each blackbox is given the exact stream shape we need for its output,
# expressed as a tuple of integers (stream_dim, tile_rows, tile_cols).
# All raw off‑chip tensors are passed directly to the children – the
# children internally perform the off‑chip load and reshape to the
# vanilla shapes they expect. No explicit tensor arithmetic or PyTorch
# methods are used; only tuple construction and blackbox calls appear.
def attention_block(input_tensor, q_proj, k_proj, v_proj, cos, sin,
                    o_proj_weight, *, out_shapes, out_perms=None):
    # 1️⃣ Pre‑attention RMSNorm: produce a stream (64, 1, 512) so the
    #    subsequent projection sees the expected vanilla shape (64, 512).
    normed = pre_attention_norm(
        input_tensor,
        out_shapes=((64, 1, 512),),
        out_perms=(None,),
    )

    # 2️⃣ QKV projections: each head dimension is emitted as a separate stream.
    Q, K, V = qkv_projection(
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=(
            (64, 16, 32),   # Q
            (64, 4, 32),    # K
            (64, 4, 32),    # V
        ),
        out_perms=(None, None, None),
    )

    # 3️⃣ Per‑head RMSNorm
    Q, K = per_head_norm(
        Q,
        K,
        out_shapes=((64, 16, 32), (64, 4, 32)),
        out_perms=(None, None),
    )

    # 4️⃣ Apply RoPE
    Q, K = apply_rope(
        Q,
        K,
        cos,
        sin,
        out_shapes=((64, 16, 32), (64, 4, 32)),
        out_perms=(None, None),
    )

    # 5️⃣ GQA attention
    attn = attention(
        Q,
        K,
        V,
        out_shapes=((64, 16, 32),),
        out_perms=(None,),
    )

    # 6️⃣ O‑projection + residual addition.
    #    The required output shape for this node is (64, 1, 512);
    #    we forward the caller‑provided shape/permutation.
    res = o_proj_residual(
        attn,
        o_proj_weight,
        input_tensor,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return res