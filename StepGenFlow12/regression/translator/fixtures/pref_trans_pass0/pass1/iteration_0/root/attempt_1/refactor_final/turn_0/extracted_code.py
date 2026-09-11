# Implementation reasoning:
# The root node simply wires together the three child blackboxes that implement the
# pre‑attention, attention + O‑projection, and MoE stages.  All inputs arriving at
# this node are raw (off‑chip) tensors; the contract allows them to be passed
# directly to a child blackbox without any transformation.  The children expect
# vanilla tensors and internally recover the shape from the provided stream, so
# we only need to specify the desired *stream* shapes for their outputs via the
# `out_shapes` argument.
#
# For simplicity we use a tile size of (1, 1) for every tensor.  This yields a
# stream shape that exactly mirrors the vanilla shape, with two trailing tile
# dimensions of size 1.  All `out_shapes` entries therefore have rank ≥ 3, meeting
# the DSL requirement.  No additional DSL operations are needed between the
# children because each child returns a stream that can be fed directly into the
# next one.
#
# Finally, the root must write the result off‑chip, which we do with
# `offchip_store`.  The resulting tensor has vanilla shape (64, 512) as required.
def tiled_reference(dims, tensors):
    # Raw inputs (off‑chip tensors)
    input_tensor = tensors["input_tensor"]
    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    o_proj_weight = tensors["o_proj_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    expert_weights = tensors["expert_weights"]
    expert_onehot = tensors["expert_onehot"]

    # 1️⃣ Pre‑attention: produce Q, K, V streams.
    Q, K, V = pre_attention(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(
            (64, 16, 32, 1, 1),   # Q: (seq_len, num_q_heads, head_dim, tile_r, tile_c)
            (64, 4, 32, 1, 1),    # K: (seq_len, num_kv_heads, head_dim, 1, 1)
            (64, 4, 32, 1, 1),    # V: (seq_len, num_kv_heads, head_dim, 1, 1)
        ),
    )

    # 2️⃣ Attention + O‑projection (produces the first residual addition stream)
    res_add_0 = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
        out_shapes=(
            (64, 512, 1, 1),      # output of attention + O‑proj (vanilla 64×512, tiled 1×1)
        ),
    )

    # 3️⃣ MoE (final transformer block output)
    out = moe(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=(
            (64, 512, 1, 1),      # final output tensor (vanilla 64×512, tiled 1×1)
        ),
    )

    # Write the result off‑chip.
    return offchip_store(out)