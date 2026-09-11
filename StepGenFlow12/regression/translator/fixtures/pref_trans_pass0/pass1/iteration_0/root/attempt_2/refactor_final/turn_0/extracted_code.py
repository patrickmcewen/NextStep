# Implementation reasoning:
# - The root node simply orchestrates three blackbox sub‑models: pre_attention,
#   attention_o_proj, and moe.  All raw inputs are passed directly to these
#   blackboxes; they handle any off‑chip loading internally.
# - Each blackbox requires an `out_shapes` specification describing the
#   desired tile‑stream shape of its outputs.  For Q, K, V we keep the natural
#   (seq_len, num_heads, head_dim) layout, which already satisfies the
#   rank‑≥‑3 requirement (stream dim + two tile dims).
# - The attention output and the final MoE output have vanilla shape (64, 512).
#   To satisfy the stream‑rank constraint we expose them as a stream of 64
#   tokens with tile shape (1, 512); i.e. stream shape (64, 1, 512).  This way
#   `offchip_store` will flatten the leading stream dimension into rows,
#   yielding the expected off‑chip shape (64, 512).
# - The final result is written off‑chip via `offchip_store`.
def tiled_reference(dims, tensors):
    # Unpack raw tensors provided by the caller.
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
            (64, 16, 32),   # Q: stream=64, tile=(16,32)
            (64, 4, 32),    # K: stream=64, tile=(4,32)
            (64, 4, 32),    # V: stream=64, tile=(4,32)
        ),
        out_perms=(None, None, None),
    )

    # 2️⃣ Attention + O‑projection + residual addition.
    res_add_0 = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
        out_shapes=((64, 1, 512),),  # stream=64, tile=(1,512)
        out_perms=(None,),
    )

    # 3️⃣ MoE block.
    out = moe(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=((64, 1, 512),),  # stream=64, tile=(1,512)
        out_perms=(None,),
    )

    # Write the final result off‑chip.  `offchip_store` will reshape the
    # (64,1,512) stream into the expected vanilla shape (64,512).
    return offchip_store(out)