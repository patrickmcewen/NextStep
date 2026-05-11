# Implementation reasoning:
# The root node simply forwards the raw off‑chip tensors to the three child
# blackboxes – `pre_attention`, `attention`, and `moe`.  Each blackbox expects
# vanilla‑shaped inputs, so no `offchip_load` or other tensor‑method transforms
# are needed before the call.  The only responsibility of this function is to
# specify the desired stream shapes for the children (via `out_shapes`) and to
# write the final result back to memory with `offchip_store`.  We keep the
# sequence length (`seq_len`) from `dims` so the implementation works for any
# static length, and we use stream shapes that respect the required rank ≥ 3:
#   – Q:   (seq_len, 16, 32)
#   – K,V: (seq_len, 4, 32)
#   – attention and MoE outputs: (seq_len, 1, 512)  (one stream dim + a single‑row tile)
def tiled_reference(dims, tensors):
    # Extract raw tensors from the input dictionary.
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

    seq_len = dims["seq_len"]

    # Stage 1: RMSNorm → QKV → per‑head RMSNorm → RoPE
    Q, K, V = pre_attention(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=((seq_len, 16, 32), (seq_len, 4, 32), (seq_len, 4, 32)),
    )

    # Stage 2: GQA attention, O‑projection and first residual add
    res_add_0 = attention(
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
        out_shapes=((seq_len, 1, 512),),
    )

    # Stage 3: Post‑attention RMSNorm, MoE, final residual add
    out = moe(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=((seq_len, 1, 512),),
    )

    # Write the final stream back to off‑chip memory.
    return offchip_store(out)