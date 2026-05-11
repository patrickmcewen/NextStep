# Implementation reasoning:
# - The root node receives all raw model tensors in the `tensors` dict.
# - Each pipeline stage is delegated to its corresponding blackbox:
#   * `pre_attention` produces the Q, K, V streams.
#   * `attention_o_proj` consumes Q, K, V together with the output projection
#     weight and the original input tensor (as residual) and produces the
#     intermediate residual stream.
#   * `moe` consumes the residual stream along with the MoE parameters and
#     yields the final transformer output.
# - All blackboxes expect *tile‑stream* shapes for their outputs:
#   the trailing two dimensions are tile rows/cols, the leading dimensions
#   form the streaming shape.  Using the vanilla shapes directly satisfies
#   this contract (e.g. Q has vanilla shape (64,16,32) → stream shape
#   (64,16,32) with tile 16×32).
# - No tensor‑method transformations are performed between raw inputs and
#   blackbox calls, satisfying the “no transform” rule.
# - The final stream is written off‑chip via `offchip_store`, which returns a
#   raw torch.Tensor as required for the root.
def tiled_reference(dims, tensors):
    # Raw off‑chip inputs
    input_tensor   = tensors["input_tensor"]
    q_proj         = tensors["q_proj"]
    k_proj         = tensors["k_proj"]
    v_proj         = tensors["v_proj"]
    cos            = tensors["cos"]
    sin            = tensors["sin"]
    o_proj_weight  = tensors["o_proj_weight"]
    w_gate         = tensors["w_gate"]
    w_up           = tensors["w_up"]
    w_down         = tensors["w_down"]
    expert_weights = tensors["expert_weights"]
    expert_onehot  = tensors["expert_onehot"]

    # 1️⃣ Pre‑attention: produce Q, K, V streams.
    Q, K, V = pre_attention(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(
            (64, 16, 32),   # Q: stream dim 64, tile 16×32
            (64,  4, 32),   # K: stream dim 64, tile  4×32
            (64,  4, 32),   # V: stream dim 64, tile  4×32
        ),
        out_perms=(None, None, None),
    )

    # 2️⃣ Attention + O‑projection + residual addition.
    #   Output shape (64, 512) is expressed as stream (64, 1, 512).
    res_add_0 = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
        out_shapes=((64, 1, 512),),
        out_perms=(None,),
    )

    # 3️⃣ MoE (Mixture‑of‑Experts) final layer.
    out = moe(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=((64, 1, 512),),
        out_perms=(None,),
    )

    # Off‑chip write of the final result.
    return offchip_store(out)