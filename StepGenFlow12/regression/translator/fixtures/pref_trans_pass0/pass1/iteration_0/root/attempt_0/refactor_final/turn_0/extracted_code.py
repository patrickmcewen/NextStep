# Root node orchestrates the three sub‑models.
# All raw inputs are forwarded unchanged to the blackboxes.
# The blackboxes are asked to emit stream tensors whose tile dimensions
# match the vanilla shapes they would have in the original PyTorch model.
# For the attention output and the final MoE output we add a singleton
# stream dimension so they are valid 3‑D stream tensors before the final
# `offchip_store` flattens them back to the vanilla (64, 512) shape.
def tiled_reference(dims, tensors):
    # 1️⃣ Pre‑attention: produce Q, K, V streams.
    Q, K, V = pre_attention(
        tensors["input_tensor"],
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        out_shapes=((64, 16, 32), (64, 4, 32), (64, 4, 32)),
        out_perms=(None, None, None),
    )

    # 2️⃣ Attention + O‑projection + first residual add.
    res_add_0 = attention_o_proj(
        Q,
        K,
        V,
        tensors["o_proj_weight"],
        tensors["input_tensor"],
        out_shapes=((1, 64, 512),),
        out_perms=(None,),
    )

    # 3️⃣ MoE (RMSNorm + MoE + final residual add).
    out = moe(
        res_add_0,
        tensors["w_gate"],
        tensors["w_up"],
        tensors["w_down"],
        tensors["expert_weights"],
        tensors["expert_onehot"],
        out_shapes=((1, 64, 512),),
        out_perms=(None,),
    )

    # Write the final result off‑chip.
    return offchip_store(out)