# moe_block: RMS‑norm the residual and then run MoE aggregation.
# Both RMSNorm and MoEAggregation are provided as blackboxes.
# The inputs are already streams (or raw tensors that the blackboxes can
# handle internally).  We request the children to emit their results in the
# same tiled shape required by the parent: (1, 64, 1, 512).  No explicit DSL
# operators are needed because the blackboxes perform any necessary off‑chip
# loads and shape handling internally.
def moe_block(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # 1️⃣ RMSNorm on the residual stream.
    normed_2 = rms_norm(
        res_add_0,
        out_shapes=((1, 64, 1, 512),),   # produce a stream matching the parent's contract
        out_perms=(None,),
    )

    # 2️⃣ MoE aggregation using the normalized representation and the expert weights.
    moe_out = moe_aggregation(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=((1, 64, 1, 512),),   # final output stream shape
        out_perms=(None,),
    )

    return moe_out