# This node simply forwards all inputs to the `moe_dispatch` blackbox, then
# adds the residual connection (`res_add_0`).  The child stub handles any
# required off‑chip loads for the raw tensors, so we do not need to load them
# ourselves.  Both `moe_dispatch` and the residual have the same stream shape
# (64, 1, 512), thus a plain `binary_add` yields the correct final output.
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # MoE dispatch & aggregation (produces a stream of shape out_shapes)
    moe_out = moe_dispatch(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Final residual addition
    return binary_add(moe_out, res_add_0)