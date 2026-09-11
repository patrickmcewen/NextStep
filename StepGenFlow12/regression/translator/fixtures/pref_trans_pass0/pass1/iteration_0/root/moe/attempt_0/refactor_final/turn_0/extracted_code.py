# The MoE node simply delegates to the pre‑implemented child `moe_dispatch`.
# All inputs are provided as raw or already‑streamed tensors, and the child
# stub handles any necessary off‑chip loads internally.  We just forward the
# arguments and propagate the caller‑specified `out_shapes`/`out_perms`,
# which request a tiled output of shape (64, 1, 512) as required.
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    return moe_dispatch(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )