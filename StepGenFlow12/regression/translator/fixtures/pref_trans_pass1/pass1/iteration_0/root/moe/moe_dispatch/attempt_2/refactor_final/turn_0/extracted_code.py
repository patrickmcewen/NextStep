# The activation `res_add_0` is already a tiled StepTensor (produced by the
# sibling RMS‑norm node).  All weight tensors and routing metadata are RAW:
# they are still off‑chip tensors that can be handed straight to the
# MoE‑dispatch blackbox – the blackbox itself will perform any required
# off‑chip loads (e.g. via `offchip_load` or `select_gen`).  No reshaping,
# promotion, or other DSL ops are needed in this node; we simply forward the
# inputs to the child blackbox, propagating the caller‑provided `out_shapes`
# and `out_perms` arguments.
def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    return moe_dispatch__root_moe_moe_dispatch(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )