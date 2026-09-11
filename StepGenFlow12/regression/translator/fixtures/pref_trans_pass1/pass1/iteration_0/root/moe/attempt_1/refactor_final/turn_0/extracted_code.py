# Compute the MoE dispatch and add the residual connection.
# - `res_add_0` is already a stream tensor (on‑chip).
# - Weight and routing tensors are RAW; we can pass them directly to the
#   child blackbox, which will perform any necessary off‑chip loads.
# - The child `moe_dispatch` returns a stream with the shape requested in
#   `out_shapes`; we forward the node's own `out_shapes` and `out_perms`
#   so its output matches the required (64, 1, 512) stream.
# - Finally we add the residual using `binary_add`, which requires both
#   operands to have identical stream shapes and tile dimensions.
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # Invoke the child that implements the full MoE dispatch.
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
    # Residual addition (stream‑wise).
    return binary_add(moe_out, res_add_0)