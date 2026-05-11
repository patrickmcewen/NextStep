# The MoE node simply forwards its inputs to the `moe_dispatch` child,
# requesting the output stream shape defined by the parent's contract
# (a tiled shape (64, 1, 512) in this case).  The child handles loading the
# RAW weight and routing tensors internally, so we do not apply any
# `offchip_load` here.  After obtaining the dispatched tensor we add the
# residual (`res_add_0`) using the DSL `binary_add` operator, which
# preserves the stream metadata and yields the final tensor matching the
# required output shape.
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # Compute the MoE dispatch output with the expected tiling.
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
    # Add the residual connection.
    return binary_add(moe_out, res_add_0)