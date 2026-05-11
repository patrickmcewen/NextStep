# Implementation reasoning:
# The node consists of two sequential blackbox calls:
#   1. RMSNorm on the on‑chip stream `res_add_0` producing `normed_2`.
#   2. MoE dispatch that consumes `normed_2` together with the RAW weight tensors.
# No DSL operations are needed between the calls because the blackboxes handle
# all necessary shape transformations (they flatten/reshape the incoming streams
# to vanilla shapes internally). The RAW tensors are passed directly to the MoE
# child without an off‑chip load, which the child stub will perform if required.
# The output shape requested by the parent is `(64, 1, 512)`; we forward this
# shape (and any permutation) to both children so that the final stream has the
# correct tile‑stream layout.
def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # Apply RMSNorm to the residual addition stream.
    normed_2 = rms_norm(
        res_add_0,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Perform the Mixture‑of‑Experts dispatch using the normalized tensor and
    # the (still off‑chip) weight tensors.
    result = moe_dispatch__root_moe_moe_dispatch(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return result