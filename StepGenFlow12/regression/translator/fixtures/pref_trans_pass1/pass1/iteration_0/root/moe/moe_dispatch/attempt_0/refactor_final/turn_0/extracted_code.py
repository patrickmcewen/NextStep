# The MoE node first normalises the residual stream with RMS‑Norm, then
# forwards the normalised tensor together with the raw MoE parameters to
# the `moe_dispatch__root_moe_moe_dispatch` child.  Both children expect
# vanilla tensors; the black‑box stubs internally flatten the input
# streams to vanilla shape and re‑tile the outputs according to the
# `out_shapes`/`out_perms` supplied by the parent.  Since the weight
# tensors (`w_gate`, `w_up`, `w_down`, `expert_weights`,
# `expert_onehot`) are marked RAW they can be passed directly to the MoE
# child without an off‑chip load.  The required output stream shape is
# (64, 1, 512), which we forward unchanged to the children.
def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot,
                 *, out_shapes, out_perms=None):
    # Apply RMS‑Norm to the on‑chip residual stream.
    normed_2 = rms_norm(
        res_add_0,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Dispatch through the mixture‑of‑experts layer using the raw expert
    # parameters.  The child returns the final streamed tensor.
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