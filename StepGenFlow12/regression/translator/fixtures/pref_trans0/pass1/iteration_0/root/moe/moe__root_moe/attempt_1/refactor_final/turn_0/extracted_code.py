# The MOE computation can be delegated entirely to the provided `moe_compute`
# blackbox.  All inputs are already in the form expected by that stub:
#   - `normed_2` is an on‑chip stream and can be passed straight through.
#   - The remaining tensors are RAW (off‑chip).  According to the contract,
#     RAW tensors may be handed directly to a child blackbox, which will
#     perform any required off‑chip loading internally.
# No additional DSL operations are needed here; we simply forward the
# arguments and propagate the `out_shapes` / `out_perms` contract to the child.
def moe__root_moe(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    return moe_compute(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )