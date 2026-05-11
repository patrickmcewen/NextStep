# The MoE root node is a thin wrapper around the `moe_compute` child.
# All inputs are either already on‑chip streams (e.g. `normed_2`) or raw
# off‑chip tensors (`w_gate`, `w_up`, `w_down`, `expert_weights`,
# `expert_onehot`).  Raw tensors may be passed directly to a child blackbox;
# the stub will handle any necessary off‑chip loads and reshape its vanilla
# output into the requested tiled shape.  Hence we simply forward the
# arguments together with the `out_shapes` and `out_perms` parameters.
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