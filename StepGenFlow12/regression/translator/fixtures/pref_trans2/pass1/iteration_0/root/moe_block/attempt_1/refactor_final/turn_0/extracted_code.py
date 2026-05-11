# Implementation reasoning:
# The `moe_block` node only needs to compose two child blackboxes:
#   1. `rms_norm` – applies RMS normalization to the residual stream.
#   2. `moe_aggregation` – mixes the normalized representation using the
#      MoE expert weights.
# The inputs `w_gate`, `w_up`, `w_down`, `expert_weights`, and `expert_onehot`
# are RAW off‑chip tensors, but they are only consumed by the `moe_aggregation`
# blackbox, which accepts vanilla‑shape tensors directly.  Therefore we do **not**
# load them with `offchip_load`; we simply forward them to the child.
# Both children produce a single stream tensor, and the caller already
# tells us the required output stream shape via `out_shapes`.  We propagate
# that shape (and any output permutation) unchanged to each child, then
# return the final MoE result.  No tensor‑method transforms are used, satisfying
# the call‑site rules.
def moe_block(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # RMSNorm on the residual stream.
    normed_2 = rms_norm(
        res_add_0,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # MoE aggregation using the normalized representation and raw expert tensors.
    moe_output = moe_aggregation(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return moe_output