# The `moe` node consists of two steps:
# 1. Dispatch the input tokens to the mixture‑of‑experts sub‑module.  
#    This logic lives in the blackbox `moe_dispatch`.  We forward all
#    inputs unchanged (RAW tensors may be passed directly to a blackbox)
#    and propagate the parent‑provided `out_shapes` / `out_perms` so the
#    child returns a stream whose shape matches the parent contract
#    (64, 1, 512).
# 2. Apply the final residual connection that the original PyTorch
#    model performed (`moe_output + res_add_0`).  Both tensors are now
#    tile‑streams with identical stream shape, so we can use the DSL
#    binary addition `binary_add`.
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # MoE routing & expert computation.
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
    return binary_add(res_add_0, moe_out, compute_bw=1)