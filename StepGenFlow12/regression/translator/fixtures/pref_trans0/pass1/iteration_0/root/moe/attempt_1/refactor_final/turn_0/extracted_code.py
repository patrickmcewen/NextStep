# The `moe` node consists of three logical steps:
#   1. Apply the RMSNorm to the residual tensor (`post_attn_rms_norm`).
#   2. Run the mixture‑of‑experts block on the normalized tensor
#      (`moe__root_moe`).  All weight tensors (`w_gate`, `w_up`, `w_down`,
#      `expert_weights`, `expert_onehot`) are RAW; they can be passed
#      directly to the child blackbox which will load them internally.
#   3. Add the original residual back (final residual connection).
# All shape handling is done via the `out_shapes` contract argument.
# No raw tensor arithmetic or indexing is used – only DSL ops and
# child blackboxes.  The final output conforms to the required shape
# `(64, 1, 512)` (or whatever shape the caller supplies in `out_shapes`).

def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # The caller asks for a single output; extract its stream shape.
    final_shape = out_shapes[0]            # e.g. (64, 1, 512)

    # 1) RMSNorm on the residual.  The child expects a vanilla tensor and
    #    will internally flatten/reshape the stream we give it.  Its output
    #    must match `final_shape`.
    normed_2 = post_attn_rms_norm(
        res_add_0,
        out_shapes=(final_shape,),
        out_perms=(None,),
    )

    # 2) Mixture‑of‑Experts.  RAW weight tensors are forwarded unchanged;
    #    the child blackbox will handle off‑chip loads.  Its output also
    #    follows `final_shape`.
    moe_out = moe__root_moe(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=(final_shape,),
        out_perms=(None,),
    )

    # 3) Final residual addition (binary_add is a DSL compute op).
    final_out = binary_add(moe_out, res_add_0)

    # The contract specifies the output permutation as `None` (identity),
    # so we return the stream directly.
    return final_out