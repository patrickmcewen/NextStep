# moe node implementation
#   1. Apply post‑attention RMSNorm via the provided child blackbox.
#   2. Run the MoE core (moe__root_moe) using the normalized tensor and the
#      routing / expert weight tensors (all RAW, the child can load them as needed).
#   3. Add the original residual back with a binary_add DSL operation.
#   The required output stream shape is supplied by the caller via `out_shapes`;
#   we reuse that shape for the intermediate children, since they all produce a
#   tensor of the same logical dimensions (seq_len × dim) tiled as (S,1,T_C).
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # 1) RMSNorm
    normed_2 = post_attn_rms_norm(
        res_add_0,
        out_shapes=(out_shapes[0],),    # stream shape for the normalized tensor
        out_perms=(None,),
    )

    # 2) Mixture‑of‑Experts processing
    moe_out = moe__root_moe(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=(out_shapes[0],),    # same stream shape as the final result
        out_perms=(None,),
    )

    # 3) Residual addition (binary_add is a DSL op)
    final = binary_add(moe_out, res_add_0)

    return final