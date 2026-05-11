# The expert_contribute node takes a stream of token embeddings (normed_selected) and
# expert‑specific weight matrices that are already on‑chip.  The weight tensors have a
# single stream dimension (batch = 1) while the token stream has a second stream dimension
# for the tokens (size = normed_selected.shape[1]).  We broadcast each weight matrix across
# that token dimension using `repeat_static`, then perform the expert's linear projections
# with `binary_matmul`.  A SILU activation is applied to the gate projection and multiplied
# element‑wise with the up‑projection (`binary_mul`).  The result is down‑projected, and
# finally weighted by the per‑token routing scores.  All tensor arithmetic is expressed
# via the DSL primitives; only a scalar Python int (`token_dim`) is extracted from the
# stream shape to drive the repeat factor.
def expert_contribute(normed_selected, w_gate_e, w_up_e, w_down_e, routing_weights, *, out_shapes, out_perms=None):
    # Number of tokens in the stream (int) – used to broadcast the expert weights.
    token_dim = normed_selected.shape[1]

    # Broadcast expert weight matrices across the token stream dimension.
    w_gate_exp = repeat_static(w_gate_e, token_dim)
    w_up_exp   = repeat_static(w_up_e, token_dim)
    w_down_exp = repeat_static(w_down_e, token_dim)

    # Linear projections for the selected tokens.
    gate_out = binary_matmul(normed_selected, w_gate_exp)   # [*, token, 1, moe_inter_dim]
    up_out   = binary_matmul(normed_selected, w_up_exp)     # same shape as gate_out

    # SILU activation on gate output, then element‑wise multiply with up projection.
    hidden = binary_mul(unary_silu(gate_out), up_out)

    # Down projection back to hidden dimension.
    down_out = binary_matmul(hidden, w_down_exp)

    # Weight by routing scores (broadcasted over the hidden dimension).
    weighted = binary_mul(down_out, routing_weights)

    return weighted