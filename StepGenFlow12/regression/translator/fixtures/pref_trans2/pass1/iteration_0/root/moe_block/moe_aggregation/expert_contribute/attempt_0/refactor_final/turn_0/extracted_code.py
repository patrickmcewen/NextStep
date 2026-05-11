# The expert contribution consists of three matmul projections, a SILU activation,
# and a final weight‑by‑routing step.  
# All inputs are already on‑chip streams. The expert weight matrices are
# per‑expert (stream shape (1, …)); they must be replicated across the token
# dimension (size taken from `routing_weights`) so that the binary_matmul
# operands have identical stream shapes.  This replication is done with
# `repeat_static`.  Afterwards we perform:
#   gate_out = normed_selected @ w_gate_e
#   up_out   = normed_selected @ w_up_e
#   hidden   = silu(gate_out) * up_out
#   down_out = hidden @ w_down_e
#   weighted = down_out * routing_weights
# All operations are expressed using the DSL compute primitives.
def expert_contribute(normed_selected, w_gate_e, w_up_e, w_down_e, routing_weights, *, out_shapes, out_perms=None):
    # Token count (the variable stream dimension) – taken from any token‑wise
    # input, e.g. routing_weights.  This is a plain Python int.
    token_cnt = routing_weights.shape[1]

    # Replicate each expert weight matrix across the token dimension so that
    # the stream shapes of the matrices match `normed_selected` (and later
    # `hidden`).  `repeat_static` inserts a new stream dimension before the
    # existing ones and expands it to `token_cnt`.
    w_gate_rep = repeat_static(w_gate_e, token_cnt)
    w_up_rep   = repeat_static(w_up_e,   token_cnt)
    w_down_rep = repeat_static(w_down_e, token_cnt)

    # Linear projections for the selected tokens.
    gate_out = binary_matmul(normed_selected, w_gate_rep)
    up_out   = binary_matmul(normed_selected, w_up_rep)

    # SILU activation on gate output and element‑wise multiply with up output.
    hidden = binary_mul(unary_silu(gate_out), up_out)

    # Down projection back to hidden dimension.
    down_out = binary_matmul(hidden, w_down_rep)

    # Weight by routing scores (broadcasted across the hidden dimension).
    weighted = binary_mul(down_out, routing_weights)

    return weighted