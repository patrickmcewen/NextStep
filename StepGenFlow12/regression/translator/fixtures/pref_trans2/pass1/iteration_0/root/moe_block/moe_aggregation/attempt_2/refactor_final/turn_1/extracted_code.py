# MoE aggregation expressed entirely with DSL primitives.
#   * Off‑chip tensors (weights, routing data) are loaded with `offchip_load`.
#   * The integer one‑hot selector is turned into a stream with `select_gen`.
#   * Token representations are broadcast across the top‑k dimension using
#     `repeat_ref` (the one‑hot stream provides the extra dimension).
#   * `flat_partition` splits the token and routing‑weight streams into per‑expert
#     sub‑streams according to the one‑hot mask.
#   * For each expert we load its three weight matrices, expand them to the
#     token‑stream shape with `expand_ref`, and compute the expert contribution
#     inline (gate·up·silu → down → scale by routing weight) using
#     `binary_matmul`, `binary_mul`, and `unary_silu`.
#   * Empty expert streams (no tokens assigned) are handled by emitting a zero‑size
#     tensor via `torch.zeros` (allowed by the DSL rules).
#   * The per‑expert contributions are reassembled into the original (seq, top)
#     layout with `flat_reassemble`, then the top‑k dimension (and the intermediate
#     singleton) are collapsed by `accum_add(rank=2)`, yielding the required
#     output stream shape `(1, seq_len, 1, hidden_dim)`.
#
# All tensor shape manipulations go through DSL ops; no raw PyTorch tensor
# methods appear between producers and consumers.

def moe_aggregation(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # 1) Load routing‑weight tensor (seq_len × top_k) as a 1×1 tiled stream.
    # -----------------------------------------------------------------
    seq_len, top_k = expert_weights.shape
    routing_weight_stream = offchip_load(
        expert_weights,
        stride=(1, 1),
        out_shape_tiled=(seq_len, top_k),
        tile_row=1,
        tile_col=1,
    )  # shape: (1, seq_len, top_k, 1, 1)

    # -----------------------------------------------------------------
    # 2) Load the integer one‑hot selector (seq_len × top_k × n_experts).
    # -----------------------------------------------------------------
    n_experts = w_gate.shape[0]
    expert_onehot_ctrl = select_gen(
        expert_onehot,
        is_multihot=True,
        n=n_experts,
    )  # shape: (1, seq_len)×tile(2, n_experts)

    # -----------------------------------------------------------------
    # 3) Broadcast the token stream over the top‑k dimension.
    #    `repeat_ref` adds the extra stream dim (size = top_k) using the
    #    one‑hot control as the reference.
    # -----------------------------------------------------------------
    normed_rep = repeat_ref(normed_2, expert_onehot_ctrl)  # (1, seq_len, top_k, 1, hidden)

    # -----------------------------------------------------------------
    # 4) Partition tokens and routing‑weights per expert according to the mask.
    # -----------------------------------------------------------------
    token_streams = flat_partition(normed_rep, expert_onehot_ctrl, n=n_experts)      # list[ (N_e,1,hidden) ]
    weight_streams = flat_partition(routing_weight_stream, expert_onehot_ctrl, n=n_experts)  # list[ (N_e,1,1) ]

    # -----------------------------------------------------------------
    # 5) Compute each expert's contribution (inline Mixtral FFN logic).
    # -----------------------------------------------------------------
    hid_dim = w_gate.shape[1]    # 512
    inter_dim = w_gate.shape[2]  # 1792

    contributions = []
    for e_idx in range(n_experts):
        tokens_e = token_streams[e_idx]      # (N, 1, hidden)
        rout_e   = weight_streams[e_idx]      # (N, 1, 1)

        n_selected = tokens_e.shape[0]
        if n_selected == 0:
            # No tokens routed to this expert – produce an empty contribution.
            contributions.append(
                torch.zeros((0, 1, hid_dim), dtype=tokens_e.dtype)
            )
            continue

        # Load the three weight matrices for this expert (single tile each).
        w_gate_e = offchip_load(
            w_gate,
            stride=(e_idx,),
            out_shape_tiled=(1,),
            tile_row=hid_dim,
            tile_col=inter_dim,
        )  # (1, 1, hid, inter)
        w_up_e = offchip_load(
            w_up,
            stride=(e_idx,),
            out_shape_tiled=(1,),
            tile_row=hid_dim,
            tile_col=inter_dim,
        )
        w_down_e = offchip_load(
            w_down,
            stride=(e_idx,),
            out_shape_tiled=(1,),
            tile_row=inter_dim,
            tile_col=hid_dim,
        )

        # Broadcast weights to the token stream shape.
        w_gate_exp = expand_ref(w_gate_e, tokens_e, expand_rank=2)
        w_up_exp   = expand_ref(w_up_e,   tokens_e, expand_rank=2)
        w_down_exp = expand_ref(w_down_e, tokens_e, expand_rank=2)

        # Gate and up projections.
        gate = binary_matmul(tokens_e, w_gate_exp)   # (N, 1, inter)
        up   = binary_matmul(tokens_e, w_up_exp)    # (N, 1, inter)

        # SiLU(gate) * up
        gate_act = unary_silu(gate)
        act = binary_mul(gate_act, up)              # (N, 1, inter)

        # Down projection.
        down = binary_matmul(act, w_down_exp)       # (N, 1, hidden)

        # Scale by routing weight.
        contrib = binary_mul(down, rout_e)           # (N, 1, hidden)

        contributions.append(contrib)

    # -----------------------------------------------------------------
    # 6) Reassemble per‑expert contributions back to (seq_len, top_k) layout.
    # -----------------------------------------------------------------
    merged = flat_reassemble(contributions, expert_onehot_ctrl)  # (1, seq_len, top_k, 1, 1, hidden)

    # -----------------------------------------------------------------
    # 7) Collapse the top‑k dimension (and the singleton n_active) to produce
    #    the final output stream of shape (1, seq_len, 1, hidden).
    # -----------------------------------------------------------------
    output = accum_add(merged, rank=2)  # (1, seq_len, 1, hidden)

    return output