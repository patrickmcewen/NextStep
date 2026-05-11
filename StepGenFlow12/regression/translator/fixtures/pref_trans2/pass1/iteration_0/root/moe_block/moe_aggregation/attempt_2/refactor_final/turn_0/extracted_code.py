# The MoE aggregation is expressed as a stream‑based pipeline.
#   * expert_weights (float) and expert_onehot (int) are brought on‑chip:
#       - expert_weights via offchip_load (tile 1×1) → shape (1,seq,top,1,1)
#       - expert_onehot via select_gen (no tiling needed) → shape (1,seq,top,experts)
#   * normed_2 (already on‑chip) is repeated over the top‑position dimension
#     with repeat_ref so it aligns with the shape of expert_weights.
#   * flat_partition uses the one‑hot mask to split both the repeated token
#     stream and the routing‑weight stream into per‑expert sub‑streams.
#   * For each expert we lazily load its three weight matrices (gate, up,
#     down) with offchip_load, selecting the single tile that corresponds to
#     the expert index (stride=(e,)).
#   * The child blackbox `expert_contribute` computes the expert’s
#     contribution for the (possibly empty) token sub‑stream.  When the sub‑stream
#     is empty we emit a zero‑row tensor via torch.zeros, which is allowed.
#   * flat_reassemble restores the per‑expert contributions to the original
#     (seq, top) layout using the same one‑hot control tensor.
#   * Finally accum_add with rank=1 collapses the top‑position dimension,
#     yielding the required output stream shape (1, seq_len, 1, hidden_dim).
#
# All tensor shape manipulations use DSL primitives only; no raw Tensor
# methods (reshape, index, arithmetic, etc.) appear between producers and
# consumers.  The implementation respects RAW/on‑chip annotations and
# conforms to the contract’s required output shape.

def moe_aggregation(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # 1) Load the routing metadata (expert_weights) as a 1×1 tiled stream.
    seq_len, top_k = expert_weights.shape  # (seq_len, n_activated_experts)
    exp_weights_stream = offchip_load(
        expert_weights,
        stride=(1, 1),
        out_shape_tiled=(seq_len, top_k),
        tile_row=1,
        tile_col=1,
    )  # shape (1, seq_len, top_k, 1, 1)

    # 2) Convert the integer one‑hot selector to a stream control tensor.
    n_experts = w_gate.shape[0]  # number of routed experts
    exp_onehot_ctrl = select_gen(
        expert_onehot,
        is_multihot=True,
        n=n_experts,
    )  # shape (1, seq_len, top_k, n_experts)

    # 3) Broadcast normed token representations over the top‑position dimension.
    #    normed_2 already has shape (1, seq_len, 1, hidden_dim).
    #    repeat_ref adds the extra top_k stream dimension.
    normed_rep = repeat_ref(normed_2, exp_weights_stream)  # shape (1, seq_len, top_k, 1, hidden_dim)

    # 4) Partition token and routing‑weight streams per expert using the one‑hot mask.
    normed_per_expert = flat_partition(normed_rep, exp_onehot_ctrl, n=n_experts)
    weights_per_expert = flat_partition(exp_weights_stream, exp_onehot_ctrl, n=n_experts)

    # 5) Loop over experts, loading the three weight matrices for the current expert
    #    and invoking the expert_contribute blackbox.
    hid_dim, inter_dim = w_gate.shape[1], w_gate.shape[2]  # 512, 1792
    contribs = []
    for e_idx in range(n_experts):
        # Load the weight tiles for this expert (single tile per matrix).
        w_gate_e = offchip_load(
            w_gate,
            stride=(e_idx,),
            out_shape_tiled=(1,),
            tile_row=hid_dim,
            tile_col=inter_dim,
        )  # (1, 1, hid_dim, inter_dim)
        w_up_e = offchip_load(
            w_up,
            stride=(e_idx,),
            out_shape_tiled=(1,),
            tile_row=hid_dim,
            tile_col=inter_dim,
        )  # (1, 1, hid_dim, inter_dim)
        w_down_e = offchip_load(
            w_down,
            stride=(e_idx,),
            out_shape_tiled=(1,),
            tile_row=inter_dim,
            tile_col=hid_dim,
        )  # (1, 1, inter_dim, hid_dim)

        # Retrieve the per‑expert token slice and routing‑weight slice.
        normed_selected = normed_per_expert[e_idx]      # (N_e, 1, hidden_dim)
        routing_weights = weights_per_expert[e_idx]    # (N_e, 1, 1)

        # If no tokens were routed to this expert, emit an empty contribution.
        n_selected = normed_selected.shape[0]
        if n_selected == 0:
            contrib = torch.zeros((0, 1, hid_dim), dtype=normed_selected.dtype)
        else:
            # child expects a stream shape (N, 1, hidden_dim)
            out_shape_child = (n_selected, 1, hid_dim)
            contrib = expert_contribute(
                normed_selected,
                w_gate_e,
                w_up_e,
                w_down_e,
                routing_weights,
                out_shapes=(out_shape_child,),
                out_perms=(None,),
            )
        contribs.append(contrib)

    # 6) Reassemble the per‑expert contributions back to (seq, top) layout.
    merged = flat_reassemble(contribs, exp_onehot_ctrl)   # shape (1, seq_len, top_k, 1, 1, hidden_dim)

    # 7) Collapse the top‑position dimension (rank=1) to obtain the final output.
    output = accum_add(merged, rank=1)  # shape (1, seq_len, 1, hidden_dim)

    return output