# The MoE aggregation can be expressed as a streaming pipeline.
# 1. Turn the integer one‑hot routing tensor into a DSL control stream with `select_gen`.
# 2. Load the floating‑point routing weights and expand the token stream (`normed_2`)
#    so that each token appears once per activated‑expert position (the model has 2 positions).
# 3. Partition both the expanded token stream and the weight stream per expert
#    using `flat_partition` driven by the one‑hot mask.
# 4. For each expert, slice the raw expert weight matrices (still off‑chip),
#    load the slice with `offchip_load`, and invoke the black‑box `expert_contribute`
#    on the per‑expert token and routing‑weight streams.
# 5. Re‑assemble the per‑expert contributions back into the original token order
#    with `flat_reassemble`, then collapse the extra dimensions with two
#    `accum_add` passes (first over the “active‑expert” dimension, then over the
#    two top‑position slots).
# 6. Store the final token stream off‑chip.

def moe_aggregation(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # 1. Control mask – canonical DSL stream (int64)
    onehot_ctrl = select_gen(
        expert_onehot,
        is_multihot=False,
        n=expert_onehot.shape[-1],  # n_routed_experts = 8
    )  # shape: (1, seq_len, n_activated_experts, n_routed_experts)

    # 2. Load routing weights (float) as a stream and expand the token stream
    #    so that each token is duplicated for the two activated‑expert positions.
    w_weights = offchip_load(
        expert_weights,
        stride=(1, 1),
        out_shape_tiled=(expert_weights.shape[0], expert_weights.shape[1]),  # (64, 2)
        tile_row=1,
        tile_col=1,
    )  # shape: (1, 64, 2, 1, 1)

    # Expand normed_2 to match the (seq_len, n_activated_experts) stream shape.
    normed_rep = repeat_ref(normed_2, ref=w_weights)   # shape: (1, 64, 2, 1, 512)

    # 3. Partition both streams per expert using the one‑hot mask.
    n_experts = expert_onehot.shape[-1]
    normed_parts = flat_partition(normed_rep, onehot_ctrl, n=n_experts)   # list of 8 tensors
    weight_parts = flat_partition(w_weights, onehot_ctrl, n=n_experts)    # list of 8 tensors

    # 4. Per‑expert contribution via the black‑box helper.
    contribs = []
    for e_idx in range(n_experts):
        # Slice the raw weight matrices for this expert (still off‑chip).
        w_gate_e_raw = w_gate[e_idx]      # shape (512, 1792)
        w_up_e_raw = w_up[e_idx]          # shape (512, 1792)
        w_down_e_raw = w_down[e_idx]      # shape (1792, 512)

        # Load each slice onto‑chip as a stream with a single element.
        w_gate_e = offchip_load(
            w_gate_e_raw,
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=512,
            tile_col=1792,
        )  # shape: (1, 1, 512, 1792)

        w_up_e = offchip_load(
            w_up_e_raw,
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=512,
            tile_col=1792,
        )  # shape: (1, 1, 512, 1792)

        w_down_e = offchip_load(
            w_down_e_raw,
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=1792,
            tile_col=512,
        )  # shape: (1, 1, 1792, 512)

        # Streams selected for this expert.
        normed_sel = normed_parts[e_idx]   # (k_e, 1, 512)
        weight_sel = weight_parts[e_idx]   # (k_e, 1, 1)

        # Call the expert‑contribute black‑box.
        contrib = expert_contribute(
            normed_sel,
            w_gate_e,
            w_up_e,
            w_down_e,
            weight_sel,
            out_shapes=(normed_sel.shape,),
            out_perms=(None,),
        )
        contribs.append(contrib)

    # 5. Re‑assemble contributions to the original token order.
    moe_out = flat_reassemble(contribs, onehot_ctrl)   # (1, 64, 2, N_active, 1, 512)

    # Collapse the active‑expert dimension and the top‑position dimension.
    moe_out = accum_add(moe_out, rank=1)   # sum over N_active -> (1, 64, 2, 1, 512)
    moe_out = accum_add(moe_out, rank=1)   # sum over the 2 positions -> (1, 64, 1, 512)

    # 6. Write the result off‑chip.
    return offchip_store(moe_out)