def tiled_reference(dims, tensors):
    # --------------------------------------------------------------
    #  MoE forward pass using the STeP DSL.
    #
    #  1. Load the token matrix `x` (B×D) as a stream: one tile per
    #     token row (tile_row = 1, tile_col = D).
    #  2. Replicate each token across the `n_active` positions (static factor).
    #  3. Load the per‑token scalar weights (`expert_weights`) as a
    #     (1×1) tile stream.
    #  4. Build an Index selector from `expert_onehot` that tells,
    #     for every (token, position) pair, which expert (0‑based) is
    #     selected.
    #  5. Partition the token stream and the scalar‑weight stream per expert
    #     with `flat_partition`.
    #  6. For each expert:
    #        • Broadcast the expert’s gate, up and down weight matrices
    #          to the expert’s token sub‑stream with `offchip_load_ref`.
    #        • `flatten` merges the extra singleton dimension added by
    #          `offchip_load_ref` back into the dynamic token dimension.
    #        • Compute:
    #              gate_out = token @ gate_weight
    #              up_out   = token @ up_weight
    #          Apply SiLU to the gate, multiply element‑wise with `up_out`,
    #          then project back with the down weight matrix.
    #        • Multiply the result by the per‑token scalar weight.
    #  7. Re‑assemble the ragged per‑expert streams to the original
    #     (token, position) order with `flat_reassemble`.
    #  8. Reduce the two extra stream dimensions (the ragged token count
    #     and the `n_active` dimension) with two `accum_add` ops.
    #  9. Store the final (B, D) tensor off‑chip.
    # --------------------------------------------------------------

    # 1️⃣ Load tokens x : (B, D) → stream (1, B) with tile (1, D)
    x = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(dims["B"],),
        tile_row=1,
        tile_col=dims["D"],
    )

    # 2️⃣ Replicate each token over the n_active positions (static factor)
    x_rep = repeat_static(x, factor=dims["n_active"])

    # 3️⃣ Load per‑token scalar expert_weights as (1×1) tiles
    weight = offchip_load(
        tensors["expert_weights"],
        stride=(dims["n_active"], 1),      # row‑major stride for (B, n_active)
        out_shape_tiled=(dims["B"], dims["n_active"]),
        tile_row=1,
        tile_col=1,
    )

    # 4️⃣ Selector from per‑token one‑hot expert IDs (Index, not MultiHot)
    selector = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=dims["n_experts"]
    )

    # 5️⃣ Partition tokens & scalar weights per expert
    token_parts = flat_partition(x_rep, selector, dims["n_experts"])
    weight_parts = flat_partition(weight, selector, dims["n_experts"])

    # 6️⃣ Compute per‑expert contributions
    contributions = []
    for i in range(dims["n_experts"]):
        token_i = token_parts[i]    # Stream of tokens for expert i
        weight_i = weight_parts[i]  # Corresponding scalar weights

        # ----- Gate weight (D → 2048) broadcast to token stream -----
        gate_i = offchip_load_ref(
            token_i,
            tensors["gate_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=tensors["gate_weights"][i].shape[0],   # 1024
            tile_col=tensors["gate_weights"][i].shape[1],   # 2048
        )
        gate_i = flatten(gate_i, min_rank=0, max_rank=1)

        # ----- Up weight (D → 2048) broadcast to token stream -----
        up_i = offchip_load_ref(
            token_i,
            tensors["up_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=tensors["up_weights"][i].shape[0],     # 1024
            tile_col=tensors["up_weights"][i].shape[1],     # 2048
        )
        up_i = flatten(up_i, min_rank=0, max_rank=1)

        # MatMuls producing (1 × 2048) tiles
        gate_out = binary_matmul(token_i, gate_i)   # (dyn, 1, 2048)
        up_out   = binary_matmul(token_i, up_i)     # (dyn, 1, 2048)

        # SiLU on gate and element‑wise multiply with up
        proj = binary_mul(unary_silu(gate_out), up_out)   # (dyn, 1, 2048)

        # ----- Down weight (2048 → D) broadcast -----
        down_i = offchip_load_ref(
            token_i,
            tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=tensors["down_weights"][i].shape[0],   # 2048
            tile_col=tensors["down_weights"][i].shape[1],   # 1024
        )
        down_i = flatten(down_i, min_rank=0, max_rank=1)

        # Final projection back to D‑dimensional space
        down_out = binary_matmul(proj, down_i)   # (dyn, 1, D)

        # Apply per‑token scalar expert weight (broadcasted)
        contrib = binary_mul(down_out, weight_i)  # (dyn, 1, D)
        contributions.append(contrib)

    # 7️⃣ Re‑assemble per‑expert streams to original (token, position) order
    merged = flat_reassemble(contributions, selector)

    # 8️⃣ Collapse the ragged token dimension and the n_active dimension
    merged = accum_add(merged, rank=1)   # remove ragged token dim
    merged = accum_add(merged, rank=1)   # sum over n_active positions

    # 9️⃣ Write the final (B, D) result off‑chip
    return offchip_store(merged)