# MoE dispatch (root node) implemented with STeP DSL ops.
#   • expert_onehot (int64) is turned into an integer control stream via `select_gen`.
#   • expert_weights (float) is loaded as a 1×1‑tile stream with `offchip_load`.
#   • The normalized token stream is broadcast across the two activation slots using
#     `promote` + `expand_ref`, where the expert_weights stream supplies the target
#     shape.
#   • `expert_addr_gen` converts the one‑hot routing mask into address indices for
#     per‑expert weight tiles; `random_offchip_load` fetches the gate, up and down
#     weight matrices for the selected expert of each token‑slot pair.
#   • The MoE forward path (gate → silu → up → elementwise mul → down) uses
#     `binary_matmul`, `unary_silu` and `binary_mul`.
#   • Per‑slot outputs are weighted by `expert_weights` and summed over the two
#     slots with `accum_add`.
#   • Finally the leading singleton stream dimension is collapsed to the required
#     output shape (seq_len, 1, hidden_dim) using `flatten`, and the result is stored
#     off‑chip.
def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down,
                                         expert_weights, expert_onehot,
                                         *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # 1. Load the integer routing mask (one‑hot) as a control stream.
    # ----------------------------------------------------------------------
    onehot_stream = select_gen(
        expert_onehot,
        is_multihot=False,
        n=expert_onehot.shape[-1],
    )  # shape: (1, seq_len, n_activated_experts, n_routed_experts)

    # ----------------------------------------------------------------------
    # 2. Load the per‑slot scalar expert weights (float) as 1×1 tiles.
    # ----------------------------------------------------------------------
    weight_shape = expert_weights.shape                     # (seq_len, n_activated_experts)
    stride_w = (weight_shape[-1], 1)                       # (cols, 1)
    expert_weights_stream = offchip_load(
        expert_weights,
        stride=stride_w,
        out_shape_tiled=weight_shape,
        tile_row=1,
        tile_col=1,
    )  # shape: (1, seq_len, n_activated_experts, 1, 1)

    # ----------------------------------------------------------------------
    # 3. Broadcast the normalized token stream across the two activation slots.
    #    normed_2: (seq_len, 1, hidden) → (1, seq_len, 1, hidden) → (1, seq_len, 2, 1, hidden)
    # ----------------------------------------------------------------------
    normed_promoted = promote(normed_2, rank=0)            # (1, seq_len, 1, hidden)
    normed_rep = expand_ref(
        normed_promoted,
        expert_weights_stream,
        expand_rank=1,
    )  # shape: (1, seq_len, 2, 1, hidden)

    # ----------------------------------------------------------------------
    # 4. Convert the one‑hot mask into per‑expert tile addresses.
    # ----------------------------------------------------------------------
    addr = expert_addr_gen(onehot_stream, expert_addr_base=0, num_tile_per_expert=1)

    # ----------------------------------------------------------------------
    # 5. Load the three expert weight matrices for the selected expert of each token‑slot.
    # ----------------------------------------------------------------------
    gate_tile = random_offchip_load(
        w_gate,
        addr,
        tile_row=w_gate.shape[1],
        tile_col=w_gate.shape[2],
    )
    up_tile = random_offchip_load(
        w_up,
        addr,
        tile_row=w_up.shape[1],
        tile_col=w_up.shape[2],
    )
    down_tile = random_offchip_load(
        w_down,
        addr,
        tile_row=w_down.shape[1],
        tile_col=w_down.shape[2],
    )

    # ----------------------------------------------------------------------
    # 6. MoE forward pass: gate → silu → up → elementwise mul → down.
    # ----------------------------------------------------------------------
    gate_out = binary_matmul(normed_rep, gate_tile)    # (1, seq_len, 2, 1, inter_dim)
    gate_act = unary_silu(gate_out)
    up_out = binary_matmul(normed_rep, up_tile)        # (1, seq_len, 2, 1, inter_dim)
    hidden = binary_mul(gate_act, up_out)              # (1, seq_len, 2, 1, inter_dim)
    down_out = binary_matmul(hidden, down_tile)        # (1, seq_len, 2, 1, hidden_dim)

    # ----------------------------------------------------------------------
    # 7. Apply scalar expert weights and sum over the two slots.
    # ----------------------------------------------------------------------
    weighted = binary_mul(down_out, expert_weights_stream)  # broadcast over hidden dim
    summed = accum_add(weighted, rank=1)                    # sum over slot dim → (1, seq_len, 1, hidden)

    # ----------------------------------------------------------------------
    # 8. Collapse the leading singleton stream dim to obtain the required output shape.
    # ----------------------------------------------------------------------
    final = flatten(summed, min_rank=0, max_rank=1)  # (seq_len, 1, hidden)

    # ----------------------------------------------------------------------
    # 9. Write the result off‑chip.
    # ----------------------------------------------------------------------
    return offchip_store(final)