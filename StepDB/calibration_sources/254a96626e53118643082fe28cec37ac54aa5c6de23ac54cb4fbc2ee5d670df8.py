def tiled_reference(dims, tensors):
    """
    Faster MoE forward pass (still ≤ 25 276 420 B on‑chip).

    Main change vs. the previously accepted variant:
      * After the SiLU‑gated up projection we replace the separate
        `binary_matmul` + `accum_add` that performed the down‑projection
        with a single `binary_map_accum(..., rank=1)`.  This fuses the
        matrix‑multiply and the reduction over the F‑chunks into one
        operator, cutting the per‑expert compute latency by roughly one
        matmul + one accumulation.
      * All other structure (tiling on the F‑dimension, expansion of the
        token stream, per‑expert `offchip_load_ref`, etc.) is unchanged,
        so the on‑chip memory usage stays within the budget (≈ 25.25 MiB).
    """
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # ------------------------------------------------------------------
    # 1️⃣ Load tokens (B × D) as a stream of (1 × D) tiles.
    # ------------------------------------------------------------------
    x = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )

    # ------------------------------------------------------------------
    # 2️⃣ Replicate each token over the n_active positions.
    # ------------------------------------------------------------------
    x_rep = repeat_static(x, factor=n_active)          # stream (B, n_active), tile (1, D)

    # ------------------------------------------------------------------
    # 3️⃣ Load per‑token scalar expert_weights (1 × 1) tiles.
    # ------------------------------------------------------------------
    weight = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),                # row‑major stride for (B, n_active)
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )

    # ------------------------------------------------------------------
    # 4️⃣ Selector that maps each (token, position) to an expert.
    # ------------------------------------------------------------------
    selector = select_gen(
        tensors["expert_onehot"],
        is_multihot=False,
        n=n_experts,
    )

    # ------------------------------------------------------------------
    # 5️⃣ Partition tokens and scalar weights per expert.
    # ------------------------------------------------------------------
    token_parts = flat_partition(x_rep, selector, n_experts)
    weight_parts = flat_partition(weight, selector, n_experts)

    # ------------------------------------------------------------------
    # 6️⃣ Tile configuration for the F dimension.
    # ------------------------------------------------------------------
    tile_f = dims["tile_f"]                      # 256 (default)
    num_f_chunks = F // tile_f                    # 8 for F=2048

    contributions = []
    for i in range(n_experts):
        # --------------------------------------------------------------
        # a) Token sub‑stream for this expert and its scalar weight.
        # --------------------------------------------------------------
        token_i = token_parts[i]          # stream (ragged, n_active), tile (1, D)
        weight_i = weight_parts[i]        # same stream shape, tile (1, 1)

        # --------------------------------------------------------------
        # b) Load gate weight (D → F) tiled on the F dimension.
        # --------------------------------------------------------------
        gate_i = offchip_load_ref(
            token_i,
            tensors["gate_weights"][i],
            stride=(1,),
            out_shape_tiled=(num_f_chunks,),
            tile_row=D,
            tile_col=tile_f,
        )

        # --------------------------------------------------------------
        # c) Expand the token stream to also contain the F‑chunk dimension.
        # --------------------------------------------------------------
        token_i_exp = expand_ref(promote(token_i, rank=0), gate_i, expand_rank=1)

        # --------------------------------------------------------------
        # d) Load up weight (D → F) tiled identically.
        # --------------------------------------------------------------
        up_i = offchip_load_ref(
            token_i,
            tensors["up_weights"][i],
            stride=(1,),
            out_shape_tiled=(num_f_chunks,),
            tile_row=D,
            tile_col=tile_f,
        )

        # --------------------------------------------------------------
        # e) Gate and up matmuls → tiles (1 × tile_f), streamed over F‑chunks.
        # --------------------------------------------------------------
        gate_out = binary_matmul(token_i_exp, gate_i)   # (ragged, n_active, num_f_chunks, 1, tile_f)
        up_out   = binary_matmul(token_i_exp, up_i)     # same shape

        # --------------------------------------------------------------
        # f) SiLU on gate and element‑wise multiply with up.
        # --------------------------------------------------------------
        proj = binary_mul(unary_silu(gate_out), up_out)   # (ragged, n_active, num_f_chunks, 1, tile_f)

        # --------------------------------------------------------------
        # g) Load down weight (F → D) tiled on the F dimension.
        # --------------------------------------------------------------
        down_i = offchip_load_ref(
            token_i,
            tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(num_f_chunks,),
            tile_row=tile_f,
            tile_col=D,
        )

        # --------------------------------------------------------------
        # h) Down‑projection + reduction over the F‑chunks in ONE op.
        # --------------------------------------------------------------
        # `binary_map_accum` performs the matmul (proj @ down_i) and then
        # reduces the innermost stream dim (the F‑chunk dim) by summing.
        down_out = binary_map_accum(proj, down_i, rank=1)

        # --------------------------------------------------------------
        # i) Apply per‑token scalar expert weight (broadcasted).
        # --------------------------------------------------------------
        contrib = binary_mul(down_out, weight_i)   # (ragged, n_active, 1, D)

        contributions.append(contrib)

    # ------------------------------------------------------------------
    # 7️⃣ Re‑assemble per‑expert streams to the original (token, position) order.
    # ------------------------------------------------------------------
    merged = flat_reassemble(contributions, selector)

    # ------------------------------------------------------------------
    # 8️⃣ Collapse the ragged token dimension and the n_active dimension.
    # ------------------------------------------------------------------
    merged = accum_add(merged, rank=1)   # ragged token dim → summed
    merged = accum_add(merged, rank=1)   # n_active dim → summed

    # ------------------------------------------------------------------
    # 9️⃣ Store the final (B, D) result off‑chip.
    # ------------------------------------------------------------------
    return offchip_store(merged)