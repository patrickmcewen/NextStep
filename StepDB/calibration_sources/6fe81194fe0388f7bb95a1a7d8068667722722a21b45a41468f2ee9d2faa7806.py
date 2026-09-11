def tiled_reference(dims, tensors):
    """
    Memory‑efficient MoE forward pass.

    Overview
    --------
    * Tokens are streamed as (1 × D) tiles and replicated over the
      ``n_active`` positions.
    * Expert weight matrices are tiled on the **F** dimension with a static
      chunk size ``tile_f`` (default 256).  The chunk dimension becomes an
      explicit stream dimension after loading with ``offchip_load_ref``.
    * The token stream is expanded to also contain this chunk dimension via
      ``promote`` + ``expand_ref`` so that matmul works without stream‑shape
      mismatches.
    * After the two matmuls we obtain a stream of shape
      ``(..., num_f_chunks)`` with tile ``(1, tile_f)``; the down‑projection
      yields tiles ``(1, D)`` still streamed over ``num_f_chunks``.
    * ``accum_add(rank=1)`` reduces the ``num_f_chunks`` stream dimension,
      producing the per‑expert contribution of shape ``(..., 1, D)``.
    * Contributions are re‑assembled, the ragged‑token and the ``n_active``
      dimensions are collapsed with two ``accum_add`` calls, and the final
      result is stored off‑chip.
    """
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # ------------------------------------------------------------------
    # 1️⃣ Load the token matrix (B × D) as a stream of (1 × D) tiles.
    # ------------------------------------------------------------------
    x = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )

    # ------------------------------------------------------------------
    # 2️⃣ Replicate each token over the ``n_active`` positions.
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
    # 6️⃣ Tiling parameters for the F dimension.
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
        # No flatten – we keep the extra F‑chunk stream dimension.

        # --------------------------------------------------------------
        # c) Expand the token stream to also contain the F‑chunk dimension.
        # --------------------------------------------------------------
        # `promote` adds a trailing singleton stream dim; `expand_ref`
        # replaces it with the corresponding dimension from `gate_i`.
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
        # (again, keep the extra stream dimension)

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
        # (no flatten – keep the extra stream dimension)

        # --------------------------------------------------------------
        # h) Down‑projection → tiles (1 × D), still streamed over F‑chunks.
        # --------------------------------------------------------------
        down_out = binary_matmul(proj, down_i)   # (ragged, n_active, num_f_chunks, 1, D)

        # --------------------------------------------------------------
        # i) Reduce over the F‑chunk stream dimension.
        # --------------------------------------------------------------
        down_out = accum_add(down_out, rank=1)   # (ragged, n_active, 1, D)

        # --------------------------------------------------------------
        # j) Apply per‑token scalar expert weight (broadcasted).
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