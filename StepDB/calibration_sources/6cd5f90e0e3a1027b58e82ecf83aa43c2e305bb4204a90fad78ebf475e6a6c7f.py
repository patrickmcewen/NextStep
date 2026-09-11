def tiled_reference(dims, tensors):
    """
    Faster MoE forward pass (still ≤ 25 276 420 B on‑chip).

    Changes vs. the previous variant:
      * The extra `flatten` after each `offchip_load_ref` (gate, up, down)
        is removed.  The reference load already provides the needed stream
        shape, so we can broadcast the token stream onto it directly with
        `expand_ref`.  This eliminates several cheap but non‑zero‑cost ops,
        shaving a few hundred cycles.
      * All other structure (tiling on the F dimension, `binary_map_accum`
        for the down‑projection, two separate `accum_add` reductions after
        `flat_reassemble`) remains unchanged, preserving functional
        equivalence and on‑chip memory usage (≈ 25.25 MiB).
    """
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]
    tile_f = dims["tile_f"]                # default 256
    num_f_chunks = F // tile_f              # e.g. 2048 // 256 = 8

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

    contributions = []
    for i in range(n_experts):
        # --------------------------------------------------------------
        # a) Token sub‑stream for this expert and its scalar weight.
        # --------------------------------------------------------------
        token_i = token_parts[i]          # stream (ragged,), tile (1, D)
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
        # No flatten – keep stream shape (ragged, num_f_chunks)

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
        # No flatten – same stream shape as gate_i

        # --------------------------------------------------------------
        # e) Gate and up matmuls → tiles (1 × tile_f), streamed over F‑chunks.
        # --------------------------------------------------------------
        gate_out = binary_matmul(token_i_exp, gate_i)   # (ragged, num_f_chunks, 1, tile_f)
        up_out   = binary_matmul(token_i_exp, up_i)     # same shape

        # --------------------------------------------------------------
        # f) SiLU on gate and element‑wise multiply with up.
        # --------------------------------------------------------------
        proj = binary_mul(unary_silu(gate_out), up_out)   # (ragged, num_f_chunks, 1, tile_f)

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
        # No flatten – stream shape matches `proj`.

        # --------------------------------------------------------------
        # h) Down‑projection + reduction over the F‑chunks in ONE op.
        # --------------------------------------------------------------
        down_out = binary_map_accum(proj, down_i, rank=1)

        # --------------------------------------------------------------
        # i) Apply per‑token scalar expert weight (broadcasted).
        # --------------------------------------------------------------
        contrib = binary_mul(down_out, weight_i)   # (ragged, 1, D)
        contributions.append(contrib)

    # ------------------------------------------------------------------
    # 7️⃣ Re‑assemble per‑expert streams to the original (token, position) order.
    # ------------------------------------------------------------------
    merged = flat_reassemble(contributions, selector)

    # ------------------------------------------------------------------
    # 8️⃣ Collapse the ragged token dimension and the n_active dimension.
    # ------------------------------------------------------------------
    merged = accum_add(merged, rank=1)   # remove ragged token dim
    merged = accum_add(merged, rank=1)   # sum over n_active positions

    # ------------------------------------------------------------------
    # 9️⃣ Store the final (B, D) result off‑chip.
    # ------------------------------------------------------------------
    return offchip_store(merged)