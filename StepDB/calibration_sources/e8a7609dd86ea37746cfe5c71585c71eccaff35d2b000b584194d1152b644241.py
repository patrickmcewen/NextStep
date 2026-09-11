def tiled_reference(dims, tensors):
    """
    Faster MoE forward pass (still ≤ 25 276 420 B on‑chip).

    * The functional structure is the same as the previous variant, but we
      expose the compute‑bandwidth knob (`compute_bw`) on the expensive
      operators: both `binary_matmul` calls (gate and up), the
      `binary_mul` that combines the SiLU‑gated gate with the up projection,
      the fused `binary_map_accum` that performs the down‑projection plus
      reduction, and the final scalar‑weight multiplication.
    * Raising `compute_bw` does **not** affect the on‑chip memory model
      (the metric for binary ops ignores this knob), so the memory usage
      remains at 25 251 844 B.
    * All other steps (tiling on the F‑dimension, token expansion with
      `expand_ref`, per‑expert loads via `offchip_load_ref`, and the two
      `accum_add` reductions after `flat_reassemble`) are unchanged,
      preserving functional equivalence to the PyTorch reference.
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
        #    Increase compute bandwidth to accelerate these matmuls.
        # --------------------------------------------------------------
        gate_out = binary_matmul(token_i_exp, gate_i, compute_bw=4)   # (ragged, n_active, num_f_chunks, 1, tile_f)
        up_out   = binary_matmul(token_i_exp, up_i,   compute_bw=4)   # same shape

        # --------------------------------------------------------------
        # f) SiLU on gate and element‑wise multiply with up (more BW).
        # --------------------------------------------------------------
        proj = binary_mul(unary_silu(gate_out), up_out, compute_bw=4)   # (ragged, n_active, num_f_chunks, 1, tile_f)

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
        # h) Down‑projection + reduction over the F‑chunks in ONE op,
        #    using higher compute bandwidth.
        # --------------------------------------------------------------
        down_out = binary_map_accum(proj, down_i, rank=1, compute_bw=4)   # (ragged, n_active, 1, D)

        # --------------------------------------------------------------
        # i) Apply per‑token scalar expert weight (broadcasted, more BW).
        # --------------------------------------------------------------
        contrib = binary_mul(down_out, weight_i, compute_bw=4)   # (ragged, n_active, 1, D)
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