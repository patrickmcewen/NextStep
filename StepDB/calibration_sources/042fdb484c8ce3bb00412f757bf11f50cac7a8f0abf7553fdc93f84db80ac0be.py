def tiled_reference(dims, tensors):
    # --------------------------------------------------------------
    # MoE forward – lower‑latency, memory‑budget‑friendly variant.
    #
    # 1️⃣ Load token matrix x (B×D) as a stream of 1‑row tiles.
    # 2️⃣ Replicate each token across the n_active positions (static factor).
    # 3️⃣ Load per‑token scalar expert_weights (1×1 tiles) unchanged.
    # 4️⃣ Build an Index selector from expert_onehot.
    # 5️⃣ Partition the token stream and the scalar‑weight stream per expert.
    # 6️⃣ For each expert:
    #      • Broadcast the token stream over the F‑chunks dimension.
    #      • Load gate, up and down weight matrices in 16‑column chunks
    #        (tile_f = 16, f_chunks = F // tile_f = 128).
    #      • Perform three matmuls, SiLU, element‑wise multiply.
    #      • Fuse the reduction over the F‑chunks using `binary_map_accum`
    #        (removes a separate accumulator).
    #      • Multiply by the per‑token scalar expert weight.
    # 7️⃣ Re‑assemble per‑expert streams to the original (token, position) order.
    # 8️⃣ Collapse the ragged token dimension and the n_active dimension.
    # 9️⃣ Store the final (B, D) result off‑chip.
    # --------------------------------------------------------------

    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # ------------------------------------------------------------------
    # 1️⃣ Load tokens (B × D) → stream (B,) with tile (1, D)
    # ------------------------------------------------------------------
    x = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )

    # ------------------------------------------------------------------
    # 2️⃣ Replicate each token over the n_active positions (static factor)
    # ------------------------------------------------------------------
    x_rep = repeat_static(x, factor=n_active)   # stream (B, n_active), tile (1, D)

    # ------------------------------------------------------------------
    # 3️⃣ Load per‑token scalar expert_weights (1×1 tiles)
    # ------------------------------------------------------------------
    weight = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),               # row‑major stride for (B, n_active)
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )

    # ------------------------------------------------------------------
    # 4️⃣ Selector from per‑token one‑hot expert IDs (Index)
    # ------------------------------------------------------------------
    selector = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=n_experts
    )

    # ------------------------------------------------------------------
    # 5️⃣ Partition tokens & scalar weights per expert
    # ------------------------------------------------------------------
    token_parts = flat_partition(x_rep, selector, n_experts)
    weight_parts = flat_partition(weight, selector, n_experts)

    # ------------------------------------------------------------------
    # Chunking parameters for the F‑dimension.
    # ------------------------------------------------------------------
    tile_f = 16                     # 16‑column chunks
    f_chunks = F // tile_f          # = 2048 // 16 = 128

    contributions = []
    for i in range(n_experts):
        # ------------------------------------------------------------------
        # Streams for this expert.
        # ------------------------------------------------------------------
        token_i_base = token_parts[i]      # stream (DynDim_i, n_active), tile (1, D)
        weight_i = weight_parts[i]          # same stream shape, tile (1, 1)

        # ------------------------------------------------------------------
        # Broadcast the token stream over the F‑chunks dimension.
        # ------------------------------------------------------------------
        token_i = repeat_static(token_i_base, factor=f_chunks)   # stream (DynDim_i, n_active, f_chunks), tile (1, D)

        # ------------------------------------------------------------------
        # Gate weight (D → F) – tiled in 16‑column chunks.
        # ------------------------------------------------------------------
        gate_i = offchip_load_ref(
            token_i_base,                               # reference stream
            tensors["gate_weights"][i],
            stride=(1,),
            out_shape_tiled=(f_chunks,),
            tile_row=D,                                 # 1024
            tile_col=tile_f,                            # 16
        )   # stream (DynDim_i, n_active, f_chunks), tile (1024, 16)

        # ------------------------------------------------------------------
        # Up weight (D → F) – same tiling as gate.
        # ------------------------------------------------------------------
        up_i = offchip_load_ref(
            token_i_base,
            tensors["up_weights"][i],
            stride=(1,),
            out_shape_tiled=(f_chunks,),
            tile_row=D,
            tile_col=tile_f,
        )   # stream (DynDim_i, n_active, f_chunks), tile (1024, 16)

        # ------------------------------------------------------------------
        # MatMuls producing (1 × 16) tiles per chunk.
        # ------------------------------------------------------------------
        gate_out = binary_matmul(token_i, gate_i)   # stream (DynDim_i, n_active, f_chunks), tile (1, 16)
        up_out   = binary_matmul(token_i, up_i)     # same shape

        # SiLU + element‑wise multiply.
        proj = binary_mul(unary_silu(gate_out), up_out)   # stream (DynDim_i, n_active, f_chunks), tile (1, 16)

        # ------------------------------------------------------------------
        # Down weight (F → D) – tiled in 16‑row chunks.
        # ------------------------------------------------------------------
        down_i = offchip_load_ref(
            token_i_base,
            tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(f_chunks,),
            tile_row=tile_f,                     # 16
            tile_col=D,                          # 1024
        )   # stream (DynDim_i, n_active, f_chunks), tile (16, 1024)

        # ------------------------------------------------------------------
        # Fuse projection back to D and reduce over the F‑chunks.
        # ------------------------------------------------------------------
        down_out = binary_map_accum(proj, down_i, rank=1)   # stream (DynDim_i, n_active), tile (1, D)

        # ------------------------------------------------------------------
        # Apply per‑token scalar expert weight.
        # ------------------------------------------------------------------
        contrib = binary_mul(down_out, weight_i)   # stream (DynDim_i, n_active), tile (1, D)

        contributions.append(contrib)

    # ------------------------------------------------------------------
    # 7️⃣ Re‑assemble per‑expert streams to original (token, position) order.
    # ------------------------------------------------------------------
    merged = flat_reassemble(contributions, selector)

    # ------------------------------------------------------------------
    # 8️⃣ Collapse the ragged token dimension and the n_active dimension.
    # ------------------------------------------------------------------
    merged = accum_add(merged, rank=1)   # drop ragged token dim
    merged = accum_add(merged, rank=1)   # sum over n_active positions

    # ------------------------------------------------------------------
    # 9️⃣ Write final (B, D) result off‑chip.
    # ------------------------------------------------------------------
    return offchip_store(merged)