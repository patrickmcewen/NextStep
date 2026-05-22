def tiled_reference(dims, tensors):
    # --------------------------------------------------------------
    # Memory‑optimized MoE forward pass.
    #
    # 1️⃣ Load the token matrix `x` (B×D) as a stream of 1×D tiles.
    # 2️⃣ Replicate each token across the `n_active` positions.
    # 3️⃣ Load per‑token scalar expert weights (`expert_weights`) as 1×1 tiles.
    # 4️⃣ Build an Index selector from `expert_onehot`.
    # 5️⃣ Partition tokens and scalar weights per expert.
    # 6️⃣ Per‑expert computation:
    #      • Broadcast tokens over the F‑chunks dimension (static factor).
    #      • Split the D dimension into K‑chunks (tile_k = 256) and expose
    #        them as a stream dimension.
    #      • Load gate and up weight matrices in (tile_k×tile_f) tiles,
    #        streaming over both F‑chunks and K‑chunks, and reduce over
    #        the K‑chunks with `binary_map_accum`.
    #      • SiLU on the gate, element‑wise multiply with the up projection.
    #      • Load the down weight matrix in (tile_f×D) tiles, streaming
    #        over the F‑chunks only, multiply with the projected activation,
    #        then accumulate over the F‑chunks.
    #      • Multiply by the per‑token scalar expert weight.
    # 7️⃣ Re‑assemble per‑expert streams to original order.
    # 8️⃣ Collapse the ragged token dimension and the `n_active` dimension.
    # 9️⃣ Store the final (B, D) result off‑chip.
    # --------------------------------------------------------------

    # ------------------------------------------------------------------
    # 1️⃣ Load tokens (B, D) → stream (B,), tile (1, D)
    # ------------------------------------------------------------------
    x = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(dims["B"],),
        tile_row=1,
        tile_col=dims["D"],
    )

    # ------------------------------------------------------------------
    # 2️⃣ Replicate each token across the n_active positions
    # ------------------------------------------------------------------
    x_rep = repeat_static(x, factor=dims["n_active"])

    # ------------------------------------------------------------------
    # 3️⃣ Load per‑token scalar expert weights (1×1 tiles)
    # ------------------------------------------------------------------
    weight = offchip_load(
        tensors["expert_weights"],
        stride=(dims["n_active"], 1),
        out_shape_tiled=(dims["B"], dims["n_active"]),
        tile_row=1,
        tile_col=1,
    )

    # ------------------------------------------------------------------
    # 4️⃣ Selector (Index) from one‑hot expert IDs
    # ------------------------------------------------------------------
    selector = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=dims["n_experts"]
    )

    # ------------------------------------------------------------------
    # 5️⃣ Partition token stream and scalar weight stream per expert
    # ------------------------------------------------------------------
    token_parts = flat_partition(x_rep, selector, dims["n_experts"])
    weight_parts = flat_partition(weight, selector, dims["n_experts"])

    # ------------------------------------------------------------------
    # Tiling parameters
    # ------------------------------------------------------------------
    tile_k = 256                                   # split D (1024) → 4 K‑chunks
    tile_f = 256                                   # split F (2048) → 8 F‑chunks
    num_k_chunks = dims["D"] // tile_k             # 4
    num_f_chunks = dims["F"] // tile_f             # 8

    contributions = []
    for i in range(dims["n_experts"]):
        # ------------------------------------------------------------------
        # Streams for this expert (ragged token count + n_active)
        # ------------------------------------------------------------------
        token_raw = token_parts[i]                 # stream shape: (ragged, n_active)
        weight_i  = weight_parts[i]                # same stream shape

        # ------------------------------------------------------------------
        # Broadcast tokens over the F‑chunks dimension (static factor)
        # ------------------------------------------------------------------
        token_f = repeat_static(token_raw, factor=num_f_chunks)  # (ragged, n_active, 8)

        # ------------------------------------------------------------------
        # Split the D dimension into K‑chunks (expose as stream dim)
        # ------------------------------------------------------------------
        token_k = retile_streamify(
            token_f, chunk=tile_k, split_row=False
        )                                             # (ragged, n_active, 32), tile (1,256)

        # ------------------------------------------------------------------
        # Separate the combined (F×K) stream dim into distinct F and K dims
        # ------------------------------------------------------------------
        token_i = reshape_stream(
            token_k, chunk_size=num_k_chunks, rank=0
        )                                             # (ragged, n_active, 8, 4), tile (1,256)

        # ------------------------------------------------------------------
        # Load gate and up weight matrices.
        #   - Ref = token_raw (so the weights are broadcast over the token stream)
        #   - out_shape_tiled = (F_chunks, K_chunks) = (8,4)
        #   - stride must map (f,k) → linear tile index = k*8 + f
        # ------------------------------------------------------------------
        gate_i = offchip_load_ref(
            token_raw,
            tensors["gate_weights"][i],
            stride=(1, num_f_chunks),               # (f,k) → k*8 + f
            out_shape_tiled=(num_f_chunks, num_k_chunks),
            tile_row=tile_k,
            tile_col=tile_f,
        )
        up_i = offchip_load_ref(
            token_raw,
            tensors["up_weights"][i],
            stride=(1, num_f_chunks),
            out_shape_tiled=(num_f_chunks, num_k_chunks),
            tile_row=tile_k,
            tile_col=tile_f,
        )

        # ------------------------------------------------------------------
        # Reduce over the K‑chunks (inner stream dim) using binary_map_accum
        # ------------------------------------------------------------------
        gate_out = binary_map_accum(token_i, gate_i, rank=1)   # (ragged, n_active, 8), tile (1,256)
        up_out   = binary_map_accum(token_i, up_i,   rank=1)   # same shape

        # ------------------------------------------------------------------
        # SiLU on gate and element‑wise multiply with up projection
        # ------------------------------------------------------------------
        proj = binary_mul(unary_silu(gate_out), up_out)        # (ragged, n_active, 8), tile (1,256)

        # ------------------------------------------------------------------
        # Load down weight matrix (stream over F‑chunks only)
        # ------------------------------------------------------------------
        down_i = offchip_load_ref(
            token_raw,
            tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(num_f_chunks,),
            tile_row=tile_f,
            tile_col=dims["D"],
        )                                                       # (ragged, n_active, 8), tile (256,1024)

        # ------------------------------------------------------------------
        # Multiply projected activation with down weight (per‑F‑chunk matmul)
        # ------------------------------------------------------------------
        down_out = binary_matmul(proj, down_i)                # (ragged, n_active, 8), tile (1,1024)

        # ------------------------------------------------------------------
        # Accumulate over the F‑chunks to obtain full‑dim result
        # ------------------------------------------------------------------
        down_sum = accum_add(down_out, rank=1)                # (ragged, n_active), tile (1,1024)

        # ------------------------------------------------------------------
        # Apply per‑token scalar expert weight (broadcast automatically)
        # ------------------------------------------------------------------
        contrib = binary_mul(down_sum, weight_i)              # (ragged, n_active), tile (1,1024)

        contributions.append(contrib)

    # ----------------------------------------------------------------------
    # 7️⃣ Re‑assemble per‑expert streams to original (token, position) order
    # ----------------------------------------------------------------------
    merged = flat_reassemble(contributions, selector)

    # ----------------------------------------------------------------------
    # 8️⃣ Collapse the ragged token dimension and the n_active dimension
    # ----------------------------------------------------------------------
    merged = accum_add(merged, rank=1)   # remove ragged token dim
    merged = accum_add(merged, rank=1)   # sum over n_active positions

    # ----------------------------------------------------------------------
    # 9️⃣ Store final (B, D) result off‑chip
    # ----------------------------------------------------------------------
    return offchip_store(merged)