def tiled_reference(dims, tensors):
    # --------------------------------------------------------------
    #  Memory‑efficient MoE forward pass.
    #
    #  Strategy:
    #   * Tokens are streamed as 1 × D tiles.
    #   * All large weight matrices are split along the F dimension
    #     into chunks of size `tile_f` = 256.
    #   * The per‑expert gate/up/down weights are loaded with
    #     `offchip_load_ref`, streaming over the F‑chunks.
    #   * The token stream is broadcast over the same chunk dimension
    #     using `expand_ref(promote(...), ...)`.
    #   * Matmul on each chunk yields a (1 × 256) tile; the chunks are
    #     summed with `accum_add` to obtain a (1 × D) tile.
    #   * The per‑token scalar expert weight (1 × 1 tile) is multiplied
    #     directly (broadcast) with the result.
    #   * Ragged per‑expert streams are re‑assembled and the two extra
    #     stream dimensions (ragged token count and n_active) are collapsed.
    # --------------------------------------------------------------

    # ------------------------------------------------------------------
    #  Chunking parameters
    # ------------------------------------------------------------------
    CHUNK_F = dims["tile_f"]               # 256 columns per F‑chunk
    F_CHUNKS = dims["F"] // CHUNK_F        # 2048 / 256 = 8 chunks
    D = dims["D"]                          # token dimension = 1024

    # ------------------------------------------------------------------
    #  1️⃣ Load the token matrix `x` (B × D) as a stream of 1 × D tiles.
    # ------------------------------------------------------------------
    x = offchip_load(
        tensors["x"],
        stride=(1,),                         # one tile per token row
        out_shape_tiled=(dims["B"],),        # stream over B tokens
        tile_row=1,
        tile_col=D,
    )

    # ------------------------------------------------------------------
    #  2️⃣ Replicate each token over the `n_active` expert slots.
    # ------------------------------------------------------------------
    x_rep = repeat_static(x, factor=dims["n_active"])

    # ------------------------------------------------------------------
    #  3️⃣ Load per‑token scalar expert_weights as (1 × 1) tiles.
    # ------------------------------------------------------------------
    weight = offchip_load(
        tensors["expert_weights"],
        stride=(dims["n_active"], 1),        # row‑major stride for (B, n_active)
        out_shape_tiled=(dims["B"], dims["n_active"]),
        tile_row=1,
        tile_col=1,
    )

    # ------------------------------------------------------------------
    #  4️⃣ Selector from per‑token one‑hot expert IDs (Index, not MultiHot).
    # ------------------------------------------------------------------
    selector = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=dims["n_experts"]
    )

    # ------------------------------------------------------------------
    #  5️⃣ Partition tokens & scalar weights per expert.
    # ------------------------------------------------------------------
    token_parts = flat_partition(x_rep, selector, dims["n_experts"])
    weight_parts = flat_partition(weight, selector, dims["n_experts"])

    # ------------------------------------------------------------------
    #  6️⃣ Per‑expert computation
    # ------------------------------------------------------------------
    contributions = []
    for i in range(dims["n_experts"]):
        token_i = token_parts[i]          # stream: (ragged, n_active), tile (1, D)
        weight_i = weight_parts[i]        # stream: (ragged, n_active), tile (1, 1)

        # ----- Gate weight (D → CHUNK_F) streamed over F‑chunks -----
        gate_i = offchip_load_ref(
            token_i,
            tensors["gate_weights"][i],
            stride=(1,),                     # step to next F‑chunk
            out_shape_tiled=(F_CHUNKS,),    # stream over 8 chunks
            tile_row=D,                      # rows = D
            tile_col=CHUNK_F,                # columns = chunk size
        )
        # Broadcast token across the chunk dimension
        token_i_gate = expand_ref(promote(token_i, rank=0), gate_i, expand_rank=1)
        gate_out = binary_matmul(token_i_gate, gate_i)   # tile (1, CHUNK_F)

        # ----- Up weight (D → CHUNK_F) streamed over F‑chunks -----
        up_i = offchip_load_ref(
            token_i,
            tensors["up_weights"][i],
            stride=(1,),
            out_shape_tiled=(F_CHUNKS,),
            tile_row=D,
            tile_col=CHUNK_F,
        )
        token_i_up = expand_ref(promote(token_i, rank=0), up_i, expand_rank=1)
        up_out = binary_matmul(token_i_up, up_i)          # tile (1, CHUNK_F)

        # ----- SiLU activation on the gate and element‑wise multiply -----
        proj = binary_mul(unary_silu(gate_out), up_out)    # tile (1, CHUNK_F)

        # ----- Down weight (CHUNK_F → D) streamed over the same chunks -----
        down_i = offchip_load_ref(
            token_i,
            tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(F_CHUNKS,),
            tile_row=CHUNK_F,
            tile_col=D,
        )
        # No extra broadcast needed – `proj` already has the matching stream shape.

        # Multiply projection with down‑weight chunk → (1 × D) per chunk
        down_out = binary_matmul(proj, down_i)            # tile (1, D)

        # Accumulate over the F‑chunks to obtain a full‑size (1 × D) tile per token
        down_out = accum_add(down_out, rank=1)

        # Apply the per‑token scalar expert weight (broadcast across D)
        contrib = binary_mul(down_out, weight_i)          # tile (1, D)
        contributions.append(contrib)

    # ------------------------------------------------------------------
    #  7️⃣ Re‑assemble per‑expert streams to original (token, position) order
    # ------------------------------------------------------------------
    merged = flat_reassemble(contributions, selector)

    # ------------------------------------------------------------------
    #  8️⃣ Collapse the ragged token dimension and the `n_active` dimension
    # ------------------------------------------------------------------
    merged = accum_add(merged, rank=1)   # drop ragged token dim
    merged = accum_add(merged, rank=1)   # sum over n_active positions

    # ------------------------------------------------------------------
    #  9️⃣ Write the final (B, D) result off‑chip
    # ------------------------------------------------------------------
    return offchip_store(merged)