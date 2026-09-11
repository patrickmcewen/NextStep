# MoE forward pass written entirely with the DSL.
# ------------------------------------------------
# 1. Load the input token matrix `x` as a stream of B tiles, each tile
#    containing the full feature vector (tile_row=1, tile_col=D).
# 2. Load the expert selector (`expert_onehot`) as a MultiHot stream
#    (shape (1, B, n_active, n_experts) with tile (1,1)).
# 3. Insert a singleton stream dimension next to the tile of `x` and
#    expand it to the n_active positions using the selector as a reference.
#    The resulting `token_stream` has stream shape (1, B, n_active) and
#    tile shape (1, D).
# 4. Load the per‑position expert weights (`expert_weights`) as a Float
#    stream with the same stream shape (1, B, n_active) and tile (1,1).
# 5. Use `flat_partition` together with the selector to split both the
#    token stream and the weight stream into per‑expert streams.  For each
#    expert i we obtain `token_i` and `weight_i` with a single dynamic stream
#    dimension that holds all tokens routed to that expert.
# 6. For each expert i:
#       • Load the three weight matrices (gate, up, down) as single‑tile
#         tensors (stream shape (1,1)).  Their tiles are:
#           gate/up : (D, F)   – tile_row=D, tile_col=F
#           down    : (F, D)   – tile_row=F, tile_col=D
#       • Broadcast each matrix to the token stream using `expand_ref`
#         (replace the two singleton stream dims with the token‑stream
#         dynamic dimension).
#       • Compute the expert forward pass:
#           gate_out = token_i @ gate_i
#           up_out   = token_i @ up_i
#           proj     = silu(gate_out) * up_out
#           down_out = proj @ down_i
#       • Scale the result by the expert weight scalar:
#           weighted_i = down_out * weight_i
# 7. Re‑assemble the per‑expert contributions back into the original token
#    order with `flat_reassemble` using the same selector.
#    This produces a stream whose innermost two dimensions are
#    (dynamic = n_active per token, tile = (1, D)).
# 8. Collapse first the ragged dimension added by `flat_reassemble`
#    and then the n_active dimension with a single `accum_add` (rank=2).
# 9. Store the final (B, D) tensor off‑chip with `offchip_store`.  The root
#    node must end with this sink.
#
# The implementation respects all DSL constraints: no raw tensor math,
# no new tensors other than allowed `torch.zeros` (none needed), and all
# shape manipulations are performed via DSL calls.

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # 1) Token stream: one tile per token, full feature vector.
    x = offchip_load(
        tensors["x"],
        stride=[1],
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )  # shape (1, B, 1, D)

    # 2) Multi‑hot selector for top‑k experts.
    control = select_gen(
        tensors["expert_onehot"],
        is_multihot=True,
        n=n_experts,
    )  # shape (1, B, n_active, n_experts, 1, 1)

    # 3) Expand token stream to have an n_active dimension.
    x_promoted = promote(x, rank=0)                 # (1, B, 1, 1, D)
    token_stream = expand_ref(x_promoted, control, expand_rank=1)  # (1, B, n_active, 1, D)

    # 4) Load per‑position scalar expert weights.
    weight_stream = offchip_load(
        tensors["expert_weights"],
        stride=[n_active, 1],
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )  # shape (1, B, n_active, 1, 1)

    # 5) Partition tokens & weights per expert.
    token_parts = flat_partition(token_stream, control, n_experts)   # list[StepTensor]
    weight_parts = flat_partition(weight_stream, control, n_experts)  # list[StepTensor]

    expert_outputs = []
    for i in range(n_experts):
        tok = token_parts[i]      # (dyn, 1, D)
        wgt = weight_parts[i]     # (dyn, 1, 1)

        # Gate weight matrix (D × F)
        gate_raw = offchip_load(
            tensors["gate_weights"][i],
            stride=[1, 1],
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F,
        )
        gate = expand_ref(gate_raw, tok, expand_rank=2)  # (dyn, D, F)

        # Up weight matrix (D × F)
        up_raw = offchip_load(
            tensors["up_weights"][i],
            stride=[1, 1],
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F,
        )
        up = expand_ref(up_raw, tok, expand_rank=2)      # (dyn, D, F)

        # Gate·x and Up·x
        gate_out = binary_matmul(tok, gate)              # (dyn, 1, F)
        up_out   = binary_matmul(tok, up)                # (dyn, 1, F)

        # SiLU(gate) * up
        proj = binary_mul(unary_silu(gate_out), up_out)  # (dyn, 1, F)

        # Down weight matrix (F × D)
        down_raw = offchip_load(
            tensors["down_weights"][i],
            stride=[1, 1],
            out_shape_tiled=(1, 1),
            tile_row=F,
            tile_col=D,
        )
        down = expand_ref(down_raw, tok, expand_rank=2)   # (dyn, F, D)

        # Down projection
        down_out = binary_matmul(proj, down)             # (dyn, 1, D)

        # Apply scalar expert weight
        weighted = binary_mul(down_out, wgt)             # (dyn, 1, D)
        expert_outputs.append(weighted)

    # 7) Re‑assemble per‑token contributions in the original order.
    merged = flat_reassemble(expert_outputs, control)

    # 8) Collapse the ragged dimension and the n_active dimension.
    y = accum_add(merged, rank=2)  # result stream shape (1, B, 1, D)

    # 9) Write the result off‑chip.
    return offchip_store(y)