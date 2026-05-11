# Implementation reasoning:
# The MoE layer routes each token to its top‑k experts (k = n_active).  We first stream the
# input tensor `x` as a (1‑D)‑by‑D tile stream (tile_row=1, tile_col=D).  The integer selector
# `expert_multihot` (shape B×n_experts) is turned into a MultiHot control stream with
# `select_gen`.  Using `flat_partition` we split the token stream into one ragged stream per
# expert, preserving the original token order inside each expert stream.
#
# For every expert we load the three weight matrices (gate, up, down) as singleton streams
# (shape (D,F), (D,F), (F,D) respectively) via `offchip_load`.  These are broadcast to the
# per‑expert token stream with `expand_ref`.  The per‑expert forward pass follows the
# reference implementation:
#   gate_out = x @ gate_w
#   up_out   = x @ up_w
#   proj     = silu(gate_out) * up_out
#   down_out = proj @ down_w
#
# Each token’s contribution must be scaled by the expert‑specific weight from
# `expert_weights`.  We build, for every expert, a Python list of these scalar weights in
# the same order that `flat_partition` emitted the token stream (by scanning
# `expert_indices` and `expert_weights`).  The list is turned into a 2‑D float tensor
# (N_i×1) and streamed as a scalar tile stream (tile_row=1, tile_col=1) with `offchip_load`.
# A simple `binary_mul` scales `down_out` by the per‑token scalar weight.
#
# After processing all experts we recombine the ragged per‑expert streams back to token
# order with `flat_reassemble`.  The reassembled stream has an extra dynamic dimension
# of size n_active (the two expert contributions for each token).  A final reduction
# `accum_add` (rank=1) sums the two contributions, yielding a stream of shape (1,B,1,D),
# which `offchip_store` writes back to off‑chip memory and returns as a dense tensor
# of shape (B, D).
#
# All tensor manipulations are expressed via the DSL; no raw arithmetic or indexing is used.

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # Stream the input token matrix (B × D) as tiles (1, D)
    x_stream = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )

    # MultiHot selector that tells which experts are active for each token
    control = select_gen(tensors["expert_multihot"], is_multihot=True, n=n_experts)

    # Split the token stream into one ragged stream per expert
    token_streams = flat_partition(x_stream, control, n=n_experts)

    # Load per‑expert weight matrices as singleton streams
    gate_streams = [
        offchip_load(
            tensors["gate_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )
        for i in range(n_experts)
    ]
    up_streams = [
        offchip_load(
            tensors["up_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )
        for i in range(n_experts)
    ]
    down_streams = [
        offchip_load(
            tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=F,
            tile_col=D,
        )
        for i in range(n_experts)
    ]

    # Build per‑expert scalar weight lists in the order produced by flat_partition
    weights_per_expert = [[] for _ in range(n_experts)]
    expert_indices = tensors["expert_indices"]   # (B, n_active), int64
    expert_weights = tensors["expert_weights"]   # (B, n_active), float32
    for b in range(B):
        for pos in range(n_active):
            i = int(expert_indices[b, pos].item())
            w = float(expert_weights[b, pos].item())
            weights_per_expert[i].append(w)

    # Compute weighted expert contributions
    weighted_outputs = []
    for i in range(n_experts):
        token_i = token_streams[i]                     # (1, N_i, 1, D)

        # Broadcast gate / up matrices to the token stream shape
        gate_i_exp = expand_ref(gate_streams[i], token_i, expand_rank=1)
        up_i_exp   = expand_ref(up_streams[i],   token_i, expand_rank=1)

        # Expert forward pass
        gate_out = binary_matmul(token_i, gate_i_exp)          # (1, N_i, 1, F)
        up_out   = binary_matmul(token_i, up_i_exp)            # (1, N_i, 1, F)
        proj     = binary_mul(unary_silu(gate_out), up_out)    # (1, N_i, 1, F)

        # Broadcast down matrix and finish matmul
        down_i_exp = expand_ref(down_streams[i], proj, expand_rank=1)
        down_out   = binary_matmul(proj, down_i_exp)           # (1, N_i, 1, D)

        # Load per‑token scalar weight for this expert (if any tokens are present)
        w_list = weights_per_expert[i]
        if len(w_list) == 0:
            weighted = down_out
        else:
            weight_tensor = torch.tensor(w_list, dtype=torch.float32).unsqueeze(-1)  # (N_i, 1)
            weight_stream = offchip_load(
                weight_tensor,
                stride=(1,),
                out_shape_tiled=(len(w_list),),
                tile_row=1,
                tile_col=1,
            )
            weighted = binary_mul(down_out, weight_stream)

        weighted_outputs.append(weighted)

    # Reassemble tokens in original order (produces an extra n_active dim)
    merged = flat_reassemble(weighted_outputs, control)

    # Sum the two expert contributions per token
    result = accum_add(merged, rank=1)

    # Write the final (B, D) matrix back off‑chip
    return offchip_store(result)