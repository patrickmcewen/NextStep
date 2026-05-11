# Implementation reasoning:
# The MoE layer routes each token to its top‑k experts (k = n_active).  We stream the
# input matrix `x` as a (1‑D)‑by‑D tile stream.  The integer selector
# `expert_multihot` (shape B×n_experts) is turned into a MultiHot control stream with
# `select_gen`.  `flat_partition` splits the token stream into one ragged stream per
# expert while preserving the original token order.
#
# Each expert loads its three weight matrices (gate, up, down) as singleton streams
# (stream shape (1,1)) via `offchip_load`.  `expand_ref` with `expand_rank=2` expands
# those singleton streams to the per‑expert token‑stream shape, after which the expert
# forward pass (gate → up → silu → down) is performed with `binary_matmul` and
# `binary_mul`.  The un‑scaled expert contributions are collected.
#
# After processing all experts we re‑assemble the per‑expert token streams back to
# token order with `flat_reassemble`.  The reassembled stream has shape
# (1, B, n_active, 1, D).  The per‑token expert weights (`expert_weights`,
# shape (B, n_active)) are loaded as a stream of the same shape using `offchip_load`
# and then multiplied element‑wise with the reassembled contributions (`binary_mul`).
#
# Finally `accum_add(rank=1)` sums the two expert contributions per token, yielding a
# stream of shape (1, B, 1, D) which `offchip_store` writes back to off‑chip memory and
# returns as a dense tensor of shape (B, D).  All tensor manipulations use only DSL
# ops; no raw arithmetic or indexing is performed.

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # Stream the input matrix (B × D) as tiles (1, D)
    x_stream = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )

    # MultiHot control indicating which experts are selected for each token
    control = select_gen(tensors["expert_multihot"], is_multihot=True, n=n_experts)

    # Split the token stream into one ragged stream per expert
    token_streams = flat_partition(x_stream, control, n=n_experts)

    # Load per‑expert weight matrices as singleton streams (stream shape (1,1))
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

    # Compute each expert's un‑scaled contribution
    expert_outputs = []
    for i in range(n_experts):
        token_i = token_streams[i]                     # (N_i,)×tile(1, D)

        # Expand the singleton weight streams to the token stream shape
        gate_i = expand_ref(gate_streams[i], token_i, expand_rank=2)
        up_i   = expand_ref(up_streams[i],   token_i, expand_rank=2)

        # Expert forward pass
        gate_out = binary_matmul(token_i, gate_i)            # (N_i,)×tile(1, F)
        up_out   = binary_matmul(token_i, up_i)              # (N_i,)×tile(1, F)
        proj     = binary_mul(unary_silu(gate_out), up_out)  # (N_i,)×tile(1, F)

        # Expand down matrix and finish matmul
        down_i = expand_ref(down_streams[i], proj, expand_rank=2)
        down_out = binary_matmul(proj, down_i)               # (N_i,)×tile(1, D)

        expert_outputs.append(down_out)

    # Re‑assemble token streams in the original order (adds an n_active dim)
    merged = flat_reassemble(expert_outputs, control)

    # Load per‑token expert scalar weights (shape B × n_active) as a stream
    weight_stream = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )

    # Apply the scalar weights to the re‑assembled contributions
    weighted = binary_mul(merged, weight_stream)

    # Sum the contributions from the n_active experts for each token
    result = accum_add(weighted, rank=1)

    # Write the final (B, D) matrix back off‑chip
    return offchip_store(result)