"""step_dsl version of the routed MoE benchmark.

Mirrors seed_kernels/mlp/moe_routed/step_impl.py at the dataflow level:
load tokens + multihot routing, FlatPartition into per-expert streams,
per-expert MLP (gate / up / silu / mul / down) via LinearOffChipLoadRef +
BinaryMap{Matmul, Mul} + BinaryMapAccum(Matmul), scale by per-(token, slot)
expert weights, FlatReassemble across experts, Accum(Add) over the
per-token active slots, OffChipStore.

The per-expert MLP runs at the (1, D) tile granularity (one matmul per
token), so the Reshape/Flatten/Accum(RetileRow) batching that step_impl
uses to fold tile_n tokens into a single (tile_n, D) tile is not
necessary here — the math is identical because matmul is per-tile.

Inputs come from precompute.py via the `tensors` dict; build_graph cannot
recompute anything.
"""


def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F_dim = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]
    tile_f = dims.get("tile_f", F_dim)

    assert F_dim % tile_f == 0, f"F={F_dim} not divisible by tile_f={tile_f}"
    F_tiles = F_dim // tile_f

    # Stage 1: load input tokens as a stream of (1, D) tiles
    x_tiled = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
        par_dispatch=4,
    )

    # Stage 2: routing control — which experts each token uses
    feature_select = select_gen(
        tensors["expert_multihot"], is_multihot=True, n=n_experts,
    )

    # Stage 3: load per-(token, slot) expert weights as (1, 1) tiles
    weights_load = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
        par_dispatch=4,
    )

    # Stage 4: dispatch each token to its selected experts
    partitioned = flat_partition(
        x_tiled, feature_select, n=n_experts, partition_rank=0,
    )

    # Stages 5-12: per-expert gated MLP
    expert_outputs = []
    for i in range(n_experts):
        tokens = partitioned[i]

        # Repeat each (1, D) token F_tiles times so it can pair with each
        # (D, tile_f) gate/up weight tile.
        tokens_rep = repeat_static(tokens, factor=F_tiles)

        gate_w = offchip_load_ref(
            ref=tokens,
            underlying=tensors["gate_weights"][i],
            stride=(1,),
            out_shape_tiled=(F_tiles,),
            tile_row=D,
            tile_col=tile_f,
            par_dispatch=4,
        )
        gate_out = binary_matmul(tokens_rep, gate_w, compute_bw=1024)

        up_w = offchip_load_ref(
            ref=tokens,
            underlying=tensors["up_weights"][i],
            stride=(1,),
            out_shape_tiled=(F_tiles,),
            tile_row=D,
            tile_col=tile_f,
            par_dispatch=4,
        )
        up_out = binary_matmul(tokens_rep, up_w, compute_bw=1024)

        gate_act = unary_silu(gate_out, compute_bw=1024)
        proj = binary_mul(up_out, gate_act, compute_bw=1024)

        # Down projection: (1, tile_f) @ (tile_f, D) → (1, D), summed
        # over the F_tiles inner stream dim.
        down_w = offchip_load_ref(
            ref=tokens,
            underlying=tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(F_tiles,),
            tile_row=tile_f,
            tile_col=D,
            par_dispatch=4,
        )
        expert_out = binary_map_accum(proj, down_w, rank=1, compute_bw=1024)

        expert_outputs.append(expert_out)

    # Stage 13: dispatch expert-weight scalars to experts via the onehot
    # control — pairs one (1, 1) weight tile with each expert_outputs[i] tile.
    weight_select = select_gen(
        tensors["expert_onehot"], is_multihot=True, n=n_experts,
    )
    weight_parts = flat_partition(
        weights_load, weight_select, n=n_experts, partition_rank=0,
    )

    # Stage 14: scale each expert's output by its per-token weight
    weighted = [
        binary_mul(weight_parts[i], expert_outputs[i], compute_bw=1024)
        for i in range(n_experts)
    ]

    # Stage 15: gather weighted contributions back into per-token slots
    feature_select_reassemble = select_gen(
        tensors["expert_multihot"], is_multihot=True, n=n_experts,
    )
    reassembled = flat_reassemble(
        weighted, feature_select_reassemble, reassemble_rank=0,
    )

    # Stage 16: sum the n_active contributions per token
    result = accum_add(reassembled, rank=1, compute_bw=1024)

    return offchip_store(result, par_dispatch=4)
