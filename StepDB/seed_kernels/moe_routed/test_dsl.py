"""Verify step_dsl.py produces correct output for moe_routed.

Implements the same dataflow as step_impl.py using DSL function calls,
then compares against reference.py's compute_gold.
"""
import sys
sys.path.insert(0, "/workspace/DEIOpt/StepGenFlow7/src")

import torch
import torch.nn.functional as F
from step_dsl import (
    offchip_load, offchip_load_ref, select_gen,
    binary_matmul, binary_mul, binary_add,
    binary_map_accum,
    unary_silu,
    accum_add, accum_retile_row,
    flat_partition, flat_reassemble,
    repeat_static, expand_ref, flatten, reshape_stream, retile_streamify,
    offchip_store,
)

SEED = 42

dims = {
    "B": 64,
    "D": 1024,
    "F": 2048,
    "n_experts": 8,
    "n_active": 2,
    "tile_n": 32,
    "tile_f": 256,
}

B = dims["B"]
D = dims["D"]
F_dim = dims["F"]
n_experts = dims["n_experts"]
n_active = dims["n_active"]
tile_n = dims["tile_n"]
tile_f = dims["tile_f"]


# ── Generate tensors (same RNG order as reference.py and step_impl.py) ──────
torch.manual_seed(SEED)

gate_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
up_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
down_weights = [torch.randn(F_dim, D) for _ in range(n_experts)]
x = torch.randn(B, D)
router_w = torch.randn(D, n_experts)

router_logits = x @ router_w
expert_weights_raw, expert_indices = torch.topk(router_logits, n_active, dim=-1)
expert_weights_tensor = torch.softmax(expert_weights_raw, dim=-1)

expert_multihot = torch.zeros(B, n_experts, dtype=torch.int64)
for b in range(B):
    for k in range(n_active):
        expert_multihot[b, expert_indices[b, k]] = 1

expert_onehot = torch.zeros(B, n_active, n_experts, dtype=torch.int64)
for b in range(B):
    for k in range(n_active):
        expert_onehot[b, k, expert_indices[b, k]] = 1


# ── Reference gold ──────────────────────────────────────────────────────────
y_gold = torch.zeros(B, D)
with torch.no_grad():
    for i in range(n_experts):
        idx, top_pos = torch.where(expert_indices == i)
        if len(idx) == 0:
            continue
        gate_out = x[idx] @ gate_weights[i]
        up_out = x[idx] @ up_weights[i]
        projected = F.silu(gate_out) * up_out
        down_out = projected @ down_weights[i]
        y_gold[idx] += down_out * expert_weights_tensor[idx, top_pos, None]

print(f"Gold output shape: {y_gold.shape}")


# ── DSL implementation (mirrors step_impl.py dataflow) ──────────────────────
def tiled_reference():
    F_tiles = F_dim // tile_f  # 8

    # Stage 1: Load input [B, D] as stream [B] of tiles [1, D]
    # One token per tile so flat_partition can route per-token.
    x_tiled = offchip_load(x, stride=(1,), out_shape_tiled=(B,),
                           tile_row=1, tile_col=D)
    # shape: (1, 64, 1, 1024)

    # Stage 2: Routing control — raw (untiled) tensors for flat_partition
    routing_ctrl = select_gen(expert_multihot)
    # shape: (64, 8)

    # Stage 3: Load expert weights [B, n_active] as (1,1) tiles
    ew_tiled = offchip_load(expert_weights_tensor, stride=(n_active, 1),
                            out_shape_tiled=(B, n_active),
                            tile_row=1, tile_col=1)
    # shape: (1, 64, 2, 1, 1)

    # Stage 4: Partition input tokens to experts
    x_parts = flat_partition(x_tiled, routing_ctrl, n_experts)
    # x_parts[i]: (count_i, 1, D)

    # Stage 4b–12: Per-expert MLP: down(silu(gate(x)) * up(x))
    expert_outputs = []
    for i in range(n_experts):
        tokens = x_parts[i]  # (count_i, 1, D)
        count_i = tokens.shape[0]
        if count_i == 0:
            expert_outputs.append(tokens)
            continue

        # Tile tokens for matmul: reshape to (tile_n, D) tiles
        # Reshape: split count_i into (ceil, tile_n)
        reshaped = reshape_stream(tokens, chunk_size=tile_n, rank=0,
                                  add_outer_dim=True)
        # shape: (1, ceil, tile_n, 1, D)

        # Flatten outer dims
        flattened = flatten(reshaped, min_rank=1, max_rank=2)
        # shape: (ceil, tile_n, 1, D)

        # Retile: merge stream dim (tile_n) into tile rows
        tiled_tokens = accum_retile_row(flattened, rank=1)
        # shape: (ceil, tile_n, D)  — tiles of (tile_n, D)

        # Repeat for weight column tiling
        tokens_rep = repeat_static(tiled_tokens, F_tiles)
        # shape: (ceil, F_tiles, tile_n, D)

        # Load gate weights: (D, F_dim) tiled as F_tiles tiles of (D, tile_f)
        gate_w = offchip_load_ref(tiled_tokens, gate_weights[i], stride=(1,),
                                  out_shape_tiled=(F_tiles,),
                                  tile_row=D, tile_col=tile_f)
        # shape: (ceil, 8, D, 256)

        gate_out = binary_matmul(tokens_rep, gate_w)
        # shape: (ceil, 8, tile_n, tile_f)

        # Load up weights
        up_w = offchip_load_ref(tiled_tokens, up_weights[i], stride=(1,),
                                out_shape_tiled=(F_tiles,),
                                tile_row=D, tile_col=tile_f)

        up_out = binary_matmul(tokens_rep, up_w)
        # shape: (ceil, 8, tile_n, tile_f)

        # SiLU(gate) * up
        gate_act = unary_silu(gate_out)
        proj = binary_mul(gate_act, up_out)
        # shape: (ceil, 8, tile_n, tile_f)

        # Load down weights: (F_dim, D) tiled as F_tiles tiles of (tile_f, D)
        down_w = offchip_load_ref(tiled_tokens, down_weights[i], stride=(1,),
                                  out_shape_tiled=(F_tiles,),
                                  tile_row=tile_f, tile_col=D)
        # shape: (ceil, 8, tile_f, D)

        # Down projection + accumulate over F_tiles dim
        expert_out = binary_map_accum(proj, down_w, rank=1)
        # matmul: (ceil, 8, tile_n, tile_f) @ (ceil, 8, tile_f, D) → (ceil, 8, tile_n, D)
        # accum: sum over dim -3 (size 8) → (ceil, tile_n, D)

        # Retile back to (1, D) per token
        retiled = retile_streamify(expert_out, chunk=1, split_row=True)
        # shape: (ceil*tile_n, 1, D)

        # Truncate padding tokens (reshape_stream may have padded)
        retiled = retiled[:count_i]
        # shape: (count_i, 1, D)

        expert_outputs.append(retiled)

    # Stage 13: Partition expert weights by expert_onehot
    weight_ctrl = select_gen(expert_onehot)
    # shape: (64, 2, 8) → reshape(-1, 8) → (128, 8)
    weight_parts = flat_partition(ew_tiled, weight_ctrl, n_experts)
    # weight_parts[i]: (count_i, 1, 1) — same count as expert_outputs[i]

    # Stage 14: Scale expert outputs by weights
    weighted = []
    for i in range(n_experts):
        if expert_outputs[i].shape[0] == 0:
            weighted.append(expert_outputs[i])
            continue
        w = binary_mul(weight_parts[i], expert_outputs[i])
        weighted.append(w)

    # Stage 15: Reassemble weighted outputs back to token order
    reassemble_ctrl = select_gen(expert_multihot)
    reassembled = flat_reassemble(weighted, reassemble_ctrl)
    # shape: (1, 64, n_active, 1, D)

    # Stage 16: Sum over active experts
    result = accum_add(reassembled, rank=1)
    # shape: (1, 64, 1, D)

    return offchip_store(result)


# ── Run and compare ─────────────────────────────────────────────────────────
with torch.no_grad():
    y_dsl = tiled_reference()

print(f"DSL output shape: {y_dsl.shape}")
print(f"Gold output shape: {y_gold.shape}")

assert y_dsl.shape == y_gold.shape, \
    f"Shape mismatch: DSL {y_dsl.shape} vs gold {y_gold.shape}"

max_abs = (y_dsl - y_gold).abs().max().item()
rel_err = (y_dsl - y_gold).norm() / y_gold.norm()
print(f"max_abs_err = {max_abs:.6e}")
print(f"rel_err     = {rel_err:.6e}")

assert rel_err < 1e-5, f"rel_err {rel_err:.6e} exceeds threshold 1e-5"
print("\nPASS — DSL output matches reference.")
