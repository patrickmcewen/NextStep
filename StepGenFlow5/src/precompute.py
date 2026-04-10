"""Precompute tensors for each kernel.

Creates all input tensors and routing metadata externally, so that
build_graph / hybrid_reference / tiled_reference receive pre-made tensors
and cannot compute results in PyTorch.

Each precompute function matches the reference's RNG order exactly.
"""
import torch
import torch.nn.functional as F


SEED = 42


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY = {}


def register(name):
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def precompute_tensors(kernel_name: str, dims: dict) -> dict:
    """Create all tensors needed by a kernel, in the reference's RNG order."""
    assert kernel_name in _REGISTRY, (
        f"No precompute function for kernel '{kernel_name}'. "
        f"Known: {sorted(_REGISTRY.keys())}"
    )
    return _REGISTRY[kernel_name](dims)


# ---------------------------------------------------------------------------
# Simple kernels: single input tensor
# ---------------------------------------------------------------------------

@register("copy_2d")
@register("silu_activation")
@register("softmax")
@register("broadcast_diamond")
@register("bufferize_roundtrip")
@register("bufferize_chain")
def _precompute_single_input(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"input": torch.randn(M, K)}


@register("rms_norm")
def _precompute_rms_norm(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"input": torch.randn(M, K), "eps": dims.get("eps", 1e-6)}


@register("layernorm")
def _precompute_layernorm(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"input": torch.randn(M, K), "eps": dims.get("eps", 1e-6)}


@register("chained_unary")
def _precompute_chained_unary(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"input": torch.rand(M, K) * 0.5 + 0.1}


# ---------------------------------------------------------------------------
# Two-input kernels
# ---------------------------------------------------------------------------

@register("element_wise_add")
def _precompute_element_wise_add(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"A": torch.randn(M, K), "B": torch.randn(M, K)}


@register("residual_add_norm")
def _precompute_residual_add_norm(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"X": torch.randn(M, K), "R": torch.randn(M, K), "eps": dims.get("eps", 1e-6)}


@register("multi_load_compute")
def _precompute_multi_load_compute(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {
        "A": torch.randn(M, K), "B": torch.randn(M, K),
        "C": torch.randn(M, K), "D": torch.randn(M, K),
    }


# ---------------------------------------------------------------------------
# GEMM-like kernels
# ---------------------------------------------------------------------------

@register("gemm")
def _precompute_gemm(dims):
    torch.manual_seed(SEED)
    M, K, N = dims["M"], dims["K"], dims["N"]
    return {"A": torch.randn(M, K), "B": torch.randn(K, N)}


@register("qkv_projection")
def _precompute_qkv_projection(dims):
    torch.manual_seed(SEED)
    B, D, proj_dim = dims["B"], dims["D"], dims["proj_dim"]
    return {"x": torch.randn(B, D), "W": torch.randn(D, proj_dim)}


@register("matmul_binarymap")
def _precompute_matmul_binarymap(dims):
    torch.manual_seed(SEED)
    B, H = dims["B"], dims["H"]
    A = torch.randn(B, H)
    model = torch.nn.Linear(H, H, bias=False)
    W = model.weight.T.detach().clone().contiguous()
    return {"A": A, "W": W}


@register("outer_product_accum")
def _precompute_outer_product_accum(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    M, N = dims["M"], dims["N"]
    return {"A": torch.randn(B, M), "B_data": torch.randn(B, N)}


# ---------------------------------------------------------------------------
# Attention kernels
# ---------------------------------------------------------------------------

@register("sdpa_core")
@register("sdpa_two_pass")
def _precompute_sdpa(dims):
    torch.manual_seed(SEED)
    M, N, D = dims["M"], dims["N"], dims["D"]
    return {"Q": torch.randn(M, D), "K": torch.randn(N, D), "V": torch.randn(N, D)}


@register("scaled_dot_product_simple")
@register("sdpa_scaled")
def _precompute_sdpa_scaled(dims):
    torch.manual_seed(SEED)
    M, N, D = dims["M"], dims["N"], dims["D"]
    return {"Q": torch.randn(M, D), "K": torch.randn(N, D), "V": torch.randn(N, D)}


@register("multi_query_attn")
def _precompute_multi_query_attn(dims):
    torch.manual_seed(SEED)
    H, M, N, D = dims["H"], dims["M"], dims["N"], dims["D"]
    return {"Q": torch.randn(H * M, D), "K": torch.randn(N, D), "V": torch.randn(N, D)}


# ---------------------------------------------------------------------------
# Vector reduce
# ---------------------------------------------------------------------------

@register("vector_reduce_sum")
def _precompute_vector_reduce_sum(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"A": torch.randn(M, K)}


# ---------------------------------------------------------------------------
# MLP kernels (model weights before input — RNG order matters)
# ---------------------------------------------------------------------------

@register("gated_mlp")
def _precompute_gated_mlp(dims):
    torch.manual_seed(SEED)
    D, F_dim = dims["D"], dims["F"]
    # RNG order: gate Linear, up Linear, down Linear, then x
    gate_w = torch.nn.Linear(D, F_dim, bias=False).weight.T.detach().clone().contiguous()
    up_w = torch.nn.Linear(D, F_dim, bias=False).weight.T.detach().clone().contiguous()
    down_w = torch.nn.Linear(F_dim, D, bias=False).weight.T.detach().clone().contiguous()
    x = torch.randn(dims["B"], D)
    return {"gate_w": gate_w, "up_w": up_w, "down_w": down_w, "x": x}


@register("moe_expert_single")
def _precompute_moe_expert_single(dims):
    torch.manual_seed(SEED)
    D, F_dim = dims["D"], dims["F"]
    gate_w = torch.nn.Linear(D, F_dim, bias=False).weight.T.detach().clone().contiguous()
    up_w = torch.nn.Linear(D, F_dim, bias=False).weight.T.detach().clone().contiguous()
    down_w = torch.nn.Linear(F_dim, D, bias=False).weight.T.detach().clone().contiguous()
    x = torch.randn(dims["B"], D)
    return {"gate_w": gate_w, "up_w": up_w, "down_w": down_w, "x": x}


# ---------------------------------------------------------------------------
# MoE routed — routing metadata computed here
# ---------------------------------------------------------------------------

@register("moe_routed")
def _precompute_moe_routed(dims):
    B = dims["B"]
    D = dims["D"]
    F_dim = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    torch.manual_seed(SEED)

    # RNG order matches reference exactly
    gate_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    up_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    down_weights = [torch.randn(F_dim, D) for _ in range(n_experts)]
    x = torch.randn(B, D)
    router_w = torch.randn(D, n_experts)

    # Routing metadata — computed here so build_graph can't cheat
    router_logits = x @ router_w
    _, expert_indices = torch.topk(router_logits, n_active, dim=-1)
    expert_weights_raw, _ = torch.topk(router_logits, n_active, dim=-1)
    expert_weights = torch.softmax(expert_weights_raw, dim=-1)

    # Multihot for FlatPartition
    expert_multihot = torch.zeros(B, n_experts, dtype=torch.int64)
    for b in range(B):
        for k in range(n_active):
            expert_multihot[b, expert_indices[b, k]] = 1

    # Onehot for weight partitioning
    expert_onehot = torch.zeros(B, n_active, n_experts, dtype=torch.int64)
    for b in range(B):
        for k in range(n_active):
            expert_onehot[b, k, expert_indices[b, k]] = 1

    return {
        "gate_weights": gate_weights,
        "up_weights": up_weights,
        "down_weights": down_weights,
        "x": x,
        "router_w": router_w,
        "expert_indices": expert_indices,
        "expert_weights": expert_weights,
        "expert_multihot": expert_multihot,
        "expert_onehot": expert_onehot,
    }
