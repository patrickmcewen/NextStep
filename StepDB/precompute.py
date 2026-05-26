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
@register("tile_grid_transpose_2d")
def _precompute_single_input(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return {"input": torch.randn(M, K)}


@register("head_split_permute")
def _precompute_head_split_permute(dims):
    torch.manual_seed(SEED)
    S, H, D = dims["S"], dims["H"], dims["D"]
    return {"input": torch.randn(S, H * D)}


@register("attn_layout_permute_4d")
def _precompute_attn_layout_permute_4d(dims):
    torch.manual_seed(SEED)
    B, H, S, D = dims["B"], dims["H"], dims["S"], dims["D"]
    return {"input": torch.randn(B, H, S, D)}


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
@register("gemm_parallel")
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
@register("sdpa_core_max")
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
# flashinfer_trace-derived attention kernels (bfloat16 inputs, fixed constants)
# RNG order must match seed_kernels/<kernel>/reference.py exactly.
# ---------------------------------------------------------------------------

@register("gqa_ragged_prefill")
def _precompute_gqa_ragged_prefill(dims):
    import math

    # Constants baked into the source JSON (gqa_ragged_prefill_causal_h32_kv16_d128)
    NUM_QO_HEADS = 32
    NUM_KV_HEADS = 16
    HEAD_DIM = 128

    torch.manual_seed(SEED)
    batch_size = dims["batch_size"]
    q_len = dims["q_len"]
    kv_len = dims["kv_len"]
    assert kv_len >= q_len, "kv_len must be >= q_len for the causal mask"

    total_q = batch_size * q_len
    total_kv = batch_size * kv_len

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * q_len
    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * kv_len
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    return {
        "q": q, "k": k, "v": v,
        "qo_indptr": qo_indptr, "kv_indptr": kv_indptr,
        "sm_scale": sm_scale,
    }


@register("gqa_paged_decode")
def _precompute_gqa_paged_decode(dims):
    import math

    # Constants from gqa_paged_decode_h32_kv16_d128_ps1
    NUM_QO_HEADS = 32
    NUM_KV_HEADS = 16
    HEAD_DIM = 128
    PAGE_SIZE = 1

    torch.manual_seed(SEED)
    batch_size = dims["batch_size"]
    kv_len = dims["kv_len"]
    num_pages = batch_size * kv_len

    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * kv_len
    kv_indices = torch.arange(num_pages, dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    return {
        "q": q, "k_cache": k_cache, "v_cache": v_cache,
        "kv_indptr": kv_indptr, "kv_indices": kv_indices,
        "sm_scale": sm_scale,
    }


@register("mla_paged_decode")
def _precompute_mla_paged_decode(dims):
    import math

    # Constants from mla_paged_decode_h16_ckv512_kpe64_ps1 (DeepSeek-V3 TP=8)
    NUM_QO_HEADS = 16
    HEAD_DIM_CKV = 512
    HEAD_DIM_KPE = 64
    PAGE_SIZE = 1

    torch.manual_seed(SEED)
    batch_size = dims["batch_size"]
    kv_len = dims["kv_len"]
    num_pages = batch_size * kv_len

    q_nope = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16)
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16)
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16)
    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * kv_len
    kv_indices = torch.arange(num_pages, dtype=torch.int32)
    # Per JSON: sm_scale = 1/sqrt(128 + 64), using pre-absorption head dims
    sm_scale = 1.0 / math.sqrt(128 + HEAD_DIM_KPE)
    return {
        "q_nope": q_nope, "q_pe": q_pe,
        "ckv_cache": ckv_cache, "kpe_cache": kpe_cache,
        "kv_indptr": kv_indptr, "kv_indices": kv_indices,
        "sm_scale": sm_scale,
    }


# ---------------------------------------------------------------------------
# Rotary position embedding (Q/K rotation block from end_to_end reference)
# ---------------------------------------------------------------------------

@register("rope")
def _precompute_rope(dims):
    """Inputs for the rope step_impl: Q and K are stacked along the heads dim
    into one (batch, num_q_heads + num_kv_heads, head_dim) tensor `QK`, since
    rotate_half + cos/sin multiply-add is per-head and applies identically to
    Q and K rows. The step_impl runs a single rotate_half pipeline on QK.
    """
    torch.manual_seed(SEED)
    batch = dims["batch"]
    num_q_heads = dims["num_q_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    assert head_dim % 2 == 0, "head_dim must be even for rotate_half"

    Q = torch.randn(batch, num_q_heads, head_dim)
    K = torch.randn(batch, num_kv_heads, head_dim)
    cos = torch.randn(batch, 1, head_dim)
    sin = torch.randn(batch, 1, head_dim)
    QK = torch.cat([Q, K], dim=1).contiguous()
    return {"QK": QK, "cos": cos, "sin": sin}


@register("qkv_gen")
def _precompute_qkv_gen(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    D = dims["D"]
    N_HEAD = dims["N_HEAD"]
    HEAD_DIM = dims["HEAD_DIM"]
    assert HEAD_DIM % 2 == 0, "HEAD_DIM must be even for rotate_half"

    q_proj = torch.randn(D, N_HEAD * HEAD_DIM)
    k_proj = torch.randn(D, N_HEAD * HEAD_DIM)
    v_proj = torch.randn(D, N_HEAD * HEAD_DIM)
    x = torch.randn(B, D)
    cos = torch.randn(B, 1, HEAD_DIM)
    sin = torch.randn(B, 1, HEAD_DIM)
    return {
        "x": x,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "cos": cos,
        "sin": sin,
    }


@register("rotate_half")
def _precompute_rotate_half(dims):
    torch.manual_seed(SEED)
    batch = dims["batch"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    assert head_dim % 2 == 0, "head_dim must be even for rotate_half"

    x = torch.randn(batch, num_heads, head_dim)
    return {"x": x}


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

@register("generated_moe")
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

    expert_multihot = torch.zeros(B, n_experts, dtype=torch.int64)
    for b in range(B):
        for k in range(n_active):
            expert_multihot[b, expert_indices[b, k]] = 1

    expert_onehot = torch.zeros(B, n_active, n_experts, dtype=torch.int64)
    for b in range(B):
        for k in range(n_active):
            expert_onehot[b, k, expert_indices[b, k]] = 1

    # Stack per-expert weights into single tensors so the DSL can index them
    # via tensors["gate_weights"][i] and still get a StepRawTensor (the
    # framework only wraps top-level torch.Tensor values, not Python lists).
    # `t[i]` on a stacked tensor matches the list-indexing semantics used by
    # reference.py / step_impl.py exactly, so this is a no-op for them.
    return {
        "gate_weights": torch.stack(gate_weights, dim=0).contiguous(),
        "up_weights": torch.stack(up_weights, dim=0).contiguous(),
        "down_weights": torch.stack(down_weights, dim=0).contiguous(),
        "x": x,
        "router_w": router_w,
        "expert_indices": expert_indices,
        "expert_weights": expert_weights,
        "expert_multihot": expert_multihot,
        "expert_onehot": expert_onehot,
    }


# ---------------------------------------------------------------------------
# End-to-end transformer layer (attention + MoE)
# ---------------------------------------------------------------------------

@register("end_to_end")
@register("generated_end_to_end")
@register("generated_end_to_end_2")
def _precompute_end_to_end(dims):
    """Precompute tensors for the end-to-end transformer layer kernel.

    Matches the RNG order in seed_kernels/end_to_end/reference.py:
      torch.manual_seed(5); random.seed(42)
      input_tensor, q_proj, k_proj, v_proj, cos, sin,
      expert_weights (softmax of randn), w_gate_list, w_up_list, w_down_list,
      per-element k_cache[i] / v_cache[i] fills, o_proj_weight
    """
    import sys
    import random
    from pathlib import Path
    import numpy as np

    # step_tl root derivation mirrors reference.py — needed for model_configs
    # import and for locating routing / trace data files.
    import step_py as _sp
    _STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
    if _STEP_TL_ROOT not in sys.path:
        sys.path.insert(0, _STEP_TL_ROOT)
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B, Qwen30B, SmallerQwen30B,
    )

    _EXPERT_ROUTING = {
        ("mixtral", 64): (8, 10),
        ("mixtral", 1024): (19, 9),
        ("qwen", 64): (32, 12),
        ("qwen", 1024): (22, 16),
    }

    model_name = dims["model_name"]
    batch = dims.get("batch", 64)
    scale_seq = dims.get("scale_seq", 1)
    is_small = dims.get("is_small", False)
    stdev = dims["stdev"]
    start = dims["start"]
    end = dims["end"]

    torch.manual_seed(5)
    random.seed(42)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        assert False, f"Unknown model_name: {model_name}"

    input_tensor = torch.randn(batch, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(batch, 1, mc.head_dim)
    sin = torch.randn(batch, 1, mc.head_dim)

    maxN = 4096 * scale_seq
    k_cache = torch.zeros(batch, maxN, mc.num_kv_heads, mc.head_dim)
    v_cache = torch.zeros(batch, maxN, mc.num_kv_heads, mc.head_dim)

    routing_key = (model_name, batch)
    assert routing_key in _EXPERT_ROUTING, f"No expert routing for {routing_key}"
    iter_idx, layer_idx = _EXPERT_ROUTING[routing_key]
    routing_path = (
        Path(_STEP_TL_ROOT)
        / f"dyn_tiling/expert_routing/{model_name}_b{batch}"
        / f"iter_{iter_idx:03d}_layer_{layer_idx:03d}.npz"
    )
    assert routing_path.exists(), f"Expert routing file not found: {routing_path}"
    expert_indices = torch.from_numpy(np.load(str(routing_path))["data"])

    # SelectGen needs these as `underlying` tensors. Pure function of
    # expert_indices (loaded from .npz, no RNG draw), so RNG sequence is
    # preserved. Mirrors the moe_routed precompute so LLM-side build_graph
    # can index tensors["expert_multihot"] / tensors["expert_onehot"]
    # without depending on step_tl's utils.moe helpers.
    _one_hot = torch.nn.functional.one_hot(
        expert_indices.to(torch.int64), num_classes=mc.n_routed_experts
    ).to(torch.int64)
    expert_multihot = _one_hot.sum(dim=-2)
    expert_onehot = _one_hot

    expert_weights = torch.softmax(
        torch.randn(batch, mc.n_activated_experts), dim=-1
    )
    w_gate_list = [
        torch.nn.Linear(mc.dim, mc.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    w_up_list = [
        torch.nn.Linear(mc.dim, mc.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    w_down_list = [
        torch.nn.Linear(mc.moe_inter_dim, mc.dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]

    assert batch == end - start + 1, f"batch={batch} != end-start+1={end - start + 1}"
    trace_path = (
        Path(_STEP_TL_ROOT)
        / f"dynamic_par/azure_trace/b{batch}"
        / f"conv_stdev{stdev:04d}_{start:04d}_{end:04d}.npy"
    )
    assert trace_path.exists(), f"Trace file not found: {trace_path}"
    num_token_list = np.load(str(trace_path)).astype(np.int64).tolist()
    num_token_list = [x * scale_seq for x in num_token_list]

    for i in range(batch):
        k_cache[i, :num_token_list[i]] = torch.randn(
            num_token_list[i], mc.num_kv_heads, mc.head_dim
        )
        v_cache[i, :num_token_list[i]] = torch.randn(
            num_token_list[i], mc.num_kv_heads, mc.head_dim
        )

    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)

    # Ragged metadata: emit as 1-D int64 tensors (not Python lists). The
    # framework wraps every tensors[...] entry as StepRawTensor at root, so
    # the LLM can pass these straight into metadata_gen / cache_read_addr_gen
    # without the (broken) ``torch.tensor(<list>) -> metadata_gen`` bridge.
    # Also pre-derive ``seq_len`` (tile/token count after append) and
    # ``idx`` (per-batch cache slot) so the LLM doesn't need to recreate
    # them with banned ops (raw tensor arithmetic / torch.arange).
    num_token_list_t = torch.tensor(num_token_list, dtype=torch.int64)
    seq_len_t = num_token_list_t + 1
    idx_t = torch.arange(batch, dtype=torch.int64)

    return {
        "input_tensor": input_tensor,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "cos": cos,
        "sin": sin,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "expert_indices": expert_indices,
        "expert_multihot": expert_multihot,
        "expert_onehot": expert_onehot,
        "expert_weights": expert_weights,
        "w_gate_list": w_gate_list,
        "w_up_list": w_up_list,
        "w_down_list": w_down_list,
        "num_token_list": num_token_list_t,
        "seq_len": seq_len_t,
        "idx": idx_t,
        "o_proj_weight": o_proj_weight,
    }


@register("gqa_tiled_decode")
def _precompute_gqa_tiled_decode(dims):
    torch.manual_seed(SEED)
    batch_size = dims["batch_size"]
    num_kv_heads = dims["num_kv_heads"]
    query_per_kvhead = dims["query_per_kvhead"]
    head_dim = dims["head_dim"]
    max_seq_len_tiles = dims["max_seq_len_tiles"]
    tile_N = dims["tile_N"]
    max_seq_len = max_seq_len_tiles * tile_N

    query = torch.randn(batch_size, num_kv_heads, query_per_kvhead, head_dim)
    key = torch.randn(batch_size, num_kv_heads, head_dim)
    value = torch.randn(batch_size, num_kv_heads, head_dim)
    k_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)
    v_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)

    idx = torch.arange(batch_size, dtype=torch.int64)
    seq_len_tiled = torch.randint(
        low=1, high=max_seq_len_tiles + 1, size=(batch_size,), dtype=torch.int64
    )
    offset = torch.randint(low=0, high=tile_N, size=(batch_size,), dtype=torch.int64)
    return {
        "query": query,
        "key": key,
        "value": value,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "idx": idx,
        "seq_len_tiled": seq_len_tiled,
        "offset": offset,
        "tile_N": tile_N,
    }


@register("gqa_decode_e2e")
def _precompute_gqa_decode_e2e(dims):
    """Inputs for the ragged-batched GQA decode + O-proj kernel.

    Mirrors the GQA + O-proj portion of end_to_end's reference (steps [6]+[7])
    on a post-pipeline cache: Q is post-RoPE/RMSNorm and the KV cache already
    has the new K/V appended (``seq_lens[i]`` here corresponds to
    ``num_token_list[i] + 1`` in end_to_end). Each batch's seq length is
    drawn uniformly from ``[seq_len_min, seq_len_max]`` constrained to be a
    multiple of ``tile_seq`` so the step_impl can use a static tile size
    while still reading variable per-batch lengths via DynN.
    """
    torch.manual_seed(SEED)
    batch = dims["batch"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    hidden_dim = dims["hidden_dim"]
    seq_len_min = dims["seq_len_min"]
    seq_len_max = dims["seq_len_max"]
    tile_seq = dims["tile_seq"]
    tile_hidden = dims["tile_hidden"]
    assert num_heads % num_kv_heads == 0, (
        f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
    )
    assert seq_len_min % tile_seq == 0 and seq_len_max % tile_seq == 0, (
        f"seq_len_min={seq_len_min} and seq_len_max={seq_len_max} must be "
        f"multiples of tile_seq={tile_seq}"
    )
    assert 1 <= seq_len_min <= seq_len_max, (
        f"need 1 <= seq_len_min={seq_len_min} <= seq_len_max={seq_len_max}"
    )
    assert hidden_dim % tile_hidden == 0, (
        f"hidden_dim={hidden_dim} must be divisible by tile_hidden={tile_hidden}"
    )

    Q = torch.randn(batch, num_heads, head_dim)
    k_cache = torch.randn(batch, seq_len_max, num_kv_heads, head_dim)
    v_cache = torch.randn(batch, seq_len_max, num_kv_heads, head_dim)
    o_proj_weight = torch.randn(num_heads * head_dim, hidden_dim)
    n_min = seq_len_min // tile_seq
    n_max = seq_len_max // tile_seq
    seq_lens_tiles = torch.randint(
        low=n_min, high=n_max + 1, size=(batch,), dtype=torch.int64
    )
    # Keep as 1-D int64 tensor (not list) so it auto-wraps as StepRawTensor
    # at root and can flow directly into metadata_gen / cache_read_addr_gen.
    seq_lens = seq_lens_tiles * tile_seq

    # tile_mask[b, t] = 1.0 if tile t is fully valid for batch b, else 0.0.
    # Used by step_impl to zero out invalid tiles after exp() so the uniform-S
    # graph reproduces the ragged reference's per-batch sum.
    n_tiles = seq_len_max // tile_seq
    tile_mask = torch.zeros(batch, n_tiles, dtype=torch.float32)
    for b in range(batch):
        tile_mask[b, : seq_lens_tiles[b]] = 1.0

    return {
        "Q": Q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "seq_lens": seq_lens,
        "tile_mask": tile_mask,
        "o_proj_weight": o_proj_weight,
    }


@register("kv_cache_tile_append")
def _precompute_kv_cache_tile_append(dims):
    torch.manual_seed(SEED)
    batch_size = dims["batch_size"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    max_seq_len_tiles = dims["max_seq_len_tiles"]
    tile_N = dims["tile_N"]
    max_seq_len = max_seq_len_tiles * tile_N

    key = torch.randn(batch_size, num_kv_heads, head_dim)
    value = torch.randn(batch_size, num_kv_heads, head_dim)
    k_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)
    v_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)

    idx = torch.arange(batch_size, dtype=torch.int64)
    seq_len_tiled = torch.randint(
        low=1, high=max_seq_len_tiles + 1, size=(batch_size,), dtype=torch.int64
    )
    offset = torch.randint(low=0, high=tile_N, size=(batch_size,), dtype=torch.int64)
    return {
        "key": key,
        "value": value,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "idx": idx,
        "seq_len_tiled": seq_len_tiled,
        "offset": offset,
        "tile_N": tile_N,
    }


# ---------------------------------------------------------------------------
# Simple prefill transformer layer (single sequence, no KV cache)
# ---------------------------------------------------------------------------

@register("prefill_transformer_simple")
@register("generated_prefill_transformer")
@register("generated_prefill_transformer_working")
def _precompute_prefill_transformer_simple(dims):
    """Precompute tensors for the simple prefill transformer kernel.

    RNG order mirrors seed_kernels/transformer_layer/prefill_transformer_simple/
    reference.py (no random.seed, no external files):
      torch.manual_seed(SEED)
      input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight,
      w_gate_list, w_up_list, w_down_list, router_w

    The pre-attention pipeline (RMSNorm, QKV projection, per-head Q/K
    RMSNorm, RoPE) and self-attention live in the step_impl dataflow
    graph (single-tile-per-head sdpa).

    Top-k routing tensors are emitted here because they depend on float64
    attention output to match the reference exactly: tiny float32 noise in
    a streaming float32 attention can flip topk decisions near boundary
    logits, so we reproduce the reference's float64 attention here just
    for the routing computation. The MoE block itself runs in dataflow.
    """
    import sys
    from pathlib import Path

    import step_py as _sp
    _STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
    if _STEP_TL_ROOT not in sys.path:
        sys.path.insert(0, _STEP_TL_ROOT)
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B, Qwen30B, SmallerQwen30B,
    )

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        assert False, f"Unknown model_name: {model_name!r}"

    torch.manual_seed(SEED)

    input_tensor = torch.randn(seq_len, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    sin = torch.randn(seq_len, 1, mc.head_dim)
    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)
    w_gate_list = [
        torch.nn.Linear(mc.dim, mc.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    w_up_list = [
        torch.nn.Linear(mc.dim, mc.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    w_down_list = [
        torch.nn.Linear(mc.moe_inter_dim, mc.dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    router_w = torch.randn(mc.dim, mc.n_routed_experts)

    # ---- Top-k routing tensors. Runs in fp32 with max-subtraction softmax so
    # routing decisions match the reference's fp32 attention exactly. ----
    def _rms_norm_t(x, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _rotate_half_t(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    with torch.no_grad():
        normed = _rms_norm_t(input_tensor)
        Q = (normed @ q_proj).view(seq_len, mc.num_heads, mc.head_dim)
        K = (normed @ k_proj).view(seq_len, mc.num_kv_heads, mc.head_dim)
        V = (normed @ v_proj).view(seq_len, mc.num_kv_heads, mc.head_dim)
        Q = _rms_norm_t(Q); K = _rms_norm_t(K)
        Q_post_rope = (Q * cos + _rotate_half_t(Q) * sin).contiguous()
        K_post_rope = (K * cos + _rotate_half_t(K) * sin).contiguous()
        V_post_rope = V.contiguous()

        Qh = (
            Q_post_rope
            .view(seq_len, mc.num_kv_heads, mc.query_per_kvhead, mc.head_dim)
            .permute(1, 2, 0, 3)
        )
        Kh = K_post_rope.permute(1, 0, 2).unsqueeze(1)
        Vh = V_post_rope.permute(1, 0, 2).unsqueeze(1)
        scores = Qh @ Kh.transpose(-1, -2)
        row_max = scores.amax(dim=-1, keepdim=True)
        e = torch.exp(scores - row_max)
        attn = e @ Vh / e.sum(dim=-1, keepdim=True)
        attn = attn.permute(2, 0, 1, 3).reshape(
            seq_len, mc.num_heads, mc.head_dim
        )
        o_proj_out = attn.reshape(seq_len, mc.num_heads * mc.head_dim) @ o_proj_weight
        normed_2 = _rms_norm_t(o_proj_out + input_tensor)

        router_logits = normed_2 @ router_w
        _, expert_indices = torch.topk(router_logits, mc.n_activated_experts, dim=-1)
        expert_weights_raw, _ = torch.topk(router_logits, mc.n_activated_experts, dim=-1)
        expert_weights = torch.softmax(expert_weights_raw, dim=-1)

    expert_multihot = torch.zeros(
        seq_len, mc.n_routed_experts, dtype=torch.int64,
    )
    for s in range(seq_len):
        for k in range(mc.n_activated_experts):
            expert_multihot[s, expert_indices[s, k]] = 1
    expert_onehot = torch.zeros(
        seq_len, mc.n_activated_experts, mc.n_routed_experts, dtype=torch.int64,
    )
    for s in range(seq_len):
        for k in range(mc.n_activated_experts):
            expert_onehot[s, k, expert_indices[s, k]] = 1

    return {
        "input_tensor": input_tensor,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "cos": cos,
        "sin": sin,
        "o_proj_weight": o_proj_weight,
        # Stacked expert weights: [n_routed_experts, ...]. ``w[i]`` returns a
        # contiguous view equivalent to the old ``w_*_list[i]``.
        "w_gate": torch.stack(w_gate_list, dim=0),
        "w_up": torch.stack(w_up_list, dim=0),
        "w_down": torch.stack(w_down_list, dim=0),
        "router_w": router_w,
        "expert_weights": expert_weights,
        "expert_multihot": expert_multihot,
        "expert_onehot": expert_onehot,
    }


@register("basic_prefill_attention")
def _precompute_basic_prefill_attention(dims):
    """Precompute tensors for the attention-only prefill kernel.

    Only raw RNG'd tensors — the full pre-attention pipeline (RMSNorm, QKV
    projection, per-head Q/K RMSNorm, RoPE) lives in the step_impl as a
    STeP sub-graph that's executed at build time to materialize Q/K/V
    post-RoPE tensors, which then become off-chip underlyings for the
    per-head sdpa main graph.
    """
    import sys
    from pathlib import Path

    import step_py as _sp
    _STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
    if _STEP_TL_ROOT not in sys.path:
        sys.path.insert(0, _STEP_TL_ROOT)
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B, Qwen30B, SmallerQwen30B,
    )

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        assert False, f"Unknown model_name: {model_name!r}"

    torch.manual_seed(SEED)

    input_tensor = torch.randn(seq_len, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    sin = torch.randn(seq_len, 1, mc.head_dim)
    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)

    return {
        "input_tensor": input_tensor,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "cos": cos,
        "sin": sin,
        "o_proj_weight": o_proj_weight,
    }

@register("sdpa_kv_read")
def _precompute_sdpa_kv_read(dims):
    torch.manual_seed(SEED)
    hidden_dim = dims["hidden_dim"]
    seq_len = dims["seq_len"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    num_kv_heads = dims["num_kv_heads"]
    batch_size = dims["batch_size"]
    batch_idx = dims["batch_idx"]
    return {"x": torch.randn(1, hidden_dim), "W_q": torch.randn(hidden_dim, num_heads * head_dim), "W_k": torch.randn(hidden_dim, num_kv_heads * head_dim), "W_v": torch.randn(hidden_dim, num_kv_heads * head_dim), "K_cache": torch.randn(batch_size, seq_len+1, num_kv_heads, head_dim), "V_cache": torch.randn(batch_size, seq_len+1, num_kv_heads, head_dim), "batch_idx": batch_idx, "seq_len": seq_len}


# ---------------------------------------------------------------------------
# HuggingFace-imported kernels (full unmodified HF stack, randn-init weights)
# ---------------------------------------------------------------------------

@register("hf__gpt2")
def _precompute_hf__gpt2(dims):
    """Inputs + every named parameter for the HF gpt2 kernel.

    Constructs the HF model under a seeded RNG via ``from_config``, then
    extracts every ``named_parameters()`` entry into the precompute dict
    as a raw detached tensor. ``input_ids`` is generated under a separate
    seed so adding inputs later doesn't perturb weight RNG.

    Returned dict layout:
      "input_ids":                                  (B, S) int64
      "<hf_param_name>":  e.g. "transformer.wte.weight", "transformer.h.0.attn.c_attn.weight", ...
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(dims["model_name"])
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(cfg)
    # state_dict() (not named_parameters()) so that tied weights appear under
    # every name HF uses (e.g. lm_head.weight aliasing transformer.wte.weight)
    # and registered buffers (causal masks, etc.) come along too.
    weights = {n: t.detach().clone() for n, t in model.state_dict().items()}

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(
        0, cfg.vocab_size, (dims["batch_size"], dims["seq_len"])
    )
    return {"input_ids": input_ids, **weights}


@register("qk_reshape_score")
def _precompute_qk_reshape_score(dims):
    torch.manual_seed(SEED)
    seq_len = dims["seq_len"]
    hidden_size = dims["hidden_size"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    return {
        "x": torch.randn(seq_len, hidden_size),
        "w": torch.randn(hidden_size, num_heads * head_dim),
        "k": torch.randn(seq_len, num_kv_heads, head_dim),
    }


@register("gemm_rms_norm")
def _precompute_gemm_rms_norm(dims):
    torch.manual_seed(SEED)
    M, K, N = dims["M"], dims["K"], dims["N"]
    return {
        "A": torch.randn(M, K),
        "B": torch.randn(K, N),
        "eps": dims.get("eps", 1e-6),
    }


# ---------------------------------------------------------------------------
# KernelBench Level 3 / Problem 1: 3-Linear MLP with ReLU
# ---------------------------------------------------------------------------

@register("kernelbench_mlp")
def _precompute_kernelbench_mlp(dims):
    # RNG order MUST match seed_kernels/kernelbench/level3/mlp/reference.py:
    #   seed -> Linear(D_in,D_h1) -> Linear(D_h1,D_h2) -> Linear(D_h2,D_out)
    #     -> torch.rand(B, D_in)
    torch.manual_seed(SEED)
    B = dims["B"]
    D_in = dims["D_in"]
    D_h1 = dims["D_h1"]
    D_h2 = dims["D_h2"]
    D_out = dims["D_out"]

    fc1 = torch.nn.Linear(D_in, D_h1)
    fc2 = torch.nn.Linear(D_h1, D_h2)
    fc3 = torch.nn.Linear(D_h2, D_out)
    x = torch.rand(B, D_in)

    # Biases reshaped to (1, D) so they are >=2-D and loadable by offchip_load.
    # Same broadcast semantics as 1-D in PyTorch.
    return {
        "x": x,
        "w1": fc1.weight.detach().clone().contiguous(),
        "b1": fc1.bias.detach().clone().contiguous().unsqueeze(0),
        "w2": fc2.weight.detach().clone().contiguous(),
        "b2": fc2.bias.detach().clone().contiguous().unsqueeze(0),
        "w3": fc3.weight.detach().clone().contiguous(),
        "b3": fc3.bias.detach().clone().contiguous().unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# KernelBench Level 4 / Problem 7: GPT-2 full forward
# ---------------------------------------------------------------------------


def _stack_gpt2_block(sd, L, key_template):
    """Matrix-shaped per-layer params: ``[i]`` is already ≥2-D after stacking."""
    return torch.stack(
        [sd[key_template.format(l=l)].detach().clone().contiguous() for l in range(L)],
        dim=0,
    ).contiguous()


def _stack_gpt2_block_vec(sd, L, key_template):
    """Vector-shaped per-layer params (``(D,)``): pad to ``(L, 1, D)`` so a
    per-layer slice ``[i]`` is ``(1, D)`` — 2-D, as required by StepGenFlow's
    on-chip ``rank >= 2`` invariant. Matmul ``+ bias`` broadcasts unchanged;
    ``F.layer_norm`` callers ``.squeeze(0)`` at use."""
    return torch.stack(
        [
            sd[key_template.format(l=l)].detach().clone().contiguous().unsqueeze(0)
            for l in range(L)
        ],
        dim=0,
    ).contiguous()


@register("kernelbench_gpt2")
def _precompute_kernelbench_gpt2(dims):
    # RNG order MUST match seed_kernels/kernelbench/level4/gpt2/reference.py:
    #   seed -> GPT2LMHeadModel(config)  (consumes all weight RNG internally)
    #        -> torch.randint(0, V, (B, T))    for input_ids
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(SEED)
    cfg = GPT2Config(
        vocab_size=dims["V"],
        n_positions=dims["P"],
        n_embd=dims["D"],
        n_layer=dims["L"],
        n_head=dims["H"],
        n_inner=dims["I"],
        activation_function="gelu_new",
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    L = dims["L"]
    D = dims["D"]

    # Fused QKV weight (L, D, 3D) and bias (L, 1, 3D) get pre-split column-wise
    # into three (L, D, D) / (L, 1, D) tensors so the DSL leaf can do three
    # independent linears — see reference.get_inputs for rationale.
    c_attn_w_stacked = _stack_gpt2_block(sd, L, "transformer.h.{l}.attn.c_attn.weight")
    c_attn_b_stacked = _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.attn.c_attn.bias")
    c_attn_w_q, c_attn_w_k, c_attn_w_v = (t.contiguous() for t in c_attn_w_stacked.split(D, dim=-1))
    c_attn_b_q, c_attn_b_k, c_attn_b_v = (t.contiguous() for t in c_attn_b_stacked.split(D, dim=-1))

    out = {
        "input_ids": None,    # filled after weights, to match reference's RNG order
        "causal_mask": None,  # filled after weights, also RNG-free but kept adjacent to input_ids
        "wte": sd["transformer.wte.weight"].clone().contiguous(),
        "wpe": sd["transformer.wpe.weight"].clone().contiguous(),
        # LN affine + matmul biases stored as (L, 1, D) so per-layer [i]
        # slices stay ≥2-D. Matrix weights are stacked as-is.
        "block_ln1_w": _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.ln_1.weight"),
        "block_ln1_b": _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.ln_1.bias"),
        "block_c_attn_w_q": c_attn_w_q,
        "block_c_attn_w_k": c_attn_w_k,
        "block_c_attn_w_v": c_attn_w_v,
        "block_c_attn_b_q": c_attn_b_q,
        "block_c_attn_b_k": c_attn_b_k,
        "block_c_attn_b_v": c_attn_b_v,
        "block_c_proj_w": _stack_gpt2_block(sd, L, "transformer.h.{l}.attn.c_proj.weight"),
        "block_c_proj_b": _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.attn.c_proj.bias"),
        "block_ln2_w": _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.ln_2.weight"),
        "block_ln2_b": _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.ln_2.bias"),
        "block_mlp_fc_w": _stack_gpt2_block(sd, L, "transformer.h.{l}.mlp.c_fc.weight"),
        "block_mlp_fc_b": _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.mlp.c_fc.bias"),
        "block_mlp_proj_w": _stack_gpt2_block(sd, L, "transformer.h.{l}.mlp.c_proj.weight"),
        "block_mlp_proj_b": _stack_gpt2_block_vec(sd, L, "transformer.h.{l}.mlp.c_proj.bias"),
        # Final LN params reshaped to (1, D) so offchip_load (>=2-D underlying)
        # can consume them. The reference squeezes back to 1-D for F.layer_norm.
        "ln_f_w": sd["transformer.ln_f.weight"].clone().contiguous().unsqueeze(0),
        "ln_f_b": sd["transformer.ln_f.bias"].clone().contiguous().unsqueeze(0),
    }
    # input_ids draws last, after all weights — see reference.get_inputs.
    out["input_ids"] = torch.randint(0, dims["V"], (dims["B"], dims["T"]))

    # Additive causal mask: shape (T, T), 0 on/below diagonal, -inf above.
    # RNG-free; built here so the STeP impl can offchip-load it instead of
    # fabricating a tensor inside Model.forward.
    T = dims["T"]
    causal_mask = torch.zeros(T, T, dtype=torch.float32)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    out["causal_mask"] = causal_mask
    return out


def _stack_bart_block(sd, L, key_template):
    """Matrix-shaped per-layer params: stacked form is ≥3-D, slice ``[i]`` is ≥2-D."""
    return torch.stack(
        [sd[key_template.format(l=l)].detach().clone().contiguous() for l in range(L)],
        dim=0,
    ).contiguous()


def _stack_bart_block_vec(sd, L, key_template):
    """Vector-shaped per-layer params (``(D,)``): pad to ``(L, 1, D)`` so a
    per-layer slice ``[i]`` is ``(1, D)`` — 2-D, as required by StepGenFlow's
    on-chip ``rank >= 2`` invariant. Matmul ``+ bias`` broadcasts unchanged;
    ``F.layer_norm`` callers ``.squeeze(0)`` at use."""
    return torch.stack(
        [
            sd[key_template.format(l=l)].detach().clone().contiguous().unsqueeze(0)
            for l in range(L)
        ],
        dim=0,
    ).contiguous()


@register("kernelbench_bart_large")
def _precompute_kernelbench_bart_large(dims):
    # RNG order MUST match seed_kernels/kernelbench/level4/bart_large/reference.py:
    #   seed -> BartForCausalLM(config)  (consumes all weight RNG internally,
    #                                     including unused encoder_attn + the
    #                                     pre-tying lm_head weight)
    #        -> torch.randint(0, V, (B, T))   for input_ids
    from transformers import BartConfig, BartForCausalLM

    torch.manual_seed(SEED)
    cfg = BartConfig(
        vocab_size=dims["V"],
        d_model=dims["D"],
        encoder_layers=dims["L"],
        decoder_layers=dims["L"],
        encoder_attention_heads=dims["H"],
        decoder_attention_heads=dims["H"],
        encoder_ffn_dim=dims["I"],
        decoder_ffn_dim=dims["I"],
        max_position_embeddings=dims["P"],
        activation_function="gelu",
        scale_embedding=False,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        is_decoder=True,
        pad_token_id=1,
        use_cache=False,
    )
    model = BartForCausalLM(cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    L = dims["L"]

    out = {
        "input_ids": None,    # filled after weights, to match reference's RNG order
        "causal_mask": None,  # RNG-free, kept adjacent to input_ids
        "wte": sd["model.decoder.embed_tokens.weight"].clone().contiguous(),
        # BartLearnedPositionalEmbedding adds +2 to position ids and sizes the
        # table as (max_position_embeddings + 2, D). We keep the full table.
        "wpe": sd["model.decoder.embed_positions.weight"].clone().contiguous(),
        # Layernorm-after-embedding params reshaped to (1, D) so offchip_load
        # (which requires a ≥2-D underlying) can consume them.
        "layernorm_embedding_w": (
            sd["model.decoder.layernorm_embedding.weight"].clone().contiguous().unsqueeze(0)
        ),
        "layernorm_embedding_b": (
            sd["model.decoder.layernorm_embedding.bias"].clone().contiguous().unsqueeze(0)
        ),
        # Self-attn projections — separate q/k/v/out (BART uses nn.Linear, not
        # Conv1D), so no fused-QKV column split is needed.
        "block_self_attn_w_q": _stack_bart_block(sd, L, "model.decoder.layers.{l}.self_attn.q_proj.weight"),
        "block_self_attn_w_k": _stack_bart_block(sd, L, "model.decoder.layers.{l}.self_attn.k_proj.weight"),
        "block_self_attn_w_v": _stack_bart_block(sd, L, "model.decoder.layers.{l}.self_attn.v_proj.weight"),
        "block_self_attn_w_out": _stack_bart_block(sd, L, "model.decoder.layers.{l}.self_attn.out_proj.weight"),
        "block_self_attn_b_q": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.q_proj.bias"),
        "block_self_attn_b_k": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.k_proj.bias"),
        "block_self_attn_b_v": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.v_proj.bias"),
        "block_self_attn_b_out": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.out_proj.bias"),
        "block_self_attn_ln_w": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.self_attn_layer_norm.weight"),
        "block_self_attn_ln_b": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.self_attn_layer_norm.bias"),
        # FFN.
        "block_fc1_w": _stack_bart_block(sd, L, "model.decoder.layers.{l}.fc1.weight"),
        "block_fc1_b": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.fc1.bias"),
        "block_fc2_w": _stack_bart_block(sd, L, "model.decoder.layers.{l}.fc2.weight"),
        "block_fc2_b": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.fc2.bias"),
        # Post-FFN final LN.
        "block_final_ln_w": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.final_layer_norm.weight"),
        "block_final_ln_b": _stack_bart_block_vec(sd, L, "model.decoder.layers.{l}.final_layer_norm.bias"),
    }
    # input_ids draws last, after all weights — see reference.get_inputs.
    out["input_ids"] = torch.randint(0, dims["V"], (dims["B"], dims["T"]))

    # Additive causal mask: shape (T, T), 0 on/below diagonal, -inf above.
    # RNG-free; built here so the STeP impl can offchip-load it instead of
    # fabricating a tensor inside Model.forward.
    T = dims["T"]
    causal_mask = torch.zeros(T, T, dtype=torch.float32)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    out["causal_mask"] = causal_mask
    return out


def _stack_opt_block(sd, L, key_template):
    """Matrix-shaped per-layer params: stacked form is >=3-D, slice ``[i]`` is >=2-D."""
    return torch.stack(
        [sd[key_template.format(l=l)].detach().clone().contiguous() for l in range(L)],
        dim=0,
    ).contiguous()


def _stack_opt_block_vec(sd, L, key_template):
    """Vector-shaped per-layer params (``(D,)``): pad to ``(L, 1, D)`` so a
    per-layer slice ``[i]`` is ``(1, D)`` — 2-D, as required by StepGenFlow's
    on-chip ``rank >= 2`` invariant. Matmul ``+ bias`` broadcasts unchanged;
    ``F.layer_norm`` callers ``.squeeze(0)`` at use."""
    return torch.stack(
        [
            sd[key_template.format(l=l)].detach().clone().contiguous().unsqueeze(0)
            for l in range(L)
        ],
        dim=0,
    ).contiguous()


@register("kernelbench_opt_1p3b")
def _precompute_kernelbench_opt_1p3b(dims):
    # RNG order MUST match seed_kernels/kernelbench/level4/opt_1p3b/reference.py:
    #   seed -> OPTForCausalLM(config)  (consumes all weight RNG internally,
    #                                    including the pre-tying lm_head weight)
    #        -> torch.randint(0, V, (B, T))   for input_ids
    from transformers import OPTConfig, OPTForCausalLM

    torch.manual_seed(SEED)
    cfg = OPTConfig(
        vocab_size=dims["V"],
        hidden_size=dims["D"],
        ffn_dim=dims["I"],
        num_hidden_layers=dims["L"],
        num_attention_heads=dims["H"],
        max_position_embeddings=dims["P"],
        word_embed_proj_dim=dims["D"],  # OPT-1.3b has no project_in/project_out.
        activation_function="relu",
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        tie_word_embeddings=True,
        use_cache=False,
        pad_token_id=1,
    )
    model = OPTForCausalLM(cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    L = dims["L"]

    out = {
        "input_ids": None,    # filled after weights, to match reference's RNG order
        "causal_mask": None,  # RNG-free, kept adjacent to input_ids
        "wte": sd["model.decoder.embed_tokens.weight"].clone().contiguous(),
        # OPTLearnedPositionalEmbedding sizes the table as
        # (max_position_embeddings + 2, D) for the +2 offset baked into its forward.
        "wpe": sd["model.decoder.embed_positions.weight"].clone().contiguous(),
        # Per-layer self-attention PRE-norm (do_layer_norm_before=True).
        "block_self_attn_ln_w": _stack_opt_block_vec(
            sd, L, "model.decoder.layers.{l}.self_attn_layer_norm.weight"
        ),
        "block_self_attn_ln_b": _stack_opt_block_vec(
            sd, L, "model.decoder.layers.{l}.self_attn_layer_norm.bias"
        ),
        # OPT uses nn.Linear (weight shape (out, in)); q/k/v/out are stored
        # separately (no fused-QKV like GPT-2's Conv1D).
        "block_self_attn_w_q": _stack_opt_block(sd, L, "model.decoder.layers.{l}.self_attn.q_proj.weight"),
        "block_self_attn_w_k": _stack_opt_block(sd, L, "model.decoder.layers.{l}.self_attn.k_proj.weight"),
        "block_self_attn_w_v": _stack_opt_block(sd, L, "model.decoder.layers.{l}.self_attn.v_proj.weight"),
        "block_self_attn_w_out": _stack_opt_block(sd, L, "model.decoder.layers.{l}.self_attn.out_proj.weight"),
        "block_self_attn_b_q": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.q_proj.bias"),
        "block_self_attn_b_k": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.k_proj.bias"),
        "block_self_attn_b_v": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.v_proj.bias"),
        "block_self_attn_b_out": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.out_proj.bias"),
        # Per-layer FFN PRE-norm (this is OPT's per-layer ``final_layer_norm`` —
        # it sits between attention and FFN, distinct from the decoder-level
        # final LN extracted below).
        "block_final_ln_w": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.final_layer_norm.weight"),
        "block_final_ln_b": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.final_layer_norm.bias"),
        # FFN.
        "block_fc1_w": _stack_opt_block(sd, L, "model.decoder.layers.{l}.fc1.weight"),
        "block_fc1_b": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.fc1.bias"),
        "block_fc2_w": _stack_opt_block(sd, L, "model.decoder.layers.{l}.fc2.weight"),
        "block_fc2_b": _stack_opt_block_vec(sd, L, "model.decoder.layers.{l}.fc2.bias"),
        # Decoder-level final LayerNorm (applied after the last decoder layer
        # when do_layer_norm_before=True and not _remove_final_layer_norm).
        # Stored as (1, D) so offchip_load (>=2-D underlying) can consume it.
        "decoder_ln_w": sd["model.decoder.final_layer_norm.weight"].clone().contiguous().unsqueeze(0),
        "decoder_ln_b": sd["model.decoder.final_layer_norm.bias"].clone().contiguous().unsqueeze(0),
    }
    # input_ids draws last, after all weights — see reference.get_inputs.
    out["input_ids"] = torch.randint(0, dims["V"], (dims["B"], dims["T"]))

    # Additive causal mask: shape (T, T), 0 on/below diagonal, -inf above.
    # RNG-free; built here so the STeP impl can offchip-load it instead of
    # fabricating a tensor inside Model.forward.
    T = dims["T"]
    causal_mask = torch.zeros(T, T, dtype=torch.float32)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    out["causal_mask"] = causal_mask
    return out


# ---------------------------------------------------------------------------
# kernelbench_reformer_enwik8 — KernelBench Level 4 / Problem 13.
# ---------------------------------------------------------------------------

def _stack_reformer_block(sd, L, key_template):
    """Matrix-shaped per-layer params: stacked form is >=3-D, slice ``[i]`` is >=2-D."""
    return torch.stack(
        [sd[key_template.format(l=l)].detach().clone().contiguous() for l in range(L)],
        dim=0,
    ).contiguous()


def _stack_reformer_block_vec(sd, L, key_template):
    """Vector-shaped per-layer params (``(D,)``): pad to ``(L, 1, D)`` so a
    per-layer slice ``[i]`` is ``(1, D)`` — 2-D, as required by StepGenFlow's
    on-chip ``rank >= 2`` invariant. ``F.layer_norm`` callers ``.squeeze(0)`` at use."""
    return torch.stack(
        [
            sd[key_template.format(l=l)].detach().clone().contiguous().unsqueeze(0)
            for l in range(L)
        ],
        dim=0,
    ).contiguous()


def _stack_reformer_qkv(sd, L, attn_types):
    """Build per-layer (L, D, D) q_w, k_w, v_w tensors. Local layers carry
    distinct ``query/key/value``; LSH layers share ``query_key`` for both q_w
    and k_w slots (the forward branches on attn_type to decide whether to
    length-normalise K). Duplication keeps the per-layer tensor signature
    uniform without changing the math."""
    q_list, k_list, v_list = [], [], []
    for l in range(L):
        if attn_types[l] == "local":
            q = sd[f"reformer.encoder.layers.{l}.attention.self_attention.query.weight"]
            k = sd[f"reformer.encoder.layers.{l}.attention.self_attention.key.weight"]
            v = sd[f"reformer.encoder.layers.{l}.attention.self_attention.value.weight"]
        else:  # lsh
            qk = sd[f"reformer.encoder.layers.{l}.attention.self_attention.query_key.weight"]
            v = sd[f"reformer.encoder.layers.{l}.attention.self_attention.value.weight"]
            q = qk
            k = qk
        q_list.append(q.detach().clone().contiguous())
        k_list.append(k.detach().clone().contiguous())
        v_list.append(v.detach().clone().contiguous())
    return (
        torch.stack(q_list, dim=0).contiguous(),
        torch.stack(k_list, dim=0).contiguous(),
        torch.stack(v_list, dim=0).contiguous(),
    )


@register("kernelbench_reformer_enwik8")
def _precompute_kernelbench_reformer_enwik8(dims):
    # RNG order MUST match
    # seed_kernels/kernelbench/level4/reformer_enwik8/reference.py:
    #   seed -> ReformerModelWithLMHead(config)  (consumes all weight RNG
    #                                             internally)
    #        -> torch.randint(0, V, (B, T))      for input_ids
    from transformers import ReformerConfig, ReformerModelWithLMHead

    attn_layers = ["local" if c == "l" else "lsh" for c in dims["ATTN_LAYERS"]]
    assert len(attn_layers) == dims["L"]
    assert dims["AE0"] + dims["AE1"] == dims["D"]
    assert dims["AX0"] * dims["AX1"] == dims["MAX_POS"]

    torch.manual_seed(SEED)
    cfg = ReformerConfig(
        vocab_size=dims["V"],
        hidden_size=dims["D"],
        num_attention_heads=dims["H"],
        attention_head_size=dims["D_HEAD"],
        feed_forward_size=dims["FF"],
        num_hidden_layers=dims["L"],
        attn_layers=attn_layers,
        axial_pos_embds=True,
        axial_pos_shape=(dims["AX0"], dims["AX1"]),
        axial_pos_embds_dim=(dims["AE0"], dims["AE1"]),
        max_position_embeddings=dims["MAX_POS"],
        num_buckets=dims.get("NUM_BUCKETS", 8),
        num_hashes=1,
        lsh_attn_chunk_length=dims["LSH_CL"],
        local_attn_chunk_length=dims["LOC_CL"],
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        chunk_size_lm_head=0,
        chunk_size_feed_forward=0,
        hash_seed=SEED,
        is_decoder=True,
        use_cache=False,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        lsh_attention_probs_dropout_prob=0.0,
        local_attention_probs_dropout_prob=0.0,
        hidden_act="relu",
        layer_norm_eps=1e-12,
        axial_norm_std=1.0,
        tie_word_embeddings=False,
    )
    model = ReformerModelWithLMHead(cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    L = dims["L"]

    # Axial position embeddings reshaped to drop their size-1 axes — see
    # reference.get_inputs for the (T <= AX1) eval-mode derivation.
    axial_pos_0 = (
        sd["reformer.embeddings.position_embeddings.weights.0"][0, 0]
        .clone().contiguous().unsqueeze(0)
    )  # (1, AE0)
    axial_pos_1 = (
        sd["reformer.embeddings.position_embeddings.weights.1"][0]
        .clone().contiguous()
    )  # (AX1, AE1)

    block_attn_q_w, block_attn_k_w, block_attn_v_w = _stack_reformer_qkv(sd, L, attn_layers)

    out = {
        "input_ids": None,  # filled after weights, to match reference's RNG order
        "axial_pos_0": axial_pos_0,
        "axial_pos_1": axial_pos_1,
        "wte": sd["reformer.embeddings.word_embeddings.weight"].clone().contiguous(),
        # Per-layer attention.
        "block_attn_ln_w": _stack_reformer_block_vec(sd, L, "reformer.encoder.layers.{l}.attention.layer_norm.weight"),
        "block_attn_ln_b": _stack_reformer_block_vec(sd, L, "reformer.encoder.layers.{l}.attention.layer_norm.bias"),
        "block_attn_q_w": block_attn_q_w,
        "block_attn_k_w": block_attn_k_w,
        "block_attn_v_w": block_attn_v_w,
        "block_attn_out_w": _stack_reformer_block(sd, L, "reformer.encoder.layers.{l}.attention.output.dense.weight"),
        # Per-layer FFN.
        "block_ff_ln_w": _stack_reformer_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.layer_norm.weight"),
        "block_ff_ln_b": _stack_reformer_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.layer_norm.bias"),
        "block_ff_dense_w": _stack_reformer_block(sd, L, "reformer.encoder.layers.{l}.feed_forward.dense.dense.weight"),
        "block_ff_dense_b": _stack_reformer_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.dense.dense.bias"),
        "block_ff_out_w": _stack_reformer_block(sd, L, "reformer.encoder.layers.{l}.feed_forward.output.dense.weight"),
        "block_ff_out_b": _stack_reformer_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.output.dense.bias"),
        # Final encoder LN (over 2*D) — stored as (1, 2D) so offchip_load
        # (>=2-D underlying) can consume it. Reference squeezes back at use.
        "enc_ln_w": sd["reformer.encoder.layer_norm.weight"].clone().contiguous().unsqueeze(0),
        "enc_ln_b": sd["reformer.encoder.layer_norm.bias"].clone().contiguous().unsqueeze(0),
        # LM head decoder (bias=False; lm_head.bias is allocated by HF but
        # never used in forward).
        "lm_head_w": sd["lm_head.decoder.weight"].clone().contiguous(),
    }
    out["input_ids"] = torch.randint(0, dims["V"], (dims["B"], dims["T"]))
    return out


# ---------------------------------------------------------------------------
# Examples (origin: example) — self-contained didactic kernels under examples/
# ---------------------------------------------------------------------------

@register("kv_append_load")
def _precompute_kv_append_load(dims):
    # Delegates to examples/kv/precompute.py so the example folder stays a
    # single source of truth and remains runnable via the standalone smoke
    # test in HANDOFF.md. Loaded by file path (examples/ is not a package).
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent / "examples" / "kv" / "precompute.py"
    if not hasattr(_precompute_kv_append_load, "_mod"):
        spec = importlib.util.spec_from_file_location("_examples_kv_precompute", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _precompute_kv_append_load._mod = mod
    return _precompute_kv_append_load._mod.precompute(dims)
