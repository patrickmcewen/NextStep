"""STeP implementation: End-to-end transformer layer (attention + MoE).

Builds the full static Mixtral/Qwen transformer layer graph:
  RMSNorm -> QKV -> Attention -> O-proj -> ResAdd -> RMSNorm -> MoE -> ResAdd

Extracted from step_tl/end_to_end/static_baseline.py::run_static_baseline.
"""
import sys
import random
from pathlib import Path

# Derive step_tl root from step_py's location (step_py lives in step_tl/src/step_py/)
import step_py as _sp
_STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
if _STEP_TL_ROOT not in sys.path:
    sys.path.insert(0, _STEP_TL_ROOT)

from end_to_end import AttentionMetric, MoeMetric
from end_to_end.model_configs import (
    Mixtral8x7B, SmallerMixtral8x7B, Qwen30B, SmallerQwen30B,
)
from end_to_end.attention.qkv_gen import qkv_gen, rms_norm
from end_to_end.attention.qkv_gen_attention import qkv_gen_attention
from end_to_end.attention.static_parallel import build_static_par
from end_to_end.conversion.o_proj import reshape_o_proj
from end_to_end.conversion.static_par_input_reassemble import input_reassemble
from end_to_end.moe import build_and_calculate_static_moe_metric
from utils.moe import topk_to_multihot, topk_to_onehot

SEED = 42

# Map (model_name, batch) -> (iter, layer) for expert routing file lookup
_EXPERT_ROUTING = {
    ("mixtral", 64): (8, 10),
    ("mixtral", 1024): (19, 9),
    ("qwen", 64): (32, 12),
    ("qwen", 1024): (22, 16),
}


def build_graph(dims):
    model_name = dims["model_name"]
    tile_N = dims["tile_N"]
    tile_F = dims.get("tile_F", 48)
    batch = dims.get("batch", 64)
    scale_seq = dims.get("scale_seq", 1)
    is_small = dims.get("is_small", False)
    par_factor = dims.get("par_factor", 4)
    stdev = dims["stdev"]
    start = dims["start"]
    end = dims["end"]

    torch.manual_seed(5)
    random.seed(42)

    # ---- Model config ----
    if model_name == "mixtral":
        model_config = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        model_config = SmallerQwen30B() if is_small else Qwen30B()
    else:
        assert False, f"Unknown model_name: {model_name}"

    # ---- Result dict (needed by sub-functions for metric accumulation) ----
    result_dict = {
        "attention_metric": AttentionMetric(),
        "moe_metric": MoeMetric(),
        "tracked_by_sim": [],
        "model_name": model_config.__class__.__name__,
        "batch": batch,
    }

    # ---- Sim config ----
    unit_comp = 1024
    sim_config = {
        "metadata_fifo_depth": 16,
        "cache_write_back_fifo_depth": 4,
        "residual_fifo_depth": 32,
        "par_dispatch": 4,
        "mock_bf16": True,
        "channel_dict": {},
        "compute_bw": {
            "matmul": unit_comp, "exp": unit_comp, "multv": unit_comp,
            "tile_wise_rowsum": unit_comp, "intra_tile_rowsum": unit_comp,
            "softmax_div": unit_comp, "retile": unit_comp,
        },
        "gate_compute_bw": 1024,
        "up_compute_bw": 1024,
        "act_fn_compute_bw": 1024,
        "mult_compute_bw": 1024,
        "down_compute_bw": 1024,
        "weight_scale_compute_bw": 1024,
        "accum_compute_bw": 1024,
        "expert_counts": [],  # populated below
    }

    # ---- Cache config ----
    maxN = 4096 * scale_seq
    tile_seq_len = 32
    cache_row_offset_tiled = maxN // tile_seq_len
    tile_b = 16
    hidden_dim_tile = 64

    # ---- Input tensors ----
    input_tensor = torch.randn(batch, model_config.hidden_dim)
    q_proj = torch.randn(model_config.hidden_dim, model_config.num_heads * model_config.head_dim)
    k_proj = torch.randn(model_config.hidden_dim, model_config.num_kv_heads * model_config.head_dim)
    v_proj = torch.randn(model_config.hidden_dim, model_config.num_kv_heads * model_config.head_dim)
    cos = torch.randn(batch, 1, model_config.head_dim)
    sin = torch.randn(batch, 1, model_config.head_dim)

    # ---- KV cache ----
    k_cache = torch.zeros(batch, maxN, model_config.num_kv_heads, model_config.head_dim)
    v_cache = torch.zeros(batch, maxN, model_config.num_kv_heads, model_config.head_dim)

    # ---- Expert routing ----
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
    expert_counts = torch.bincount(expert_indices.flatten(), minlength=model_config.n_routed_experts)
    sim_config["expert_counts"] = expert_counts.tolist()

    expert_multihot = topk_to_multihot(expert_indices, model_config.n_routed_experts)
    expert_onehot = topk_to_onehot(expert_indices, model_config.n_routed_experts)
    expert_weights = torch.softmax(torch.randn(batch, model_config.n_activated_experts), dim=-1)

    # ---- Expert weight matrices ----
    w_gate_list = [
        torch.nn.Linear(model_config.dim, model_config.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(model_config.n_routed_experts)
    ]
    w_up_list = [
        torch.nn.Linear(model_config.dim, model_config.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(model_config.n_routed_experts)
    ]
    w_down_list = [
        torch.nn.Linear(model_config.moe_inter_dim, model_config.dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(model_config.n_routed_experts)
    ]

    # ---- Trace data (KV cache lengths per batch element) ----
    assert batch == end - start + 1, f"batch={batch} != end-start+1={end - start + 1}"
    trace_path = (
        Path(_STEP_TL_ROOT)
        / f"dynamic_par/azure_trace/b{batch}"
        / f"conv_stdev{stdev:04d}_{start:04d}_{end:04d}.npy"
    )
    assert trace_path.exists(), f"Trace file not found: {trace_path}"
    num_token_list = np.load(str(trace_path)).astype(np.int64).tolist()
    num_token_list = [x * scale_seq for x in num_token_list]

    seq_len_tiled = torch.tensor(
        [(x + 1 + tile_seq_len - 1) // tile_seq_len for x in num_token_list]
    )
    offset = torch.tensor([x % tile_seq_len for x in num_token_list])

    # Initialize KV cache with random data up to each element's sequence length
    for i in range(batch):
        k_cache[i, :num_token_list[i]] = torch.randn(
            num_token_list[i], model_config.num_kv_heads, model_config.head_dim
        )
        v_cache[i, :num_token_list[i]] = torch.randn(
            num_token_list[i], model_config.num_kv_heads, model_config.head_dim
        )

    # ==================== Build the graph ====================
    step_graph = Graph()

    # Input loads: 4 parallel chunks of [16, HID]
    par_inputs = [
        LinearOffChipLoad(
            underlying=input_tensor[i * 16 : (i + 1) * 16, :],
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=tile_b,
            tile_col=model_config.hidden_dim,
            par_dispatch=sim_config["par_dispatch"],
            mock_bf16=sim_config["mock_bf16"],
        )
        for i in range(par_factor)
    ]

    # [RMS Norm] [B,HID] -> [B,HID]
    input_layer_norm = rms_norm(step_graph, par_inputs, model_config, sim_config)

    # [QKV Gen] [B,HID] -> (q,k,v)
    q, k, v = qkv_gen(
        step_graph=step_graph, inputs=input_layer_norm,
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        cos=cos, sin=sin,
        model_config=model_config, sim_config=sim_config,
    )

    # [QKV -> Attention format]
    merged_q, merged_k, merged_v = qkv_gen_attention(step_graph, q, k, v, model_config)

    # [Static Parallel Attention]
    par_atten_outputs = build_static_par(
        step_graph=step_graph, model_config=model_config,
        query=merged_q, key=merged_k, value=merged_v,
        k_cache=k_cache.flatten(0, -2),
        v_cache=v_cache.flatten(0, -2),
        idx=torch.arange(batch, dtype=torch.int64).to(torch.uint64),
        seq_len=seq_len_tiled.to(torch.uint64),
        offset=offset.to(torch.uint64),
        par_factor=4,
        metadata_fifo_depth=sim_config["metadata_fifo_depth"],
        cache_write_back_fifo_depth=sim_config["cache_write_back_fifo_depth"],
        cache_row_offset_tiled=cache_row_offset_tiled,
        tile_N=tile_seq_len,
        compute_bw=sim_config["compute_bw"],
        mock_bf16=sim_config["mock_bf16"],
        par_dispatch=sim_config["par_dispatch"],
        channel_dict=sim_config["channel_dict"],
    )

    # [Collect parallel attention outputs]
    collected_inputs = StaticReassemble(
        graph=step_graph, inputs=par_atten_outputs,
        merge_rank=1, switch_cycles=[1] * 4,
    )

    # [O-proj]
    o_proj = reshape_o_proj(
        step_graph, collected_inputs, model_config, sim_config, hidden_dim_tile
    )

    # [Residual add: o_proj + input]
    reordered_inputs = input_reassemble(step_graph, par_inputs, model_config, sim_config)
    sim_config["channel_dict"][reordered_inputs.instance_id] = sim_config["residual_fifo_depth"]

    res_add_0_src = BinaryMap(
        graph=step_graph, in1=o_proj, in2=reordered_inputs,
        fn=Add(), write_back_mu=False,
        compute_bw=sim_config["compute_bw"]["exp"],
    )

    res_add_0 = (Broadcast(step_graph, res_add_0_src, 1), 0)
    sim_config["channel_dict"][res_add_0[0].instance_id] = sim_config["residual_fifo_depth"]

    # [Post-attention RMS Norm]
    post_attention_layer_norm = rms_norm(
        step_graph, [res_add_0_src], model_config, sim_config,
    )[0]

    # [Prepare for MoE: retile + reshape]
    prepare_moe = ReshapePadStream(
        graph=step_graph,
        input=RetileStreamify(graph=step_graph, input=post_attention_layer_norm, split_row=True),
        chunk_size=batch,
        reshape_rank=0,
        write_back_mu=False,
        pad_fn=init_fn.Zero(shape=(1, model_config.hidden_dim), dtype=Float32()),
        have_pad_stream=False,
    )

    # [MoE block]
    moe_output = build_and_calculate_static_moe_metric(
        step_graph, prepare_moe, res_add_0,
        expert_multihot, expert_onehot, expert_weights,
        w_gate_list, w_up_list, w_down_list,
        model_config, sim_config, batch, tile_N, tile_F, result_dict,
    )

    # [Output store]
    output = OffChipStore(
        graph=step_graph,
        input=moe_output,
        par_dispatch=sim_config["par_dispatch"],
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
