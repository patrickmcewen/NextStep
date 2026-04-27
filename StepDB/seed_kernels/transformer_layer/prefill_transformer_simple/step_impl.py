"""STeP implementation: simple prefill transformer layer (attention + MoE).

The full pre-attention pipeline + self-attention run in the dataflow graph,
using row-streaming sdpa with K and V replayed across the seq dim via
Broadcast(stream, S) + StaticReassemble(merge_rank=outermost) (the cycle
pattern that LinearOffChipLoad(stride=(0, 1)) provides for an off-chip
cache, but driven from an upstream stream):

    Load input -> RMSNorm
        -> per-head (Q proj | K proj | V proj)
        -> per-head Q/K RMSNorm
        -> per-head RoPE (rotate_half-style)
        -> per-head row-streaming sdpa (sdpa_core pattern with K/V replay)
        -> per-head O-proj slice + sum across heads
        -> ResAdd
        -> post-attention RMSNorm
        -> routed top-k MoE
        -> ResAdd -> OffChipStore

Routing tensors (expert_weights, expert_multihot, expert_onehot) come from
precompute.py because they depend on float64 attention output to match
the reference exactly (float32 noise can flip topk decisions on boundary
logits).
"""
import sys
from pathlib import Path

import step_py as _sp
_STEP_TL_ROOT = Path(_sp.__file__).resolve().parent.parent.parent
_STEPDB_ROOT = _STEP_TL_ROOT.parent / "StepDB"
for _p in (str(_STEP_TL_ROOT), str(_STEPDB_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from end_to_end.model_configs import (
    Mixtral8x7B, SmallerMixtral8x7B, Qwen30B, SmallerQwen30B,
)
from precompute import precompute_tensors


def _model_config(model_name, is_small):
    if model_name == "mixtral":
        return SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    if model_name == "qwen":
        return SmallerQwen30B() if is_small else Qwen30B()
    assert False, f"Unknown model_name: {model_name!r}"


# ---------------------------------------------------------------------------
# RMSNorm sub-pipeline
# ---------------------------------------------------------------------------

def _rms_norm_chain(graph, in_stream, K_features, eps=1e-6):
    in_b = Broadcast(graph, in_stream, 2)
    x_squared = UnaryMap(
        graph=graph, input=(in_b, 0), fn=Square(),
        write_back_mu=False, compute_bw=1024,
    )
    scaled = UnaryMap(
        graph=graph, input=x_squared,
        fn=MulImmediate(constant=1.0 / K_features),
        write_back_mu=False, compute_bw=1024,
    )
    row_sum = UnaryMap(
        graph=graph, input=scaled, fn=RowWiseSum(),
        write_back_mu=False, compute_bw=1024,
    )
    add_eps = UnaryMap(
        graph=graph, input=row_sum, fn=AddImmediate(constant=eps),
        write_back_mu=False, compute_bw=1024,
    )
    rsqrt = UnaryMap(
        graph=graph, input=add_eps, fn=Rsqrt(),
        write_back_mu=False, compute_bw=1024,
    )
    return BinaryMap(
        graph=graph, in1=(in_b, 1), in2=rsqrt,
        fn=Mul(), write_back_mu=False, compute_bw=1024,
    )


# ---------------------------------------------------------------------------
# rotate_half / RoPE  (mirrors seed_kernels/pre_attention/rope)
# Input: stream (1, batch, 1) of tile (num_heads, head_dim).
# ---------------------------------------------------------------------------

def _rotate_half(graph, input_stream, batch, num_heads, head_dim):
    split_half = RetileStreamify(
        graph=graph, input=input_stream,
        split_row=False, chunk=head_dim // 2,
    )
    flat = Flatten(graph=graph, input=split_half, min_rank=0, max_rank=2)
    splitted = Parallelize(
        graph=graph, input=flat,
        parallelize_rank=0, num_consumers=2, switch_cycles=[1, 1],
    )
    neg_second = UnaryMap(
        graph=graph, input=(splitted, 1),
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False, compute_bw=1024,
    )
    concat = StaticReassemble(
        graph=graph, inputs=[neg_second, (splitted, 0)],
        merge_rank=0, switch_cycles=[1, 1],
    )
    reshaped = ReshapePadStream(
        graph=graph, input=concat,
        chunk_size=2, reshape_rank=0,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )
    full_tile = Accum(
        graph=graph, input=reshaped,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(num_heads, head_dim)),
        fn=RetileCol(),
        init_fn=Empty(shape=(num_heads, 0), dtype=Float32()),
        accum_rank=1, write_back_mu=True, compute_bw=1024,
    )
    restored_inner = ReshapePadStream(
        graph=graph, input=full_tile,
        chunk_size=1, reshape_rank=0,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )
    restored_outer = ReshapePadStream(
        graph=graph, input=restored_inner,
        chunk_size=batch, reshape_rank=1,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )
    return restored_outer


def _rope(graph, x_stream, cos_stream, sin_stream, batch, num_heads, head_dim):
    x_b = Broadcast(graph, x_stream, 2)
    rot = _rotate_half(graph, (x_b, 0), batch, num_heads, head_dim)
    x_cos = BinaryMap(
        graph=graph, in1=(x_b, 1), in2=cos_stream,
        fn=Mul(), write_back_mu=False, compute_bw=1024,
    )
    x_sin = BinaryMap(
        graph=graph, in1=rot, in2=sin_stream,
        fn=Mul(), write_back_mu=False, compute_bw=1024,
    )
    return BinaryMap(
        graph=graph, in1=x_cos, in2=x_sin,
        fn=Add(), write_back_mu=False, compute_bw=1024,
    )


# ---------------------------------------------------------------------------
# Cycle-replay a stream: turn (1, S) of tile (1, head_dim) into
# (1, S, S) where for each outer position the inner iterates the full
# stream sequence.  Implemented as Broadcast(stream, S) + StaticReassemble
# (merge_rank=outermost) + Promote.  This is the stream analogue of the
# off-chip K-replay LinearOffChipLoad(stride=(0, 1), out_shape_tiled=(S, S)).
# ---------------------------------------------------------------------------

def _cycle_replay(graph, in_stream, S):
    b = Broadcast(graph, in_stream, S)
    copies = [(b, i) for i in range(S)]
    merged = StaticReassemble(
        graph=graph, inputs=copies, merge_rank=2,
        switch_cycles=[1] * S,
    )
    return Promote(graph, merged, promote_rank=2)


# ---------------------------------------------------------------------------
# Row-streaming sdpa per head, consuming Q/K/V upstream streams
# (each stream (1, S, 1) of tile (1, head_dim) post-RoPE / post-projection).
# Output: stream (1, S) of tile (1, head_dim).
# ---------------------------------------------------------------------------

def _sdpa_streamed(graph, Q_stream, K_stream, V_stream, S, head_dim):
    Q_flat = Flatten(graph=graph, input=Q_stream, min_rank=0, max_rank=1)
    q_repeated = RepeatStatic(graph=graph, input=Q_flat, repeat_factor=S)

    K_flat = Flatten(graph=graph, input=K_stream, min_rank=0, max_rank=1)
    K_replayed = _cycle_replay(graph, K_flat, S)
    V_flat = Flatten(graph=graph, input=V_stream, min_rank=0, max_rank=1)
    V_replayed = _cycle_replay(graph, V_flat, S)

    qkt = BinaryMap(
        graph=graph, in1=q_repeated, in2=K_replayed,
        fn=Matmul(weight_transposed=True),
        write_back_mu=False, compute_bw=1024,
    )
    # Numerically-stable softmax: subtract per-row max before exp to avoid
    # float32 overflow in `Exp(qkt)`.  Mirrors the
    # sdpa_core_max DSL pattern: accum_max -> promote -> expand_ref.
    # qkt has stream (1, S, S) tile (1, 1).  RetileStreamify is unnecessary
    # because tile_n is already 1.
    #   Accum(fn=accum_fn.Max(), accum_rank=1) reduces the innermost stream
    #     dim -> stream (1, S) tile (1, 1)
    #   Promote(promote_rank=1) adds a trailing stream-1 dim
    #     -> stream (1, S, 1) tile (1, 1)
    #   ExpandRef(ref=qkt, expand_rank=1) broadcasts back over qkt's inner
    #     S dim -> stream (1, S, S) tile (1, 1)
    qkt_b = Broadcast(graph, qkt, 3)
    row_max = Accum(
        graph=graph, input=(qkt_b, 0),
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(1, 1)),
        fn=accum_fn.Max(),
        init_fn=Zero(shape=(1, 1), dtype=Float32()),
        accum_rank=1, write_back_mu=False, compute_bw=1024,
    )
    neg_row_max = UnaryMap(
        graph=graph, input=row_max,
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False, compute_bw=1024,
    )
    # Promote inserts the new 1 at position `len(shape) - promote_rank`, so
    # promote_rank=0 inserts at the END: (1, S) -> (1, S, 1).
    neg_row_max_promoted = Promote(
        graph=graph, input=neg_row_max, promote_rank=0,
    )
    neg_row_max_expanded = ExpandRef(
        graph=graph, input=neg_row_max_promoted,
        ref=(qkt_b, 2), expand_rank=1,
    )
    qkt_shifted = BinaryMap(
        graph=graph, in1=(qkt_b, 1), in2=neg_row_max_expanded,
        fn=Add(), write_back_mu=False, compute_bw=1024,
    )
    exp_qkt = UnaryMap(
        graph=graph, input=qkt_shifted, fn=Exp(),
        write_back_mu=False, compute_bw=1024,
    )
    exp_b = Broadcast(graph, exp_qkt, 2)
    num = BinaryMapAccum(
        graph=graph, in1=(exp_b, 0), in2=V_replayed,
        fn=MapAccumMatmul(),
        init_fn=Zero(shape=(1, head_dim), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=1024,
    )
    exp_rowsum = Accum(
        graph=graph, input=(exp_b, 1),
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(1, 1)),
        fn=AccumAdd(),
        init_fn=Zero(shape=(1, 1), dtype=Float32()),
        accum_rank=1, write_back_mu=False, compute_bw=1024,
    )
    return BinaryMap(
        graph=graph, in1=num, in2=exp_rowsum,
        fn=Div(), write_back_mu=False, compute_bw=1024,
    )


# ---------------------------------------------------------------------------
# MoE block (verbatim from moe_routed/step_impl.py pattern)
# ---------------------------------------------------------------------------

def _moe_block(
    graph, x_stream, x_residual_stream,
    w_gate_list, w_up_list, w_down_list,
    expert_weights_t, expert_multihot, expert_onehot,
    S, D, F_dim, n_experts, n_active, tile_n, tile_f,
):
    feature_select_gen = SelectGen(
        is_multihot=True, tensor=expert_multihot, n=n_experts,
    )
    weight_select_gen = SelectGen(
        is_multihot=True, tensor=expert_onehot, n=n_experts,
    )
    feature_select_gen_reassemble = SelectGen(
        is_multihot=True, tensor=expert_multihot, n=n_experts,
    )

    partitioned = FlatPartition(
        graph, x_stream, feature_select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(n_experts)],
        write_back_mu=False,
        num_consumers=n_experts,
    )

    expert_feature_streams = []
    for i in range(n_experts):
        reshaped = Reshape(
            graph, (partitioned, i), tile_n, 0,
            write_back_mu=False, add_outer_dim=True,
            pad_fn=Zero(shape=(1, D), dtype=Float32()),
        )
        flattened = Flatten(graph, reshaped, min_rank=1, max_rank=2)
        tiled = Accum(
            graph, flattened,
            Tile(tile_dtype=Float32(), shape=(tile_n, D)),
            RetileRow(),
            Empty(shape=(0, D), dtype=Float32()),
            1, False, 1024,
        )
        expert_feature_streams.append(tiled)

    repeated_features = [
        RepeatStatic(graph, expert_feature_streams[i],
                     repeat_factor=F_dim // tile_f)
        for i in range(n_experts)
    ]

    up_loads = [
        LinearOffChipLoadRef(
            graph=graph, ref=expert_feature_streams[i],
            underlying=w_up_list[i],
            stride=(1, 1),
            out_shape_tiled=(F_dim // tile_f, 1),
            tile_row=D, tile_col=tile_f, par_dispatch=4,
        )
        for i in range(n_experts)
    ]
    ready_up_loads = [
        Flatten(graph=graph, input=up_loads[i], min_rank=0, max_rank=1)
        for i in range(n_experts)
    ]
    up_features = [
        BinaryMap(graph, repeated_features[i], ready_up_loads[i],
                  Matmul(weight_transposed=False), False, 1024)
        for i in range(n_experts)
    ]

    gate_loads = [
        LinearOffChipLoadRef(
            graph=graph, ref=expert_feature_streams[i],
            underlying=w_gate_list[i],
            stride=(1, 1),
            out_shape_tiled=(F_dim // tile_f, 1),
            tile_row=D, tile_col=tile_f, par_dispatch=4,
        )
        for i in range(n_experts)
    ]
    ready_gate_loads = [
        Flatten(graph=graph, input=gate_loads[i], min_rank=0, max_rank=1)
        for i in range(n_experts)
    ]
    gate_features = [
        BinaryMap(graph, repeated_features[i], ready_gate_loads[i],
                  Matmul(weight_transposed=False), False, 1024)
        for i in range(n_experts)
    ]
    gate_activated = [
        UnaryMap(graph=graph, input=gate_features[i], fn=Silu(),
                 write_back_mu=False, compute_bw=1024)
        for i in range(n_experts)
    ]
    projected = [
        BinaryMap(graph, up_features[i], gate_activated[i],
                  Mul(), False, 1024)
        for i in range(n_experts)
    ]

    down_loads = [
        LinearOffChipLoadRef(
            graph=graph, ref=expert_feature_streams[i],
            underlying=w_down_list[i],
            stride=(1, 1),
            out_shape_tiled=(F_dim // tile_f, 1),
            tile_row=tile_f, tile_col=D, par_dispatch=4,
        )
        for i in range(n_experts)
    ]
    ready_down_loads = [
        Flatten(graph=graph, input=down_loads[i], min_rank=0, max_rank=1)
        for i in range(n_experts)
    ]
    down_features = [
        BinaryMapAccum(
            graph, projected[i], ready_down_loads[i],
            MapAccumMatmul(),
            Zero(shape=(tile_n, D), dtype=Float32()),
            1, False, 1024,
        )
        for i in range(n_experts)
    ]

    retiled = [
        RetileStreamify(graph=graph, input=down_features[i],
                        split_row=True, filter_mask=True)
        for i in range(n_experts)
    ]
    for partitioned_stream, retiled_stream in zip(partitioned.stream_list, retiled):
        dyn_i = partitioned_stream.shape[0]
        retiled_stream.stream.shape = (dyn_i,)

    weights_load = LinearOffChipLoad(
        underlying=expert_weights_t,
        stride=(n_active, 1),
        out_shape_tiled=(S, n_active),
        tile_row=1, tile_col=1, par_dispatch=4,
    )
    expert_weight_streams = FlatPartition(
        graph, weights_load, weight_select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(n_experts)],
        write_back_mu=False,
        num_consumers=n_experts,
    )
    weighted = [
        BinaryMap(graph, (expert_weight_streams, i), retiled[i],
                  Mul(), False, 1024)
        for i in range(n_experts)
    ]

    reassembled = FlatReassemble(
        graph, weighted, feature_select_gen_reassemble,
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(n_experts)],
        write_back_mu=False,
    )
    accumed = Accum(
        graph, reassembled,
        Tile(tile_dtype=Float32(), shape=(1, D)),
        AccumAdd(),
        Zero(shape=(1, D), dtype=Float32()),
        1, False, 1024,
    )
    return BinaryMap(
        graph=graph, in1=accumed, in2=x_residual_stream,
        fn=Add(), write_back_mu=False, compute_bw=1024,
    )


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

def build_graph(dims):
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)
    mc = _model_config(dims["model_name"], is_small)

    tile_n = dims.get("tile_n", min(seq_len, 16))
    tile_f = dims.get("tile_f", min(mc.moe_inter_dim, 16))
    assert seq_len % tile_n == 0 or tile_n >= seq_len, (
        f"seq_len={seq_len} not compatible with tile_n={tile_n}"
    )
    assert mc.moe_inter_dim % tile_f == 0, (
        f"moe_inter_dim={mc.moe_inter_dim} not divisible by tile_f={tile_f}"
    )

    t = precompute_tensors("prefill_transformer_simple", dims)
    input_tensor   = t["input_tensor"]
    q_proj         = t["q_proj"]
    k_proj         = t["k_proj"]
    v_proj         = t["v_proj"]
    cos            = t["cos"]
    sin            = t["sin"]
    o_proj_weight  = t["o_proj_weight"]
    w_gate_list    = t["w_gate_list"]
    w_up_list      = t["w_up_list"]
    w_down_list    = t["w_down_list"]
    expert_weights = t["expert_weights"]
    expert_multihot = t["expert_multihot"]
    expert_onehot  = t["expert_onehot"]

    H = mc.hidden_dim
    NHQ = mc.num_heads
    NHKV = mc.num_kv_heads
    QPKV = mc.query_per_kvhead
    HD = mc.head_dim
    DIM = mc.dim
    F_DIM = mc.moe_inter_dim
    NEXP = mc.n_routed_experts
    NACT = mc.n_activated_experts
    S = seq_len

    cos_2d = cos.reshape(S, HD).contiguous()
    sin_2d = sin.reshape(S, HD).contiguous()

    graph = Graph()

    # ---- Pre-attention RMSNorm ----
    x_load = LinearOffChipLoad(
        underlying=input_tensor, stride=(1, 1), out_shape_tiled=(S, 1),
        tile_row=1, tile_col=H, par_dispatch=4,
    )
    x_normed = _rms_norm_chain(graph, x_load, K_features=H)

    cos_load = LinearOffChipLoad(
        underlying=cos_2d, stride=(1, 1), out_shape_tiled=(S, 1),
        tile_row=1, tile_col=HD, par_dispatch=4,
    )
    sin_load = LinearOffChipLoad(
        underlying=sin_2d, stride=(1, 1), out_shape_tiled=(S, 1),
        tile_row=1, tile_col=HD, par_dispatch=4,
    )

    def _project_per_head(weight_slice, do_qk_norm, do_rope):
        w_load = LinearOffChipLoad(
            underlying=weight_slice, stride=(0, 0), out_shape_tiled=(S, 1),
            tile_row=H, tile_col=HD, par_dispatch=4,
        )
        proj = BinaryMap(
            graph=graph, in1=x_normed, in2=w_load,
            fn=Matmul(), write_back_mu=False, compute_bw=1024,
        )
        if do_qk_norm:
            proj = _rms_norm_chain(graph, proj, K_features=HD)
        if do_rope:
            proj = _rope(graph, proj, cos_load, sin_load, S, 1, HD)
        return proj  # stream (1, S, 1) tile (1, head_dim)

    # ---- Per-head Q/K/V streams ----
    Q_streams = [
        _project_per_head(q_proj[:, h * HD : (h + 1) * HD].contiguous(),
                          do_qk_norm=True, do_rope=True)
        for h in range(NHQ)
    ]
    K_streams = [
        _project_per_head(k_proj[:, hkv * HD : (hkv + 1) * HD].contiguous(),
                          do_qk_norm=True, do_rope=True)
        for hkv in range(NHKV)
    ]
    V_streams = [
        _project_per_head(v_proj[:, hkv * HD : (hkv + 1) * HD].contiguous(),
                          do_qk_norm=False, do_rope=False)
        for hkv in range(NHKV)
    ]

    # ---- Per-Q-head row-streaming sdpa ----
    attn_streams = [
        _sdpa_streamed(graph, Q_streams[h], K_streams[h // QPKV],
                       V_streams[h // QPKV], S, HD)
        for h in range(NHQ)
    ]

    # ---- Per-head O-projection slice + sum across heads ----
    o_per_head = []
    for h in range(NHQ):
        slice_h = o_proj_weight[h * HD : (h + 1) * HD, :].contiguous()
        o_load_h = LinearOffChipLoad(
            underlying=slice_h, stride=(0,), out_shape_tiled=(S,),
            tile_row=HD, tile_col=H, par_dispatch=4,
        )
        o_h = BinaryMap(
            graph=graph, in1=attn_streams[h], in2=o_load_h,
            fn=Matmul(), write_back_mu=False, compute_bw=1024,
        )
        o_per_head.append(o_h)

    o_out = o_per_head[0]
    for h in range(1, NHQ):
        o_out = BinaryMap(
            graph=graph, in1=o_out, in2=o_per_head[h],
            fn=Add(), write_back_mu=False, compute_bw=1024,
        )

    # ---- First residual add: o_proj + input ----
    input_load = LinearOffChipLoad(
        underlying=input_tensor, stride=(1,), out_shape_tiled=(S,),
        tile_row=1, tile_col=H, par_dispatch=4,
    )
    res_add_0 = BinaryMap(
        graph=graph, in1=o_out, in2=input_load,
        fn=Add(), write_back_mu=False, compute_bw=1024,
    )

    # Broadcast for residual + post-attention RMSNorm
    res0_b = Broadcast(graph, res_add_0, 2)
    res0_for_norm = (res0_b, 0)
    res0_for_final_residual = (res0_b, 1)

    # ---- Post-attention RMSNorm ----
    normed_2 = _rms_norm_chain(graph, res0_for_norm, K_features=H)

    # ---- MoE block + final residual ----
    output_stream = _moe_block(
        graph,
        x_stream=normed_2,
        x_residual_stream=res0_for_final_residual,
        w_gate_list=w_gate_list,
        w_up_list=w_up_list,
        w_down_list=w_down_list,
        expert_weights_t=expert_weights,
        expert_multihot=expert_multihot,
        expert_onehot=expert_onehot,
        S=S, D=DIM, F_dim=F_DIM,
        n_experts=NEXP, n_active=NACT,
        tile_n=tile_n, tile_f=tile_f,
    )

    output = OffChipStore(
        graph=graph, input=output_stream,
        par_dispatch=4, store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output
