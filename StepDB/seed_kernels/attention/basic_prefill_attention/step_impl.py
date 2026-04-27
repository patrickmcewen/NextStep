"""STeP implementation: attention-only slice of the simple prefill transformer.

The full pre-attention pipeline runs in the dataflow graph:

    Load input -> RMSNorm
        -> per-head (Q proj | K proj | V proj)
        -> per-head Q/K RMSNorm
        -> per-head RoPE (rotate_half-style, mirrors seed_kernels/pre_attention/rope)
        -> per-head row-streaming sdpa (sdpa_core pattern) where K and V are
           replayed across the seq dim via Broadcast(stream, S) +
           StaticReassemble(merge_rank=outermost), giving the cycle
           [K[0..S-1], K[0..S-1], ...] — the exact pattern that
           LinearOffChipLoad(stride=(0, 1)) gives for an off-chip K cache,
           but driven from an upstream stream instead of a torch tensor.
        -> per-head O-proj slice + sum across heads
        -> ResAdd -> OffChipStore

All torch tensor generation lives in StepDB/precompute.py; only raw RNG'd
tensors are loaded as off-chip underlyings.
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
# rotate_half / RoPE (mirrors seed_kernels/pre_attention/rope)
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
# Cycle-replay a stream: turn (1, S) tile (1, head_dim) into
# (1, S, S) where for each outer position the inner iterates the full
# stream sequence (K[0..S-1] cycled S times).  Implemented as
# Broadcast(stream, S) + StaticReassemble(merge_rank=outermost) + Promote.
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
# Output: stream (1, S) of tile (1, head_dim) — same as sdpa_core's pattern.
# ---------------------------------------------------------------------------

def _sdpa_streamed(graph, Q_stream, K_stream, V_stream, S, head_dim):
    # Q: (1, S, 1) -> (1, S) -> (1, S, S) via RepeatStatic (broadcast on inner)
    Q_flat = Flatten(graph=graph, input=Q_stream, min_rank=0, max_rank=1)
    q_repeated = RepeatStatic(graph=graph, input=Q_flat, repeat_factor=S)

    # K: (1, S, 1) -> (1, S) -> cycle (1, S, S) via Broadcast(S) + StaticReassemble
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
# build_graph
# ---------------------------------------------------------------------------

def build_graph(dims):
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)
    mc = _model_config(dims["model_name"], is_small)

    t = precompute_tensors("basic_prefill_attention", dims)
    input_tensor   = t["input_tensor"]
    q_proj         = t["q_proj"]
    k_proj         = t["k_proj"]
    v_proj         = t["v_proj"]
    cos            = t["cos"]
    sin            = t["sin"]
    o_proj_weight  = t["o_proj_weight"]

    H = mc.hidden_dim
    NHQ = mc.num_heads
    NHKV = mc.num_kv_heads
    QPKV = mc.query_per_kvhead
    HD = mc.head_dim
    S = seq_len

    cos_2d = cos.reshape(S, HD).contiguous()
    sin_2d = sin.reshape(S, HD).contiguous()

    graph = Graph()

    # ---- Pre-attention RMSNorm of input_tensor ----
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
        """input -> projection -> (Q/K RMSNorm) -> (RoPE).  Output stream
        (1, S, 1) of tile (1, head_dim)."""
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
        return proj

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
    # attn_concat @ o_proj == sum_h(attn_h @ o_proj_slice_h),
    # where o_proj_slice_h = o_proj[h*HD : (h+1)*HD, :].
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

    # ---- Residual add: o_proj + input ----
    input_load = LinearOffChipLoad(
        underlying=input_tensor, stride=(1,), out_shape_tiled=(S,),
        tile_row=1, tile_col=H, par_dispatch=4,
    )
    output_stream = BinaryMap(
        graph=graph, in1=o_out, in2=input_load,
        fn=Add(), write_back_mu=False, compute_bw=1024,
    )

    output = OffChipStore(
        graph=graph, input=output_stream,
        par_dispatch=4, store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output
