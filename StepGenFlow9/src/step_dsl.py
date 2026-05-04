"""STeP DSL.

Each DSL function whose lowered STeP node carries a perf knob accepts that
knob as a keyword-only argument with default 1:

  - compute DSL calls (binary_*, unary_*, accum_*, binary_map_accum) accept
    ``compute_bw=N``.
  - off-chip DSL calls (offchip_load*, dyn_offchip_load, random_offchip_*,
    offchip_store) accept ``par_dispatch=N``.

The kwarg is asserted (>= 1) but otherwise inert at eager exec time —
the deterministic translator (dsl_to_step.py) reads it back from the AST
and forwards it to the STeP node constructor.
"""

import math

import torch
import torch.nn.functional as F


class Buffered:
    """Torch tensor with the buffer-rank promise made by bufferize().

    Layout: tensor.shape == (*in_stream, *buffer_grid, tile_r, tile_c)
    where len(buffer_grid) == buffer_rank.  Mirrors the IR's Bufferize ->
    Stream(stream_dtype=Buffer) state.
    """

    __slots__ = ("tensor", "buffer_rank")

    def __init__(self, tensor, buffer_rank):
        assert isinstance(tensor, torch.Tensor), \
            f"Buffered: tensor must be torch.Tensor, got {type(tensor).__name__}"
        assert isinstance(buffer_rank, int) and buffer_rank >= 1, \
            f"Buffered: buffer_rank must be int >= 1, got {buffer_rank!r}"
        assert tensor.ndim >= 2 + buffer_rank, (
            f"Buffered: tensor.ndim={tensor.ndim} too small for buffer_rank={buffer_rank}"
        )
        self.tensor = tensor
        self.buffer_rank = buffer_rank

    @property
    def buffer_shape(self):
        return tuple(self.tensor.shape[-2 - self.buffer_rank : -2])

    @property
    def in_stream_shape(self):
        return tuple(self.tensor.shape[: -2 - self.buffer_rank])

def offchip_load(underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False, par_dispatch=1):
    assert par_dispatch >= 1, f"offchip_load: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], f"offchip_load: underlying dtype must be float32 or float16, got {underlying.dtype}"
    R, C = underlying.shape[-2], underlying.shape[-1]

    # ---- Tiling invariant: must actually stream, not load as one giant tile ----
    # If the tensor is larger than one tile, there must be a streaming dimension.
    total_tiles = (R // tile_row) * (C // tile_col)
    if total_tiles > 1:
        assert any(s > 1 for s in out_shape_tiled), (
            f"offchip_load: tensor ({R}, {C}) has {total_tiles} tiles but "
            f"out_shape_tiled={out_shape_tiled} has no streaming dim > 1. "
            f"Use proper streaming, e.g. out_shape_tiled=({R // tile_row},) "
            f"with tile_row={tile_row}, tile_col={tile_col}."
        )
    grid_r, grid_c = R // tile_row, C // tile_col
    batch_shape = underlying.shape[:-2]

    # Truncate to evenly divisible size
    used_R, used_C = grid_r * tile_row, grid_c * tile_col
    if used_R != R or used_C != C:
        underlying = underlying[..., :used_R, :used_C]

    # Reshape to tile grid: (*batch, grid_r, tile_row, grid_c, tile_col)
    tiled = underlying.reshape(*batch_shape, grid_r, tile_row, grid_c, tile_col)
    # Permute so tile dims are last: (*batch, grid_r, grid_c, tile_row, tile_col)
    ndim = tiled.ndim
    perm = list(range(len(batch_shape))) + [ndim - 4, ndim - 2, ndim - 3, ndim - 1]
    tiled = tiled.permute(*perm)
    # Flatten batch + grid into single tile index
    flat = tiled.reshape(-1, tile_row, tile_col)

    # Compute linear tile index for every position in out_shape_tiled
    ranges = [torch.arange(s) for s in out_shape_tiled]
    grids = torch.meshgrid(*ranges, indexing="ij")
    linear_idx = sum(g.long() * int(s) for g, s in zip(grids, stride))

    result = flat[linear_idx.long()]  # (*out_shape_tiled, tile_row, tile_col)
    if transposed:
        result = result.transpose(-2, -1)
    return result.unsqueeze(0)  # prepend leading 1


def dyn_offchip_load(underlying, tensor_shape_tiled, tile_row, tile_col, par_dispatch=1):
    assert par_dispatch >= 1, f"dyn_offchip_load: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], (
        f"dyn_offchip_load: underlying dtype must be float32 or float16, got {underlying.dtype}"
    )
    R, C = underlying.shape[-2], underlying.shape[-1]
    assert R % tile_row == 0 and C % tile_col == 0, (
        f"dyn_offchip_load: ({R},{C}) not divisible by tile ({tile_row},{tile_col})"
    )
    grid_r, grid_c = R // tile_row, C // tile_col
    tiled = underlying.reshape(grid_r, tile_row, grid_c, tile_col).permute(0, 2, 1, 3)
    return tiled.reshape(*tensor_shape_tiled, tile_row, tile_col).unsqueeze(0)


def offchip_load_ref(ref, underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False, par_dispatch=1):
    assert par_dispatch >= 1, f"offchip_load_ref: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], f"offchip_load_ref: underlying dtype must be float32 or float16, got {underlying.dtype}"
    loaded = offchip_load(underlying, stride, out_shape_tiled, tile_row, tile_col, transposed)
    # loaded: (1, *out_shape_tiled, tile_row, tile_col)
    # target: (*ref_stream, *out_shape_tiled, tile_row, tile_col)
    ref_stream = list(ref.shape[:-2])
    target = ref_stream + list(out_shape_tiled) + [loaded.shape[-2], loaded.shape[-1]]
    # Prepend singleton dims so loaded is broadcastable to target
    while loaded.ndim < len(target):
        loaded = loaded.unsqueeze(0)
    return loaded.expand(target).contiguous()

def select_gen(underlying, is_multihot, n):
    _assert_int(underlying, "select_gen")
    assert isinstance(is_multihot, bool), (
        f"select_gen: is_multihot must be bool, got {type(is_multihot).__name__}"
    )
    assert underlying.shape[-1] == n, (
        f"select_gen: control's last dim must equal n={n}, "
        f"got shape {tuple(underlying.shape)}"
    )
    return underlying

def metadata_gen(tensor):
    return tensor.float().reshape(1, *tensor.shape, 1, 1)


def cache_read_addr_gen(idx, seq_len, row_offset):
    assert idx.shape[-2:] == (1, 1), (
        f"cache_read_addr_gen: idx tile shape must be (1,1), got {tuple(idx.shape[-2:])}"
    )
    assert seq_len.shape == idx.shape, (
        f"cache_read_addr_gen: idx {tuple(idx.shape)} and seq_len {tuple(seq_len.shape)} must match"
    )
    idx_flat = idx.reshape(-1).long()
    seq_len_flat = seq_len.reshape(-1).long()
    out = []
    for b in range(idx_flat.shape[0]):
        base = int(idx_flat[b]) * int(row_offset)
        n = int(seq_len_flat[b])
        assert n >= 0, f"cache_read_addr_gen: seq_len[{b}]={n} must be >= 0"
        out.append(torch.arange(base, base + n, dtype=torch.float32).reshape(1, n, 1, 1))
    return out


def expert_addr_gen(x, expert_addr_base, num_tile_per_expert):
    assert (x.sum(dim=-1) == 1).all(), (
        "expert_addr_gen: input must be one-hot (exactly one expert selected per element)"
    )
    expert_indices = x.argmax(dim=-1)
    base = expert_addr_base + expert_indices * num_tile_per_expert
    offsets = torch.arange(num_tile_per_expert, dtype=base.dtype)
    addrs = base.unsqueeze(-1) + offsets
    return addrs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()


def filter_last_tile(seq_len):
    assert seq_len.shape[-2:] == (1, 1), (
        f"filter_last_tile: input tile shape must be (1,1), got {tuple(seq_len.shape[-2:])}"
    )
    stream_shape = seq_len.shape[:-2]
    flat = seq_len.reshape(-1).long()
    assert (flat >= 1).all(), (
        f"filter_last_tile: seq_len must be >= 1 (Rust ref assumes >=1), got min={int(flat.min())}"
    )
    max_seq = int(flat.max().item())

    out = torch.zeros(flat.shape[0], max_seq, 2)
    for i in range(flat.shape[0]):
        n = int(flat[i])
        out[i, :n - 1, 1] = 1.0  # non-last -> column 1
        out[i, n - 1, 0] = 1.0   # last     -> column 0
    return out.reshape(*stream_shape, max_seq, 2)


def random_offchip_load(underlying, raddr, tile_row, tile_col, transposed=False, par_dispatch=1):
    assert par_dispatch >= 1, f"random_offchip_load: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], (
        f"random_offchip_load: underlying dtype must be float32 or float16, got {underlying.dtype}"
    )
    assert raddr.shape[-2:] == (1, 1), (
        f"random_offchip_load: raddr tile shape must be (1,1), got {tuple(raddr.shape[-2:])}"
    )
    R, C = underlying.shape[-2], underlying.shape[-1]
    assert R % tile_row == 0 and C % tile_col == 0, (
        f"random_offchip_load: ({R},{C}) not divisible by tile ({tile_row},{tile_col})"
    )
    grid_r, grid_c = R // tile_row, C // tile_col
    batch_shape = underlying.shape[:-2]
    tiled = underlying.reshape(*batch_shape, grid_r, tile_row, grid_c, tile_col)
    ndim = tiled.ndim
    perm = list(range(len(batch_shape))) + [ndim - 4, ndim - 2, ndim - 3, ndim - 1]
    flat = tiled.permute(*perm).reshape(-1, tile_row, tile_col)

    stream_shape = raddr.shape[:-2]
    addrs = raddr.reshape(-1).long()
    assert (addrs >= 0).all() and (addrs < flat.shape[0]).all(), (
        f"random_offchip_load: address out of range [0, {flat.shape[0]}), "
        f"got min={int(addrs.min())}, max={int(addrs.max())}"
    )
    result = flat[addrs].reshape(*stream_shape, tile_row, tile_col)
    if transposed:
        result = result.transpose(-2, -1)
    return result

def _assert_stream_match(a, b, op_name):
    a_stream = a.shape[:-2]
    b_stream = b.shape[:-2]
    assert a_stream == b_stream, (
        f"{op_name}: stream shape mismatch — a has stream {tuple(a_stream)} "
        f"(shape {tuple(a.shape)}) but b has stream {tuple(b_stream)} "
        f"(shape {tuple(b.shape)}). Both operands must have identical stream shapes."
    )


def _assert_float(x, op_name):
    assert x.dtype in (torch.float32, torch.float16), (
        f"{op_name}: input dtype must be float32 or float16, got {x.dtype}."
    )

def _assert_int(x, op_name):
    assert x.dtype in (torch.int32, torch.int64), (
        f"{op_name}: input dtype must be int32 or int64, got {x.dtype}."
    )


def binary_matmul(a, b, weight_transposed=False, compute_bw=1):
    assert compute_bw >= 1, f"binary_matmul: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_matmul")
    _assert_float(b, "binary_matmul")
    _assert_stream_match(a, b, "binary_matmul")
    if weight_transposed:
        return torch.matmul(a, b.transpose(-2, -1))
    return torch.matmul(a, b)


def binary_mul(a, b, compute_bw=1):
    assert compute_bw >= 1, f"binary_mul: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_mul")
    _assert_float(b, "binary_mul")
    _assert_stream_match(a, b, "binary_mul")
    return a * b


def binary_add(a, b, compute_bw=1):
    assert compute_bw >= 1, f"binary_add: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_add")
    _assert_float(b, "binary_add")
    _assert_stream_match(a, b, "binary_add")
    return a + b


def binary_div(a, b, compute_bw=1):
    assert compute_bw >= 1, f"binary_div: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_div")
    _assert_float(b, "binary_div")
    _assert_stream_match(a, b, "binary_div")
    return a / b


def binary_is_equal(a, b, compute_bw=1):
    assert compute_bw >= 1, f"binary_is_equal: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_is_equal")
    _assert_float(b, "binary_is_equal")
    _assert_stream_match(a, b, "binary_is_equal")
    return (a == b).float()

class _OffsetTile:
    __slots__ = ("data", "offsets")

    def __init__(self, data, offsets):
        self.data = data
        self.offsets = offsets

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype


def binary_set_offset(a, b, compute_bw=1):
    assert compute_bw >= 1, f"binary_set_offset: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_set_offset")
    _assert_float(b, "binary_set_offset")
    _assert_stream_match(a, b, "binary_set_offset")
    assert b.shape[-2:] == (1, 1), (
        f"binary_set_offset: b tile shape must be (1,1), got {tuple(b.shape[-2:])}"
    )
    offsets = b[..., 0, 0].long()
    return _OffsetTile(a, offsets)


def binary_row_wise_append(a, b, compute_bw=1):
    assert compute_bw >= 1, f"binary_row_wise_append: compute_bw must be >= 1, got {compute_bw}"
    if isinstance(a, _OffsetTile):
        data = a.data
        offsets = a.offsets
    else:
        data = a
        offsets = torch.zeros(data.shape[:-2], dtype=torch.long)
    _assert_float(data, "binary_row_wise_append")
    _assert_float(b, "binary_row_wise_append")
    _assert_stream_match(data, b, "binary_row_wise_append")
    tile_r, tile_c = data.shape[-2], data.shape[-1]
    M = b.shape[-2]
    assert b.shape[-1] == tile_c, (
        f"binary_row_wise_append: column dim mismatch ({b.shape[-1]} vs {tile_c})"
    )
    assert (offsets + M <= tile_r).all(), (
        f"binary_row_wise_append: not enough space to append {M} rows "
        f"(tile_r={tile_r}, max offset={int(offsets.max())})"
    )
    stream_shape = data.shape[:-2]
    row_idx = offsets.unsqueeze(-1) + torch.arange(M, dtype=torch.long, device=data.device)
    row_idx = row_idx.unsqueeze(-1).expand(*stream_shape, M, tile_c)
    result = data.clone()
    result.scatter_(dim=-2, index=row_idx, src=b.to(data.dtype))
    return result


def binary_cache_write_addr_gen(idx, seq_len, row_offset, compute_bw=1):
    """Compute KV-cache write address: ``idx * row_offset + seq_len``.

    Mirrors step-perf/map_fn::cache_write_addr_gen. ``idx`` and ``seq_len`` are
    (*stream, 1, 1) scalar tiles; ``row_offset`` is a Python int.
    """
    assert compute_bw >= 1, f"binary_cache_write_addr_gen: compute_bw must be >= 1, got {compute_bw}"
    assert idx.shape[-2:] == (1, 1), (
        f"binary_cache_write_addr_gen: idx tile shape must be (1,1), got {tuple(idx.shape[-2:])}"
    )
    assert seq_len.shape == idx.shape, (
        f"binary_cache_write_addr_gen: idx {tuple(idx.shape)} and seq_len "
        f"{tuple(seq_len.shape)} must match"
    )
    assert isinstance(row_offset, int), (
        f"binary_cache_write_addr_gen: row_offset must be int, got {type(row_offset).__name__}"
    )
    return idx * row_offset + seq_len


# ---------------------------------------------------------------------------
# Unary compute: UnaryMap → unary_*
# Mirrors: _apply_unary (L443) from functional.py
# ---------------------------------------------------------------------------

def unary_silu(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_silu: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_silu")
    return F.silu(x)


def unary_square(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_square: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_square")
    return x ** 2


def unary_exp(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_exp: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_exp")
    return torch.exp(x)


def unary_rsqrt(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_rsqrt: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_rsqrt")
    return torch.rsqrt(x)


def unary_pow2(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_pow2: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_pow2")
    return torch.pow(2.0, x)


def unary_mul_imm(x, constant, compute_bw=1):
    assert compute_bw >= 1, f"unary_mul_imm: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_mul_imm")
    assert constant != 0.0, "unary_mul_imm: constant must be nonzero."
    return x * constant


def unary_add_imm(x, constant, compute_bw=1):
    assert compute_bw >= 1, f"unary_add_imm: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_add_imm")
    return x + constant


def unary_sub_imm(x, constant, compute_bw=1):
    assert compute_bw >= 1, f"unary_sub_imm: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_sub_imm")
    return x - constant


def unary_rowwise_sum(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_rowwise_sum: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_rowwise_sum")
    return x.sum(dim=-1, keepdim=True)


def unary_mask_row(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_mask_row: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_mask_row")
    return torch.ones_like(x[..., :1])


def unary_select_to_scalar(x, compute_bw=1):
    assert compute_bw >= 1, f"unary_select_to_scalar: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_select_to_scalar")
    return x


def unary_to_const_int(x, constant, compute_bw=1):
    assert compute_bw >= 1, f"unary_to_const_int: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "unary_to_const_int")
    return torch.full_like(x, constant, dtype=torch.float32)

def accum_add(x, rank=1, compute_bw=1):
    assert compute_bw >= 1, f"accum_add: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "accum_add")
    assert rank > 0, f"accum_add: rank must be > 0, got {rank}"
    for _ in range(rank):
        x = x.sum(dim=-3)
    return x

def accum_mul(x, rank=1, compute_bw=1):
    assert compute_bw >= 1, f"accum_mul: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "accum_mul")
    assert rank > 0, f"accum_mul: rank must be > 0, got {rank}"
    for _ in range(rank):
        x = x.prod(dim=-3)
    return x

def accum_max(x, rank=1, compute_bw=1):
    assert compute_bw >= 1, f"accum_max: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "accum_max")
    assert rank > 0, f"accum_max: rank must be > 0, got {rank}"
    for _ in range(rank):
        x = x.amax(dim=-3)
    return x

def accum_retile_row(x, rank=1, compute_bw=1):
    assert compute_bw >= 1, f"accum_retile_row: compute_bw must be >= 1, got {compute_bw}"
    assert rank > 0, f"accum_retile_row: rank must be > 0, got {rank}"
    for _ in range(rank):
        s = x.shape
        x = x.reshape(*s[:-3], s[-3] * s[-2], s[-1])
    return x

def accum_retile_col(x, rank=1, compute_bw=1):
    assert compute_bw >= 1, f"accum_retile_col: compute_bw must be >= 1, got {compute_bw}"
    assert rank > 0, f"accum_retile_col: rank must be > 0, got {rank}"
    for _ in range(rank):
        s = x.shape
        # Permute the accum dim (dim -3) next to the tile-col dim (dim -1)
        # so the row-major reshape produces col-concatenated tiles.
        ndim = x.ndim
        perm = list(range(ndim - 3)) + [ndim - 2, ndim - 3, ndim - 1]
        x = x.permute(perm).contiguous()
        x = x.reshape(*s[:-3], s[-2], s[-3] * s[-1])
    return x


def accum_signal_req_all_read(x, rank=1, compute_bw=1):
    assert compute_bw >= 1, f"accum_signal_req_all_read: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(x, "accum_signal_req_all_read")
    assert rank > 0, f"accum_signal_req_all_read: rank must be > 0, got {rank}"
    stream_shape = x.shape[:-2 - rank]
    return torch.ones(*stream_shape, 1, 1)

def eager_merge(inputs):
    n = len(inputs)
    assert n > 0, "eager_merge: must have at least one input"
    tile_r, tile_c = inputs[0].shape[-2], inputs[0].shape[-1]
    for i, p in enumerate(inputs):
        assert p.shape[-2:] == (tile_r, tile_c), (
            f"eager_merge: input {i} tile shape {tuple(p.shape[-2:])} != ({tile_r},{tile_c})"
        )
    data = torch.cat(list(inputs), dim=0)
    counts = [p.shape[0] for p in inputs]
    select = torch.zeros(sum(counts), n)
    offset = 0
    for i, c in enumerate(counts):
        select[offset:offset + c, i] = 1.0
        offset += c
    return [data, select]


def flat_partition(x, control, n):
    assert control.shape[-1] == n, (
        f"flat_partition: control's last dim must equal n={n} (num consumers), "
        f"got control shape {tuple(control.shape)}."
    )
    tile_r, tile_c = x.shape[-2], x.shape[-1]
    flat_inp = x.reshape(-1, tile_r, tile_c)
    flat_mh = control.reshape(-1, n)
    assert flat_inp.shape[0] == flat_mh.shape[0], \
        f"Tile count mismatch: input has {flat_inp.shape[0]} vs selector with {flat_mh.shape[0]}. The input stream and selector stream must have the same number of stream elements, meaning that x.reshape(-1, tile_r, tile_c) and control.reshape(-1, n) must resolve to the same number of elements in the upper (-1) dimensions."

    results = []
    for i in range(n):
        mask = flat_mh[:, i] > 0
        results.append(flat_inp[mask])
    return results


def flat_reassemble(inputs, control):
    n = len(inputs)
    tile_r, tile_c = inputs[0].shape[-2], inputs[0].shape[-1]
    flat_mh = control.reshape(-1, n)
    total = flat_mh.shape[0]

    ptrs = [0] * n
    token_groups = []

    for t in range(total):
        group = []
        for i in range(n):
            if flat_mh[t, i] > 0:
                if ptrs[i] < inputs[i].shape[0]:
                    group.append(inputs[i][ptrs[i]])
                    ptrs[i] += 1
        if len(group) == 0:
            group.append(torch.zeros_like(inputs[0][0:1].squeeze(0)))
        token_groups.append(torch.stack(group, dim=0))

    output = torch.stack(token_groups, dim=0)

    ctrl_stream_shape = control.shape[:-1]
    if ctrl_stream_shape[0] != 1:
        ctrl_stream_shape = (1,) + ctrl_stream_shape
    n_active = output.shape[1]
    return output.reshape(*ctrl_stream_shape, n_active, tile_r, tile_c)

def flatmap_filter_row_streamify(x, mask):
    tile_r, tile_c = x.shape[-2], x.shape[-1]
    flat_data = x.reshape(-1, tile_r, tile_c)
    flat_mask = mask.reshape(-1, tile_r, 1)
    rows = []
    for i in range(flat_data.shape[0]):
        for r in range(tile_r):
            if flat_mask[i, r, 0] > 0:
                rows.append(flat_data[i, r:r + 1, :])
    assert len(rows) > 0, "flatmap_filter_row_streamify: no rows passed the mask"
    result = torch.cat(rows, dim=0)
    outer = x.shape[:-2][:-1]
    return result.reshape(*outer, len(rows), 1, tile_c)


def flatmap_counter(x):
    stream_shape = x.shape[:-2]
    flat = x.reshape(-1)
    assert flat.numel() == 1, (
        "flatmap_counter: only single-scalar input supported"
    )
    n = int(flat[0].item())
    return torch.arange(n, dtype=x.dtype).reshape(*stream_shape, n, 1, 1)


def promote(x, rank=1):
    max_rank = x.ndim - 1
    assert rank >= 0, f"promote(rank={rank}): rank must be >= 0"
    assert rank <= max_rank, (
        f"promote(rank={rank}): tensor has {x.ndim} dims ({tuple(x.shape)}), "
        f"max valid rank is {max_rank}."
    )
    return x.unsqueeze(-(3 + rank))


def promote_outer(x):
    return x.unsqueeze(0)


def flatten(x, min_rank, max_rank):
    tile_r, tile_c = x.shape[-2], x.shape[-1]
    stream_shape = list(x.shape[:-2])
    n = len(stream_shape)
    assert n >= 1, (
        f"flatten: tensor {tuple(x.shape)} has no stream dims (need at least 3D)."
    )
    assert max_rank < n, (
        f"flatten(min_rank={min_rank}, max_rank={max_rank}): tensor {tuple(x.shape)} "
        f"has {n} stream dims, so max valid rank is {n - 1}."
    )
    assert 0 <= min_rank <= max_rank, (
        f"flatten: need 0 <= min_rank <= max_rank, got min_rank={min_rank}, max_rank={max_rank}."
    )
    min_idx = n - 1 - max_rank   # max_rank -> leftmost merged index
    max_idx = n - 1 - min_rank   # min_rank -> rightmost merged index
    merged = 1
    for i in range(min_idx, max_idx + 1):
        merged *= stream_shape[i]
    new_stream = stream_shape[:min_idx] + [merged] + stream_shape[max_idx + 1:]
    return x.reshape(*new_stream, tile_r, tile_c)


def expand_ref(x, ref, expand_rank):
    ref_stream = list(ref.shape[:-2])
    inp_stream = list(x.shape[:-2])
    assert expand_rank > 0, f"expand_rank must be > 0, got {expand_rank}"
    assert inp_stream[-expand_rank:] == [1] * expand_rank, (
        f"expand_ref: trailing {expand_rank} stream dims must be 1, got {inp_stream}"
    )
    assert inp_stream[:-expand_rank] == ref_stream[:-expand_rank], (
        f"expand_ref: leading stream dims must match: {inp_stream[:-expand_rank]} vs {ref_stream[:-expand_rank]}"
    )
    expand_shape = ref_stream + list(x.shape[-2:])
    return x.expand(expand_shape).contiguous()


def repeat_static(x, factor):
    result = x.unsqueeze(-3)
    shape = list(result.shape)
    shape[-3] = factor
    return result.expand(shape).contiguous()

def reshape_stream(x, chunk_size, rank=0, add_outer_dim=False):
    tile_r, tile_c = x.shape[-2], x.shape[-1]
    stream_shape = list(x.shape[:-2])
    n = len(stream_shape)
    assert rank >= 0, f"reshape_stream(rank={rank}): rank must be >= 0"
    if add_outer_dim:
        assert n == 0, (
            f"reshape_stream(add_outer_dim=True): input stream rank must be 0 "
            f"(a single tile, x.ndim==2), got shape {tuple(x.shape)}."
        )
    else:
        assert n >= 1, (
            f"reshape_stream: tensor {tuple(x.shape)} has no stream dims."
        )
        assert rank < n, (
            f"reshape_stream(rank={rank}): tensor {tuple(x.shape)} has {n} stream dims, "
            f"max valid rank is {n - 1}."
        )

    rank_pos = n - 1 - rank
    D = stream_shape[rank_pos]
    assert D % chunk_size == 0 or rank == 0, (
        f"reshape_stream: shape[{rank_pos}]={D} not divisible by chunk_size={chunk_size}. "
        f"Automatic padding is only allowed when rank==0, got rank={rank}."
    )
    padded_D = ((D + chunk_size - 1) // chunk_size) * chunk_size

    if padded_D != D:
        pad_sizes = [0] * (2 * len(x.shape))
        pad_idx = 2 * (len(x.shape) - 1 - rank_pos)
        pad_sizes[pad_idx + 1] = padded_D - D
        x = F.pad(x, pad_sizes, value=0.0)
        stream_shape[rank_pos] = padded_D

    pre = stream_shape[:rank_pos]
    post = stream_shape[rank_pos + 1:]
    new_count = padded_D // chunk_size

    if add_outer_dim:
        new_shape = [1] + pre + [new_count, chunk_size] + post + [tile_r, tile_c]
    else:
        new_shape = pre + [new_count, chunk_size] + post + [tile_r, tile_c]

    return x.reshape(new_shape)

def reshape_pad_stream(x, chunk_size, reshape_rank=0):
    return reshape_stream(x, chunk_size=chunk_size, rank=reshape_rank)


def retile_streamify(x, chunk, split_row=True):
    if split_row:
        stream_shape = x.shape[:-2]
        last = stream_shape[-1]
        pre = stream_shape[:-1]
        tile_r, tile_c = x.shape[-2], x.shape[-1]

        actual_num_chunks = tile_r // chunk
        assert tile_r % chunk == 0, (
            f"retile_streamify: tile_r={tile_r} not divisible by chunk={chunk}"
        )
        reshaped = x.reshape(*pre, last, actual_num_chunks, chunk, tile_c)
        return reshaped.reshape(*pre, last * actual_num_chunks, chunk, tile_c)

    # split_col
    stream_shape = x.shape[:-2]
    last = stream_shape[-1]
    pre = stream_shape[:-1]
    tile_r, tile_c = x.shape[-2], x.shape[-1]

    actual_num_chunks = tile_c // chunk
    assert tile_c % chunk == 0, (
        f"retile_streamify: tile_c={tile_c} not divisible by chunk={chunk}"
    )
    reshaped = x.reshape(*pre, last, tile_r, actual_num_chunks, chunk)
    perm = list(range(len(pre))) + [len(pre), len(pre) + 2, len(pre) + 1, len(pre) + 3]
    reshaped = reshaped.permute(perm)
    return reshaped.reshape(*pre, last * actual_num_chunks, tile_r, chunk)

def repeat_ref(x, ref):
    ref_stream = list(ref.shape[:-2])
    inp_stream = list(x.shape[:-2])
    tile_dims = list(x.shape[-2:])

    assert inp_stream == ref_stream[:-1], (
        f"x stream shape must equal ref stream shape minus its trailing dim: "
        f"{inp_stream} vs {ref_stream[:-1]}"
    )

    last_stream_dim = -3

    result = x.unsqueeze(last_stream_dim)
    expand_shape = ref_stream + tile_dims
    return result.expand(expand_shape).contiguous()

def streamify(x, stride, out_shape_tiled):
    assert isinstance(x, Buffered), \
        f"streamify expects a Buffered (output of bufferize()), got {type(x).__name__}"
    assert len(stride) == len(out_shape_tiled), (
        f"streamify: stride {tuple(stride)} and out_shape_tiled {tuple(out_shape_tiled)} "
        f"must have same length"
    )

    buffer_shape = x.buffer_shape
    n_tiles = math.prod(buffer_shape)
    max_idx = sum((s - 1) * st for s, st in zip(out_shape_tiled, stride))
    assert max_idx < n_tiles, (
        f"streamify: stride {tuple(stride)} x out_shape_tiled {tuple(out_shape_tiled)} "
        f"exceeds buffer grid {buffer_shape} (max_idx={max_idx}, n_tiles={n_tiles})"
    )

    t = x.tensor
    in_stream_rank = t.ndim - 2 - x.buffer_rank
    tile_r, tile_c = t.shape[-2], t.shape[-1]
    flat = t.reshape(*t.shape[:in_stream_rank], -1, tile_r, tile_c)

    ranges = [torch.arange(s) for s in out_shape_tiled]
    grids = torch.meshgrid(*ranges, indexing="ij")
    linear_idx = sum(g.long() * int(s) for g, s in zip(grids, stride))
    return flat[..., linear_idx.long(), :, :]


def bufferize(x, rank):
    return Buffered(x, buffer_rank=rank)


def dyn_streamify(x, ref):
    assert isinstance(x, Buffered), \
        f"dyn_streamify expects a Buffered (output of bufferize()), got {type(x).__name__}"
    bufferized_rank = x.buffer_rank
    ref_stream_shape = ref.shape[:-2]
    buf_and_tile_dims = x.tensor.shape[-(2 + bufferized_rank):]
    expand_shape = list(ref_stream_shape) + list(buf_and_tile_dims)
    return x.tensor.expand(expand_shape).contiguous()

def broadcast(x, n):
    return [x.clone() for _ in range(n)]

def parallelize(x, n):
    # Cycle-level round-robin (matches Rust parallelize.rs semantics with
    # switch_cycles=[1,...]): consumer i gets tokens i, n+i, 2n+i, ...
    # rather than a contiguous chunk.
    return [x[i::n].contiguous() for i in range(n)]

def static_reassemble(inputs, target_stream_shape=None):
    # Inverse of parallelize: interleave tokens across inputs so
    # output[k*n + i] = inputs[i][k]. Matches Rust static_reassemble
    # round-robin dequeue.
    n = len(inputs)
    S = inputs[0].shape[0]
    stacked = torch.stack(list(inputs), dim=1)  # (S, n, *rest)
    result = stacked.reshape(S * n, *inputs[0].shape[1:])
    if target_stream_shape is not None:
        tile_r, tile_c = result.shape[-2], result.shape[-1]
        target = tuple(target_stream_shape) + (tile_r, tile_c)
        if result.shape != target:
            result = result.reshape(target)
    return result

def binary_map_accum(a, b, rank=1, weight_transposed=False, compute_bw=1):
    assert compute_bw >= 1, f"binary_map_accum: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_map_accum")
    _assert_float(b, "binary_map_accum")
    assert rank > 0, f"binary_map_accum: rank must be > 0, got {rank}"
    _assert_stream_match(a, b, "binary_map_accum")
    if weight_transposed:
        mapped = torch.matmul(a, b.transpose(-2, -1))
    else:
        mapped = torch.matmul(a, b)
    for _ in range(rank):
        mapped = mapped.sum(dim=-3)
    return mapped

def random_offchip_store(underlying, wdata, waddr, tile_row, tile_col, base_addr_byte=0, par_dispatch=1):
    assert par_dispatch >= 1, f"random_offchip_store: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], (
        f"random_offchip_store: underlying dtype must be float32 or float16, got {underlying.dtype}"
    )
    assert waddr.shape[-2:] == (1, 1), (
        f"random_offchip_store: waddr tile shape must be (1,1), got {tuple(waddr.shape[-2:])}"
    )
    assert wdata.shape[-2:] == (tile_row, tile_col), (
        f"random_offchip_store: wdata tile {tuple(wdata.shape[-2:])} != ({tile_row},{tile_col})"
    )
    assert wdata.shape[:-2] == waddr.shape[:-2], (
        f"random_offchip_store: wdata stream {tuple(wdata.shape[:-2])} != waddr stream {tuple(waddr.shape[:-2])}"
    )
    R, C = underlying.shape[-2], underlying.shape[-1]
    assert R % tile_row == 0 and C % tile_col == 0, (
        f"random_offchip_store: ({R},{C}) not divisible by tile ({tile_row},{tile_col})"
    )
    # Mirror random_offchip_load's flat tile walk: batch dims (row-major) -> grid_r -> grid_c.
    # The Rust impl asserts 2D underlying, but the Python op layer (ops.py) builds tensor_shape_tiled
    # with leading batch dims (e.g. KV cache [batch, maxN, num_kv_heads, head_dim]), so we follow
    # the load-side semantics and accept N-D underlying.
    assert underlying.is_contiguous(), (
        "random_offchip_store: underlying must be contiguous so writes propagate through the view"
    )
    batch_shape = underlying.shape[:-2]
    B = 1
    for d in batch_shape:
        B *= d
    grid_r, grid_c = R // tile_row, C // tile_col
    tiles_per_batch = grid_r * grid_c
    flat_batch = underlying.view(B, R, C)
    addrs = waddr.reshape(-1).long().tolist()
    wflat = wdata.reshape(-1, tile_row, tile_col)
    for i, a in enumerate(addrs):
        b = a // tiles_per_batch
        within = a % tiles_per_batch
        gr, gc = within // grid_c, within % grid_c
        flat_batch[b, gr * tile_row:(gr + 1) * tile_row, gc * tile_col:(gc + 1) * tile_col] = wflat[i]
    stream_shape = waddr.shape[:-2]
    return torch.ones(*stream_shape, 1, 1, dtype=torch.float32)


def offchip_store(x, par_dispatch=1):
    assert par_dispatch >= 1, f"offchip_store: par_dispatch must be >= 1, got {par_dispatch}"
    # Note: the Rust IR also has a DynOffChipStore whose runtime body is byte-for-byte
    # identical to OffChipStore — they only differ at construction (DynOffChipStore reads
    # tensor_shape_tiled from a JSON file at startup instead of taking it as a Vec<usize>).
    # Since this DSL doesn't deal with on-disk shape files, dyn_offchip_store is omitted;
    # offchip_store covers the value-level behavior of both.
    assert x.ndim >= 2, (
        f"offchip_store: input must be a tile stream (at least 2D for tile_r, tile_c), "
        f"got shape {tuple(x.shape)}."
    )
    tile_r, tile_c = x.shape[-2], x.shape[-1]

    # Strip leading 1 if present
    if x.shape[0] == 1:
        x = x[0]

    stream_shape = x.shape[:-2]

    if len(stream_shape) == 0:
        return x  # single tile

    if len(stream_shape) == 1:
        return x.reshape(stream_shape[0] * tile_r, tile_c)

    # 2-D+ stream_shape: last two stream_shape dims are row/col tile counts
    Tc = stream_shape[-1]
    ndim = len(x.shape)
    perm = list(range(ndim - 4)) + [ndim - 4, ndim - 2, ndim - 3, ndim - 1]
    x = x.permute(*perm).contiguous()

    total_rows = tile_r
    for d in stream_shape[:-1]:
        total_rows *= d
    return x.reshape(int(total_rows), int(Tc * tile_c))

DSL_FUNCTIONS = {
    # Source
    "offchip_load", "offchip_load_ref", "dyn_offchip_load",
    "select_gen", "metadata_gen", "expert_addr_gen",
    "cache_read_addr_gen", "random_offchip_load", "filter_last_tile",
    # Binary compute
    "binary_matmul", "binary_mul", "binary_add", "binary_div", "binary_is_equal",
    "binary_set_offset", "binary_row_wise_append", "binary_cache_write_addr_gen",
    # Fused compute
    "binary_map_accum",
    # Unary compute
    "unary_silu", "unary_square", "unary_exp", "unary_rsqrt", "unary_pow2",
    "unary_mul_imm", "unary_add_imm", "unary_sub_imm", "unary_rowwise_sum",
    "unary_mask_row", "unary_select_to_scalar", "unary_to_const_int",
    # Accumulation
    "accum_add", "accum_mul", "accum_max", "accum_retile_row", "accum_retile_col",
    "accum_signal_req_all_read",
    # Stream shape
    "promote", "promote_outer", "flatten", "reshape_stream", "reshape_pad_stream",
    "expand_ref", "repeat_ref", "repeat_static", "streamify", "dyn_streamify",
    "bufferize", "retile_streamify",
    # Multi-output
    "broadcast", "parallelize", "static_reassemble",
    # Routing
    "eager_merge", "flat_partition", "flat_reassemble",
    # Flatmap
    "flatmap_filter_row_streamify", "flatmap_counter",
    # Sink
    "offchip_store", "random_offchip_store",
}

# ---------------------------------------------------------------------------
# Shape trace (gated by env var STEP_DSL_TRACE=1).
# When enabled, every DSL_FUNCTIONS op prints input/output shapes to stdout
# so the orchestrator can capture and feed the trace back to the LLM.
# ---------------------------------------------------------------------------

import os as _step_dsl_os
import functools as _step_dsl_ft
import inspect as _step_dsl_isp

_STEP_DSL_TRACE = _step_dsl_os.environ.get("STEP_DSL_TRACE", "") == "1"


def _step_dsl_fmt(v):
    if torch.is_tensor(v):
        s = tuple(v.shape)
        if len(s) >= 2:
            tile = f"tile({s[-2]},{s[-1]})"
            stream = s[:-2]
            return f"stream{tuple(stream)}×{tile}" if stream else tile
        return f"shape{s}"
    if isinstance(v, (list, tuple)) and v and all(torch.is_tensor(x) for x in v):
        opener, closer = ("[", "]") if isinstance(v, list) else ("(", ")")
        return opener + ", ".join(_step_dsl_fmt(x) for x in v) + closer
    return repr(v)


def _step_dsl_log_shapes(_fn):
    if not _STEP_DSL_TRACE:
        return _fn
    _name = _fn.__name__
    _sig = _step_dsl_isp.signature(_fn)

    @_step_dsl_ft.wraps(_fn)
    def _wrapper(*args, **kwargs):
        bound = _sig.bind(*args, **kwargs)
        in_str = ", ".join(f"{k}={_step_dsl_fmt(v)}" for k, v in bound.arguments.items())
        print(f"[step_dsl] {_name} input shape(s): {in_str}", flush=True)
        result = _fn(*args, **kwargs)
        print(f"[step_dsl] {_name} output shape(s): {_step_dsl_fmt(result)}", flush=True)
        return result

    return _wrapper


if _STEP_DSL_TRACE:
    _step_dsl_g = globals()
    for _step_dsl_n in DSL_FUNCTIONS:
        assert _step_dsl_n in _step_dsl_g, (
            f"DSL_FUNCTIONS lists '{_step_dsl_n}' but it is not defined in step_dsl.py"
        )
        _step_dsl_g[_step_dsl_n] = _step_dsl_log_shapes(_step_dsl_g[_step_dsl_n])
    del _step_dsl_g, _step_dsl_n
