"""
STeP DSL
"""

import torch
import torch.nn.functional as F

def offchip_load(underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False):
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


def offchip_load_ref(ref, underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False):
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

def select_gen(underlying):
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


def random_offchip_load(underlying, raddr, tile_row, tile_col, transposed=False):
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


def binary_matmul(a, b, weight_transposed=False):
    _assert_stream_match(a, b, "binary_matmul")
    if weight_transposed:
        return torch.matmul(a, b.transpose(-2, -1))
    return torch.matmul(a, b)


def binary_mul(a, b):
    _assert_stream_match(a, b, "binary_mul")
    return a * b


def binary_add(a, b):
    _assert_stream_match(a, b, "binary_add")
    return a + b


def binary_div(a, b):
    _assert_stream_match(a, b, "binary_div")
    return a / b


def binary_is_equal(a, b):
    _assert_stream_match(a, b, "binary_is_equal")
    return (a == b).float()


# ---------------------------------------------------------------------------
# Unary compute: UnaryMap → unary_*
# Mirrors: _apply_unary (L443) from functional.py
# ---------------------------------------------------------------------------

def unary_silu(x):
    return F.silu(x)


def unary_square(x):
    return x ** 2


def unary_exp(x):
    return torch.exp(x)


def unary_rsqrt(x):
    return torch.rsqrt(x)


def unary_pow2(x):
    return torch.pow(2.0, x)


def unary_mul_imm(x, constant):
    return x * constant


def unary_add_imm(x, constant):
    return x + constant


def unary_sub_imm(x, constant):
    return x - constant


def unary_rowwise_sum(x):
    return x.sum(dim=-1, keepdim=True)

def accum_add(x, rank=1):
    assert rank > 0, f"accum_add: rank must be > 0, got {rank}"
    for _ in range(rank):
        x = x.sum(dim=-3)
    return x

def accum_mul(x, rank=1):
    assert rank > 0, f"accum_mul: rank must be > 0, got {rank}"
    for _ in range(rank):
        x = x.prod(dim=-3)
    return x

def accum_retile_row(x, rank=1):
    assert rank > 0, f"accum_retile_row: rank must be > 0, got {rank}"
    for _ in range(rank):
        s = x.shape
        x = x.reshape(*s[:-3], s[-3] * s[-2], s[-1])
    return x

def accum_retile_col(x, rank=1):
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

def promote(x, rank=1):
    max_rank = x.ndim - 1
    assert rank >= 0, f"promote(rank={rank}): rank must be >= 0"
    assert rank <= max_rank, (
        f"promote(rank={rank}): tensor has {x.ndim} dims ({tuple(x.shape)}), "
        f"max valid rank is {max_rank}."
    )
    return x.unsqueeze(-(2 + rank))


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
    """Split the stream dim at ``reshape_rank`` into (new_count, chunk_size).

    Mirrors ReshapePadStream in step_tl/ops.py: reshape_rank counts from the
    right (0 = rightmost stream dim). Auto-pads with zeros when the dim size
    isn't divisible by ``chunk_size`` and ``reshape_rank == 0``.
    """
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

def streamify(x, repeat_factors, rank=0):
    result = x
    offset = 2 + rank  # skip buffer dims + tile_r + tile_c
    for rf in repeat_factors:
        result = result.unsqueeze(-offset)
        shape = list(result.shape)
        shape[-offset] = rf
        result = result.expand(shape).contiguous()
        offset += 1  # account for newly inserted dim
    return result

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

def binary_map_accum(a, b, rank=1, weight_transposed=False):
    assert rank > 0, f"binary_map_accum: rank must be > 0, got {rank}"
    _assert_stream_match(a, b, "binary_map_accum")
    if weight_transposed:
        mapped = torch.matmul(a, b.transpose(-2, -1))
    else:
        mapped = torch.matmul(a, b)
    for _ in range(rank):
        mapped = mapped.sum(dim=-3)
    return mapped

def random_offchip_store(underlying, wdata, waddr, tile_row, tile_col):
    assert underlying.dtype in [torch.float32, torch.float16], (
        f"random_offchip_store: underlying dtype must be float32 or float16, got {underlying.dtype}"
    )
    assert underlying.ndim == 2, (
        f"random_offchip_store: underlying must be 2D (flatten any batch dims first), "
        f"got shape {tuple(underlying.shape)}"
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
    R, C = underlying.shape
    assert R % tile_row == 0 and C % tile_col == 0, (
        f"random_offchip_store: ({R},{C}) not divisible by tile ({tile_row},{tile_col})"
    )
    grid_c = C // tile_col
    addrs = waddr.reshape(-1).long().tolist()
    wflat = wdata.reshape(-1, tile_row, tile_col)
    for i, a in enumerate(addrs):
        gr, gc = a // grid_c, a % grid_c
        underlying[gr * tile_row:(gr + 1) * tile_row, gc * tile_col:(gc + 1) * tile_col] = wflat[i]
    stream_shape = waddr.shape[:-2]
    return torch.ones(*stream_shape, 1, 1, dtype=torch.float32)


def offchip_store(x):
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
    "offchip_load", "offchip_load_ref", "select_gen", "metadata_gen",
    "cache_read_addr_gen", "random_offchip_load", "filter_last_tile",
    # Binary compute
    "binary_matmul", "binary_mul", "binary_add", "binary_div", "binary_is_equal",
    # Fused compute
    "binary_map_accum",
    # Unary compute
    "unary_silu", "unary_square", "unary_exp", "unary_rsqrt", "unary_pow2",
    "unary_mul_imm", "unary_add_imm", "unary_sub_imm", "unary_rowwise_sum",
    # Accumulation
    "accum_add", "accum_mul", "accum_retile_row", "accum_retile_col",
    # Stream shape
    "promote", "promote_outer", "flatten", "reshape_stream", "reshape_pad_stream",
    "expand_ref", "repeat_ref", "repeat_static", "streamify",
    "retile_streamify",
    # Multi-output
    "broadcast", "parallelize", "static_reassemble",
    # Routing
    "flat_partition", "flat_reassemble",
    # Sink
    "offchip_store", "random_offchip_store",
}
