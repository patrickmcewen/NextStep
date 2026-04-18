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
    for _ in range(rank):
        x = x.sum(dim=-3)
    return x

def accum_mul(x, rank=1):
    for _ in range(rank):
        x = x.prod(dim=-3)
    return x

def accum_retile_row(x, rank=1):
    for _ in range(rank):
        s = x.shape
        x = x.reshape(*s[:-3], s[-3] * s[-2], s[-1])
    return x

def accum_retile_col(x, rank=1):
    for _ in range(rank):
        s = x.shape
        x = x.reshape(*s[:-3], s[-2], s[-3] * s[-1])
    return x

def flat_partition(x, control, n):
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
    assert n >= 1, (
        f"reshape_stream: tensor {tuple(x.shape)} has no stream dims."
    )
    assert rank < n, (
        f"reshape_stream(rank={rank}): tensor {tuple(x.shape)} has {n} stream dims, "
        f"max valid rank is {n - 1}."
    )

    rank_pos = n - 1 - rank
    D = stream_shape[rank_pos]
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
    return list(torch.chunk(x, n, dim=0))

def static_reassemble(inputs, target_stream_shape=None):
    result = torch.cat(inputs, dim=0)
    if target_stream_shape is not None:
        tile_r, tile_c = result.shape[-2], result.shape[-1]
        target = tuple(target_stream_shape) + (tile_r, tile_c)
        if result.shape != target:
            result = result.reshape(target)
    return result

def binary_map_accum(a, b, rank=1, weight_transposed=False):
    _assert_stream_match(a, b, "binary_map_accum")
    if weight_transposed:
        mapped = torch.matmul(a, b.transpose(-2, -1))
    else:
        mapped = torch.matmul(a, b)
    for _ in range(rank):
        mapped = mapped.sum(dim=-3)
    return mapped

def offchip_store(x):
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
    "promote", "promote_outer", "flatten", "reshape_stream",
    "expand_ref", "repeat_ref", "repeat_static", "streamify",
    "retile_streamify",
    # Multi-output
    "broadcast", "parallelize", "static_reassemble",
    # Routing
    "flat_partition", "flat_reassemble",
    # Sink
    "offchip_store",
}
