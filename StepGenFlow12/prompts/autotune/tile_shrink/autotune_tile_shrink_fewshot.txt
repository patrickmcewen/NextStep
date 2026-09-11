# Shrinking tile sizes to relieve on-chip memory pressure

On-chip memory cost is dominated by the tile shapes of live `StepTensor`s, **not**
the length of their stream dims. To free memory, evict dims from the tile into
the stream. The cost is that what used to be parallel compute inside a tile
becomes sequential stream processing — so shrink only as much as the capacity
constraint demands. Going all the way to `tile_row=tile_col=1` (the limit case
shown below) usually leaves performance on the table; prefer the smallest
shrink that fits.

## The recipe (apply per dim you want to evict)

1. **Rebuild the `offchip_load`.** Shrink `tile_row` / `tile_col` by factor `f`,
   grow `out_shape_tiled` by `f` along the same axis (or add a new stream dim),
   and recompute `stride` in the *new* tile-grid coordinates. A dim that used
   to broadcast inside the tile becomes a `0` entry in `stride`.

2. **Rewrite every consumer of that dim.** Use this table — left column is
   work that lived inside the tile; right column is the equivalent across the
   stream:

   | Intra-tile (before)                                     | Inter-tile (after)                                              |
   |---------------------------------------------------------|-----------------------------------------------------------------|
   | `binary_matmul(a, b)` over reduction dim K              | `binary_map_accum(a, b, rank=1)` with K as innermost stream dim |
   | `binary_matmul(a, b, weight_transposed=True)`           | same map_accum after laying both operands out with K innermost  |
   | `unary_rowwise_sum(x)` (reduce tile cols)               | `accum_add(x, rank=1)`                                          |
   | Tile-shape `(R, 1)` broadcast in `binary_mul/div/add`   | `repeat_static(x, factor=C)` then the binary op                 |
   | Tile-shape `(1, C)` broadcast vs. another stream        | `expand_ref(x, ref, expand_rank=…)` then the binary op          |
   | `retile_streamify(x, chunk=1, split_row=False)` (the larger-tile trick to expose a tile dim as stream so `accum_*` can reduce it) | redundant once the dim is already a stream dim; delete it      |

3. **Audit reduction order.** `accum_*(rank=k)` and `binary_map_accum(rank=k)`
   reduce the **innermost `k` stream dims**. If the dim you just evicted is
   not innermost, reorder it with `bufferize(rank=k) → streamify(stride=…,
   out_shape_tiled=…)` *before* the accum (see Example 3).

4. **Re-check rank-sensitive ops.** Once a tile shrinks to size 1 on any axis,
   `OffChipStore`, `ExpandRef`, `DynLinearOffChipLoad`, and `Bufferize` all have
   minimum-rank assertions that may now fire — promote the stream with
   `promote(rank=0)` or `promote_outer` if needed.

---

## Example 1 — rms_norm (the canonical case)

Reference shape: `input` is `(M, K)`. Reference computes
`y[m, k] = x[m, k] * rsqrt(mean(x[m, :]**2) + eps)`.

### Before (tile `(tile_m, tile_k) = (8, 64)`)
```python
def tiled_reference(dims, tensors):
    tile_m, tile_k = dims["tile_m"], dims["tile_k"]
    M, K = dims["M"], dims["K"]
    grid_r, grid_c = M // tile_m, K // tile_k  # grid_c = 1

    x = offchip_load(
        tensors["input"],
        stride=(grid_c,),
        out_shape_tiled=(grid_r,),
        tile_row=tile_m,
        tile_col=tile_k,
    )
    x_sq    = unary_square(x)
    sum_sq  = unary_rowwise_sum(x_sq)            # intra-tile reduce → tile (tile_m, 1)
    mean_sq = unary_mul_imm(sum_sq, constant=1.0 / K)
    mean_e  = unary_add_imm(mean_sq, constant=1e-5)
    rsqrt   = unary_rsqrt(mean_e)
    normed  = binary_mul(x, rsqrt)               # implicit (tile_m, K) ⊗ (tile_m, 1) broadcast
    return offchip_store(normed)
```

### After (tile `(1, 1)`)
```python
def tiled_reference(dims, tensors):
    M, K = dims["M"], dims["K"]
    x = offchip_load(
        tensors["input"],
        stride=(K, 1),                  # recomputed: tile grid is now M × K of 1×1 tiles
        out_shape_tiled=(M, K),         # K absorbed into the stream
        tile_row=1,
        tile_col=1,
    )
    x_sq    = unary_square(x)
    sum_sq  = accum_add(x_sq, rank=1)            # rule: rowwise_sum → accum_add(rank=1)
    mean_sq = unary_mul_imm(sum_sq, constant=1.0 / K)
    mean_e  = unary_add_imm(mean_sq, constant=1e-5)
    inv     = unary_rsqrt(mean_e)
    scale   = repeat_static(inv, factor=K)       # rule: (tile_m,1)→(tile_m,K) broadcast becomes repeat_static
    normed  = binary_mul(x, scale)
    return offchip_store(normed)
```

Transforms applied (in order):
1. Load: `tile=(tile_m, tile_k)`, `out_shape=(grid_r,)`, `stride=(grid_c,)`
   → `tile=(1,1)`, `out_shape=(M, K)`, `stride=(K, 1)`. Element budget preserved.
2. `unary_rowwise_sum(x_sq)` → `accum_add(x_sq, rank=1)`. The K dim is now the
   innermost stream dim, so `rank=1` reduces exactly it.
3. The implicit `(tile_m, K) × (tile_m, 1)` broadcast inside `binary_mul`
   becomes an explicit `repeat_static(inv, factor=K)`. After `repeat_static`,
   both operands have stream shape `(M, K)`.

---

## Example 2 — qkv_projection (matmul → map_accum)

Reference: `y = x @ W`, `x ∈ (B, D)`, `W ∈ (D, proj_dim)`.

### Before (tile `(tile_b=16, D=128)` for `x`, `(D=128, tile_proj=16)` for `W`)
```python
def tiled_reference(dims, tensors):
    B, D, proj_dim = dims["B"], dims["D"], dims["proj_dim"]
    tile_b, tile_proj = dims["tile_b"], dims["tile_proj"]
    proj_tiles = proj_dim // tile_proj

    x_stream = offchip_load(
        tensors["x"],
        stride=(0, 0),
        out_shape_tiled=(1, proj_tiles),         # x's single tile broadcast across proj tiles
        tile_row=tile_b,
        tile_col=D,                              # D lives inside the tile
    )
    w_stream = offchip_load(
        tensors["W"],
        stride=(0, 1),
        out_shape_tiled=(1, proj_tiles),
        tile_row=D,                              # D lives inside the tile
        tile_col=tile_proj,
    )
    out = binary_matmul(x_stream, w_stream)      # intra-tile matmul reduces over D
    return offchip_store(out)
```

### After (tile `(1, 1)`)
```python
def tiled_reference(dims, tensors):
    B, D, proj_dim = dims["B"], dims["D"], dims["proj_dim"]
    stream_shape = (B, proj_dim, D)              # D is now an innermost stream dim

    x_stream = offchip_load(
        tensors["x"],
        stride=(D, 0, 1),                        # B walks rows, proj_dim broadcast, D walks cols
        out_shape_tiled=stream_shape,
        tile_row=1,
        tile_col=1,
    )
    w_stream = offchip_load(
        tensors["W"],
        stride=(0, 1, proj_dim),                 # B broadcast, proj_dim walks cols, D walks rows
        out_shape_tiled=stream_shape,
        tile_row=1,
        tile_col=1,
    )
    out = binary_map_accum(x_stream, w_stream, rank=1)   # rule: matmul → map_accum over the evicted reduction dim
    return offchip_store(out)
```

Transforms applied:
1. The reduction dim **D** moves from the tile into the *innermost* stream
   position. Both operands' `out_shape_tiled` grow to `(B, proj_dim, D)`.
2. `stride` is rebuilt per operand in 1×1 grid coordinates. The dim each
   operand doesn't index becomes a `0` stride (broadcast).
3. `binary_matmul` → `binary_map_accum(rank=1)`. With 1×1 tiles each multiply
   is scalar; `rank=1` then sums over the innermost stream dim (D). No
   explicit broadcast op is needed because both operands already share
   `stream_shape`.

---

## Example 3 — sdpa_core_max (full softmax + a stream-reorder trick)

Reference: `out = softmax(Q @ Kᵀ) @ V` with row-wise-max-stable softmax.

### Before (one tile per matrix: `Q` is `(M, D)`, `K, V` are `(N, D)`)
```python
def tiled_reference(dims, tensors):
    M, N, D = dims["M"], dims["N"], dims["D"]
    Q = offchip_load(tensors["Q"], stride=(1,), out_shape_tiled=(1,), tile_row=M, tile_col=D)
    K = offchip_load(tensors["K"], stride=(1,), out_shape_tiled=(1,), tile_row=N, tile_col=D)
    V = offchip_load(tensors["V"], stride=(1,), out_shape_tiled=(1,), tile_row=N, tile_col=D)

    scores         = binary_matmul(Q, K, weight_transposed=True)           # tile (M, N)
    scores_split   = retile_streamify(scores, chunk=1, split_row=False)    # expose N as stream
    row_max        = accum_max(scores_split, rank=1)                       # tile (M, 1)
    row_max_exp    = repeat_static(row_max, factor=1)
    neg_max        = unary_mul_imm(row_max_exp, -1.0)
    shifted        = binary_add(scores, neg_max)                           # tile broadcast (M,N) ⊗ (M,1)
    exp_shifted    = unary_exp(shifted)
    norm           = unary_rowwise_sum(exp_shifted)                        # intra-tile reduce → (M, 1)
    context        = binary_matmul(exp_shifted, V)                         # tile (M, D)
    out            = binary_div(context, norm)                             # tile broadcast (M,D) ⊗ (M,1)
    return offchip_store(out)
```

### After (tile `(1, 1)`)
```python
def tiled_reference(dims, tensors):
    M, N, D = dims["M"], dims["N"], dims["D"]
    # Q broadcast across N, K broadcast across M; D is the reduction dim for the first matmul.
    Q = offchip_load(tensors["Q"], stride=(D, 0, 1), out_shape_tiled=(M, N, D), tile_row=1, tile_col=1)
    K = offchip_load(tensors["K"], stride=(0, D, 1), out_shape_tiled=(M, N, D), tile_row=1, tile_col=1)
    V = offchip_load(tensors["V"], stride=(0, D, 1), out_shape_tiled=(M, N, D), tile_row=1, tile_col=1)

    scores       = binary_map_accum(Q, K, rank=1)                          # reduce D → (M, N)
    row_max      = accum_max(scores, rank=1)                               # reduce N → (M,)

    # Broadcast row_max back to (M, N) for subtraction
    row_max_1    = reshape_stream(row_max, chunk_size=1, rank=0)           # (M, 1)
    row_max_b    = expand_ref(row_max_1, scores, expand_rank=1)            # (M, N)
    centered     = binary_add(scores, unary_mul_imm(row_max_b, -1.0))
    exp_scores   = unary_exp(centered)

    norm         = accum_add(exp_scores, rank=1)                           # reduce N → (M,)

    # context[m, d] = Σ_n exp_scores[m, n] * V[n, d]
    # exp_scores has stream (M, N) but V has (M, N, D); broadcast exp_scores across D.
    exp_expanded = expand_ref(
        reshape_stream(exp_scores, chunk_size=1, rank=0),
        V,
        expand_rank=1,
    )                                                                       # (M, N, D)
    weighted     = binary_mul(exp_expanded, V)                              # (M, N, D)

    # The reduction is over N but N is not innermost — flip via bufferize + streamify.
    buffered     = bufferize(weighted, rank=2)                              # stream (M), buffer (N, D)
    streamed     = streamify(buffered, stride=(1, D), out_shape_tiled=(D, N))  # stream (M, D, N)
    context      = accum_add(streamed, rank=1)                              # reduce N → (M, D)

    # Broadcast norm across D for the final divide.
    norm_1       = reshape_stream(norm, chunk_size=1, rank=0)               # (M, 1)
    norm_b       = expand_ref(norm_1, context, expand_rank=1)               # (M, D)
    out          = binary_div(context, norm_b)
    return offchip_store(out)
```

Transforms applied:
1. All three off-chip loads: 1×1 tiles, full `(M, N, D)` stream, stride-0 on
   the non-indexed dim of each operand.
2. `binary_matmul(Q, Kᵀ)` → `binary_map_accum(rank=1)` over D.
3. `retile_streamify(...) → accum_max(rank=1)` collapses to a direct
   `accum_max(scores, rank=1)` because N is already a stream dim.
4. Every tile-shape broadcast (`(M,1) → (M,N)`, `(M,1) → (M,D)`) becomes
   `reshape_stream + expand_ref`.
5. `unary_rowwise_sum` → `accum_add(rank=1)`.
6. `binary_matmul(exp_shifted, V)` does not directly translate, because after
   the broadcast `exp_scores * V` has stream `(M, N, D)` and the reduction is
   over **N**, not the innermost dim D. The `bufferize(rank=2) → streamify`
   pair reorders the stream to `(M, D, N)` so that `accum_add(rank=1)`
   reduces N as required. This is the most error-prone step — the
   `streamify` stride is in buffer tile-grid coordinates: `stride=(1, D)`
   maps `(d, n)` to linear index `n*D + d`, preserving the buffer's row-major
   layout.

---

## Pitfalls

- **Stride is in tile-grid units, not elements.** When you shrink tiles by
  factor `f` along an axis, the stride along that axis grows by `f`. After
  going all the way to 1×1, the stride tuple looks like the *element* layout
  (e.g. `(K, 1)`), not the original *tile-grid* layout (e.g. `(grid_c,)`).
- **Reductions always hit the innermost N stream dims.** If the dim you
  evicted lands somewhere other than innermost, plan a `bufferize → streamify`
  reorder before the `accum_*`. Don't try to fix this with `parallelize` or
  `flat_partition` — those have different semantics.
- **`repeat_static` and `expand_ref` aren't interchangeable.**
  `repeat_static(x, factor=k)` adds a fresh innermost stream dim of size `k`.
  `expand_ref(x, ref, expand_rank=k)` makes `x`'s stream shape match `ref`'s
  by extending it with the last `k` of `ref`'s dims. Use `expand_ref` when
  you already have a tensor with the target shape; use `repeat_static` when
  you need to broadcast against a static factor.
- **Shrinking a dim that was a singleton inside the tile is a no-op for
  memory and may break rank assertions.** If `tile_col=1` already, evicting
  it to the stream costs nothing and may trip `OffChipStore`/`ExpandRef`
  minimum-rank checks. Look for tile dims `> 1` first.
- **Partial shrinks are usually best.** Halving `tile_n` from 64 to 32 cuts
  the tile of a `(M, N)` intermediate in half while keeping every op
  intra-tile. Going all the way to 1×1 forces every reduction onto the
  stream and serializes compute. Reach for 1×1 only when no partial shrink
  fits.
- **Conservation check.** After the rewrite, the product of
  `out_shape_tiled` and `(tile_row, tile_col)` must still equal the
  underlying element shape (modulo broadcast dims, where the corresponding
  stride entry is 0).
