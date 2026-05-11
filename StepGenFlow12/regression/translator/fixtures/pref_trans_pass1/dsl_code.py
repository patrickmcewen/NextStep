# This node performs the pre‑attention RMSNorm and the Q, K, V linear projections.
#   1. Load the raw tensors as single‑tile streams.
#   2. Compute RMSNorm:  x * rsqrt(mean(x²) + eps)
#        - square, sum rows, scale by 1/hidden, add epsilon, rsqrt, then broadcast‑multiply.
#   3. Multiply the normalized activations with each projection weight (binary_matmul).
#   4. Reshape each result from (seq_len, heads*head_dim) tile to a stream of shape
#        (seq_len) with tile shape (heads, head_dim):
#        a) retile rows → move seq_len to a stream dimension.
#        b) retile columns → move num_heads to the same (innermost) stream dimension.
#        c) reshape_stream → split that combined stream dimension into (seq_len, num_heads).
#        d) accum_retile_row → absorb the num_heads stream dimension into the tile rows.
#        e) flatten → collapse the leading singleton stream dimension with seq_len,
#                     yielding the final stream shape (seq_len, heads, head_dim).
#   The three tensors Q, K, V are returned as StepTensors matching the required
#   output shapes [(64,16,32), (64,4,32), (64,4,32)].
def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------------
    # 1. Load raw inputs as single‑tile streams.
    # -----------------------------------------------------------------------
    seq_len = input_tensor.shape[0]          # 64
    hidden_dim = input_tensor.shape[1]       # 512
    head_dim = cos.shape[-1]                # 32 (from the supplied cosine tensor)

    # Load the input activation matrix.
    inp = offchip_load(
        input_tensor,
        stride=(0,),                # only one tile, stride irrelevant
        out_shape_tiled=(1,),       # single stream element
        tile_row=seq_len,
        tile_col=hidden_dim,
    )

    # Load projection weight matrices.
    q_w = offchip_load(
        q_proj,
        stride=(0,),
        out_shape_tiled=(1,),
        tile_row=hidden_dim,
        tile_col=q_proj.shape[1],
    )
    k_w = offchip_load(
        k_proj,
        stride=(0,),
        out_shape_tiled=(1,),
        tile_row=hidden_dim,
        tile_col=k_proj.shape[1],
    )
    v_w = offchip_load(
        v_proj,
        stride=(0,),
        out_shape_tiled=(1,),
        tile_row=hidden_dim,
        tile_col=v_proj.shape[1],
    )

    # -----------------------------------------------------------------------
    # 2. RMSNorm on the input.
    #    rms = x * rsqrt(mean(x^2) + eps)
    # -----------------------------------------------------------------------
    eps = 1e-6
    inv_hidden = 1.0 / hidden_dim

    # x^2
    sq = unary_square(inp)
    # sum over hidden dimension (row‑wise sum -> shape (seq_len, 1))
    sum_sq = unary_rowwise_sum(sq)
    # mean = sum / hidden_dim
    mean_sq = unary_mul_imm(sum_sq, constant=inv_hidden)
    # add epsilon
    mean_eps = unary_add_imm(mean_sq, constant=eps)
    # rsqrt -> scaling factor per token (shape (seq_len, 1))
    scale = unary_rsqrt(mean_eps)
    # broadcast multiply: normalized activation
    normed = binary_mul(inp, scale)

    # -----------------------------------------------------------------------
    # 3. Linear projections (matrix multiply).
    # -----------------------------------------------------------------------
    Q_raw = binary_matmul(normed, q_w)   # (1,1,seq_len, num_heads*head_dim)
    K_raw = binary_matmul(normed, k_w)   # (1,1,seq_len, num_kv_heads*head_dim)
    V_raw = binary_matmul(normed, v_w)   # (1,1,seq_len, num_kv_heads*head_dim)

    # -----------------------------------------------------------------------
    # Helper to reshape a projection from (seq_len, heads*head_dim) tile to
    # stream shape (seq_len) with tile (heads, head_dim).
    # -----------------------------------------------------------------------
    def reshape_proj(raw_proj, num_heads):
        # a) Move seq_len from tile rows into a stream dimension.
        step = retile_streamify(raw_proj, chunk=1, split_row=True)

        # b) Split the output column dimension (heads*head_dim) into heads.
        step = retile_streamify(step, chunk=head_dim, split_row=False)

        # c) Split the combined stream dimension (seq_len * num_heads) into
        #    (seq_len, num_heads).
        step = reshape_stream(step, chunk_size=num_heads, rank=0)

        # d) Absorb the innermost stream dim (num_heads) into tile rows.
        step = accum_retile_row(step, rank=1)

        # e) Collapse the leading singleton stream dim with seq_len.
        step = flatten(step, min_rank=0, max_rank=1)
        return step

    # Number of heads for Q and K/V.
    num_heads = q_proj.shape[1] // head_dim          # 16
    num_kv_heads = k_proj.shape[1] // head_dim       # 4

    # -----------------------------------------------------------------------
    # 4. Apply the reshaping pipeline to each projection.
    # -----------------------------------------------------------------------
    Q = reshape_proj(Q_raw, num_heads)
    K = reshape_proj(K_raw, num_kv_heads)
    V = reshape_proj(V_raw, num_kv_heads)

    # Return the three projected tensors.
    return Q, K, V

"""
Implementation notes:
- RMSNorm is performed as  x * rsqrt(mean(x²) + eps).
  We compute x² with `unary_square`, sum across the last (head‑dim) axis
  with `unary_rowwise_sum`, scale by 1/head_dim, add eps, take rsqrt and
  finally multiply with the original tensor.
- `cos` and `sin` are RAW Off‑chip tensors. They are loaded with
  `offchip_load`. The loader always adds a leading singleton stream
  dimension, so we collapse the two stream dimensions (1, seq_len) back
  to a single one with `flatten`.
- The RoPE rotation `rotate_half(x)` is expressed using only DSL ops:
  * split the column dimension into two halves via `retile_streamify`,
    which turns the column size into a new stream factor (seq_len × 2)
    and shrinks the tile columns to `head_dim//2`.
  * `parallelize(..., 2)` separates the two halves into distinct streams.
  * negate the second half (`unary_mul_imm` with -1.0).
  * `static_reassemble` interleaves the streams as
    [‑second_half, first_half] → implements the required
    `concat([-x_half2, x_half1])`.
  * `reshape_stream` reshapes the combined stream (seq_len*2) into a
    two‑dimensional stream (seq_len, 2) so that the inner dimension can be
    merged into the tile columns with `accum_retile_col`.  This restores the
    original column size while keeping the stream mask identical to the
    original tensors.
- The same rotation logic is applied to both Q and K after RMSNorm.
- V is passed through unchanged.
- All tensor arithmetic is expressed via the DSL functions; the only
  Python arithmetic is on scalar meta‑data (shape sizes, constants).
"""
def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # ----- RMSNorm -------------------------------------------------
    def rms_norm(x):
        # x²
        sq = unary_square(x)
        # sum over head dimension (last tile dim), keep dim → (…, heads, 1)
        sum_sq = unary_rowwise_sum(sq)
        # mean = sum / head_dim
        head_dim = x.shape[-1]               # static integer
        inv_head_dim = 1.0 / head_dim
        mean_sq = unary_mul_imm(sum_sq, inv_head_dim)
        # add epsilon and rsqrt
        eps = 1e-6
        mean_eps = unary_add_imm(mean_sq, eps)
        rsqrt = unary_rsqrt(mean_eps)
        # x * rsqrt (broadcast over head dim)
        return binary_mul(x, rsqrt)

    Q_norm = rms_norm(Q)
    K_norm = rms_norm(K)

    # ----- Load cos / sin (RAW) ------------------------------------
    # tile shape is (1, head_dim)
    tile_row = 1
    tile_col = Q_norm.shape[-1]   # head_dim (32)
    stride = (1,)
    out_shape = (Q_norm.shape[0],)   # seq_len (64)

    cos_loaded = offchip_load(cos, stride=stride,
                              out_shape_tiled=out_shape,
                              tile_row=tile_row, tile_col=tile_col)
    cos_loaded = flatten(cos_loaded, min_rank=0, max_rank=1)   # (seq_len,1,head_dim)

    sin_loaded = offchip_load(sin, stride=stride,
                              out_shape_tiled=out_shape,
                              tile_row=tile_row, tile_col=tile_col)
    sin_loaded = flatten(sin_loaded, min_rank=0, max_rank=1)   # (seq_len,1,head_dim)

    # ----- Helper to perform rotate_half using only DSL ops --------
    def rotate_half(x):
        half = x.shape[-1] // 2                     # split columns
        # split columns into two stream elements per token
        split = retile_streamify(x, chunk=half, split_row=False)
        # separate the two halves (first, second)
        halves = parallelize(split, 2)                     # [first, second]
        first_half = halves[0]
        second_half = halves[1]
        # negate the second half
        second_neg = unary_mul_imm(second_half, -1.0)
        # interleave as [‑second, first] to achieve rotate_half
        interleaved = static_reassemble([second_neg, first_half])
        # reshape stream (seq_len*2) → (seq_len, 2)
        reshaped = reshape_stream(interleaved, chunk_size=2, rank=0, add_outer_dim=False)
        # merge the inner stream dim into tile columns → restores original cols
        merged = accum_retile_col(reshaped, rank=1)
        return merged

    # ----- Apply RoPE to Q and K -----------------------------------
    Q_rot = rotate_half(Q_norm)
    K_rot = rotate_half(K_norm)

    # Q' = Q_norm * cos + Q_rot * sin
    Q_cos = binary_mul(Q_norm, cos_loaded)
    Q_sin = binary_mul(Q_rot, sin_loaded)
    Q_out = binary_add(Q_cos, Q_sin)

    # K' = K_norm * cos + K_rot * sin
    K_cos = binary_mul(K_norm, cos_loaded)
    K_sin = binary_mul(K_rot, sin_loaded)
    K_out = binary_add(K_cos, K_sin)

    # V is unchanged
    return Q_out, K_out, V

# The pre‑attention node simply wires together the two child blackboxes.
# All inputs are RAW (off‑chip), so we must **not** apply any DSL consumer
# (binary_*, unary_*, …) before the first child.  The blackbox stubs handle
# any necessary off‑chip loads internally, so we can pass the raw tensors
# directly.  The expected output shapes of both children are the same as the
# node’s own output shapes, therefore we forward `out_shapes` (and the
# optional `out_perms`) unchanged to each call.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # 1) RMS‑norm and QKV projection.
    Q_pre, K_pre, V_pre = pre_attn_norm_and_proj(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # 2) Per‑head RMS‑norm and RoPE.
    Q, K, V = per_head_norm_and_rope(
        Q_pre,
        K_pre,
        V_pre,
        cos,
        sin,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return Q, K, V

# Implementation reasoning:
# - Q, K, V arrive as on‑chip streams:
#   Q: stream(64,) tile(16,32)   (seq_len × num_heads × head_dim)
#   K, V: stream(64,) tile(4,32) (seq_len × num_kv_heads × head_dim)
# - The reference implementation reshapes Q so that the 16 heads are split into
#   (kv_heads=4, query_per_kvhead=4) and makes the sequence length the tile
#   dimension.  K and V move the 4‑head dimension into a stream slot and also
#   make the sequence length a tile dimension.
# - The DSL lacks a direct permute, but an arbitrary permutation can be built
#   with the pattern: promote → retile_streamify (move tile rows into the
#   stream) → bufferize (turn stream into a flat buffer) → streamify (read the
#   buffer back with a stride that implements the desired permutation) →
#   accum_retile_row (absorb the seq_len stream dimension back into the tile
#   rows).
# - For Q we need two new stream dimensions (kv_heads, query_per_kvhead) and
#   to keep seq_len as a tile row.  For K/V we need one new stream dimension
#   (kv_heads) plus a singleton dimension, then also absorb seq_len.
# - Stride vectors implement the index mapping:
#   * Q: linear_idx = kv*4 + q*1 + seq*16  → stride = (4, 1, 16)
#   * K/V: linear_idx = kv*1 + seq*4      → stride = (1, 0, 4) (the middle
#     dimension is size‑1, so its stride can be 0)
# - Finally, `accum_retile_row` merges the innermost stream (seq_len) into the
#   tile rows, yielding the exact shapes required by the contract.

def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Number of KV heads (the tile‑row size of K/V)
    kv_heads = int(K.shape[-2])  # == 4 for this problem

    # -----------------------------------------------------------------
    # Q: (seq_len, num_heads, head_dim) → (kv, q_per_kv, seq_len, head_dim)
    # -----------------------------------------------------------------
    q_tmp = promote(Q, rank=0)                     # stream (64, 1), tile (16, 32)
    q_tmp = retile_streamify(q_tmp, chunk=1)       # stream (64, 16), tile (1, 32)
    q_buf = bufferize(q_tmp, rank=2)               # Buffer(shape=(64, 16))
    q_tmp = streamify(
        q_buf,
        stride=(kv_heads, 1, 16),                  # (4, 1, 16)
        out_shape_tiled=(kv_heads, kv_heads, 64)   # (4, 4, 64)
    )
    Q_out = accum_retile_row(q_tmp, rank=1)        # absorb seq_len → tile rows=64

    # -----------------------------------------------------------------
    # K: (seq_len, kv_heads, head_dim) → (kv_heads, 1, seq_len, head_dim)
    # -----------------------------------------------------------------
    k_tmp = promote(K, rank=0)                     # stream (64, 1), tile (4, 32)
    k_tmp = retile_streamify(k_tmp, chunk=1)       # stream (64, 4), tile (1, 32)
    k_buf = bufferize(k_tmp, rank=2)               # Buffer(shape=(64, 4))
    k_tmp = streamify(
        k_buf,
        stride=(1, 0, kv_heads),                   # (1, 0, 4)
        out_shape_tiled=(kv_heads, 1, 64)          # (4, 1, 64)
    )
    K_out = accum_retile_row(k_tmp, rank=1)        # absorb seq_len → tile rows=64

    # -----------------------------------------------------------------
    # V: same transformation as K
    # -----------------------------------------------------------------
    v_tmp = promote(V, rank=0)                     # stream (64, 1), tile (4, 32)
    v_tmp = retile_streamify(v_tmp, chunk=1)       # stream (64, 4), tile (1, 32)
    v_buf = bufferize(v_tmp, rank=2)               # Buffer(shape=(64, 4))
    v_tmp = streamify(
        v_buf,
        stride=(1, 0, kv_heads),                   # (1, 0, 4)
        out_shape_tiled=(kv_heads, 1, 64)          # (4, 1, 64)
    )
    V_out = accum_retile_row(v_tmp, rank=1)        # absorb seq_len → tile rows=64

    return Q_out, K_out, V_out

# attention_compute
# ----------------------------------------------------------------------
# Compute full‑sequence GQA attention using only DSL primitives.
#   • Qh, Kh, Vh are already on‑chip streams.
#   • Stable softmax is implemented by extracting the per‑row maximum,
#     subtracting it, exponentiating, then normalising.
#   • The final layout must be (seq_len, num_heads, head_dim) = (64, 16, 32).
#     To achieve the required ordering we:
#       1) Promote the attention‑output tile rows into a stream dimension.
#       2) Bufferize the three stream dimensions (kv, query‑per‑kv, seq).
#       3) Streamify the buffer with a custom stride that emits the
#          dimensions in the order (seq, kv, query‑per‑kv).
#       4) Flatten the inner two stream dimensions (kv, query‑per‑kv) into
#          the head dimension.
#       5) Absorb that head‑stream dimension back into the tile‑row size.
# ----------------------------------------------------------------------
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # 1️⃣ Broadcast K and V to the Qh stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # 2️⃣ Scores = Qh @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # ------------------------------------------------------------------
    # Stable soft‑max (row‑wise over the key dimension)
    # ------------------------------------------------------------------
    # a) Promote to expose a singleton innermost stream dim.
    scores_prom = promote(scores, rank=0)

    # b) Split the *column* tile (size 64) into a fresh stream dim.
    scores_split = retile_streamify(scores_prom, chunk=1, split_row=False)

    # c) Max‑reduce over that new stream dim → row_max (shape …×tile(64,1)).
    row_max = accum_max(scores_split, rank=1)

    # d) Subtract the max: scores – row_max
    row_max_neg = unary_mul_imm(row_max, -1.0)
    scores_centered = binary_add(scores, row_max_neg)

    # e) Exponential.
    e = unary_exp(scores_centered)

    # ------------------------------------------------------------------
    # Numerator = e @ V   ;   Denominator = Σₖ e   ;   Attention = num / denom
    # ------------------------------------------------------------------
    num   = binary_matmul(e, Vh_exp, weight_transposed=False)
    denom = unary_rowwise_sum(e)
    attn  = binary_div(num, denom)          # shape: stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Reshape to vanilla (seq_len, num_heads, head_dim) = (64,16,32)
    # ------------------------------------------------------------------
    # a) Promote then split the row tile (seq_len) into a stream dim.
    attn_prom   = promote(attn, rank=0)                         # (4,4,1)×tile(64,32)
    attn_split  = retile_streamify(attn_prom, chunk=1, split_row=True)
    #    → stream (kv, qp, seq)×tile (1,32)

    # b) Bufferize the three stream dims so we can reorder them.
    attn_buf = bufferize(attn_split, rank=3)                    # Buffer(kv,qp,seq)

    # c) Streamify with a stride that yields order (seq, kv, qp).
    #    Buffer shape is (kv, qp, seq) = (4, 4, 64).
    kv_dim   = attn_buf.shape[0]      # 4
    qp_dim   = attn_buf.shape[1]      # 4
    seq_dim  = attn_buf.shape[2]      # 64
    #    Linear index = kv * (qp*seq) + qp * seq + seq
    stride_seq = 1
    stride_kv  = qp_dim * seq_dim   # 4 * 64 = 256
    stride_qp  = seq_dim            # 64
    attn_ord = streamify(
        attn_buf,
        stride=[stride_seq, stride_kv, stride_qp],
        out_shape_tiled=(seq_dim, kv_dim, qp_dim)      # (64,4,4)
    )
    #    → stream (seq, kv, qp)×tile (1,32)

    # d) Merge the inner two stream dims (kv, qp) → heads.
    attn_flat = flatten(attn_ord, min_rank=0, max_rank=1)   # stream (seq, heads)

    # e) Absorb the heads stream dim into the tile‑row dimension.
    attn_out = accum_retile_row(attn_flat, rank=1)          # stream (seq)×tile (heads,32)

    return attn_out

# The attention node simply chains the two child blackboxes that implement the
# model's QKV projection and the attention computation.  The inputs Q, K, V are
# already on‑chip (they are StepTensors produced by a sibling DSL op), so they
# can be fed directly to `compute_qkv`.  We provide the exact tile‑stream shapes
# that the child expects – these are the vanilla shapes of its three outputs,
# each expressed as a stream shape with the last two dimensions being the tile
# size.  The resulting Qh, Kh, Vh tensors are then passed to
# `attention_compute`, propagating the caller‑provided `out_shapes` and
# `out_perms` so the final output conforms to the contract.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # Qh, Kh, Vh = compute_qkv(Q, K, V)
    Qh, Kh, Vh = compute_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),   # Qh: (num_kv_heads, heads_per_kv, seq_len, head_dim)
            (4, 1, 64, 32),   # Kh: (num_kv_heads, 1, seq_len, head_dim)
            (4, 1, 64, 32),   # Vh: (num_kv_heads, 1, seq_len, head_dim)
        ),
        out_perms=None,
    )
    # attn = attention_compute(Qh, Kh, Vh)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn

# Implementation reasoning:
# 1. Call the `attention` child to obtain the attention output as a stream with
#    tile shape (num_heads, head_dim) = (16, 32).
# 2. Convert each tile (16,32) into a single‑row tile (1, 16*32=512) by:
#    - Splitting the tile rows into a new stream dimension (retile_streamify,
#      split_row=True)
#    - Splitting the tile columns similarly (split_row=False)
#    - Reshaping the long stream into (seq_len, 512) using `reshape_stream`
#    - Absorbing the inner stream dimension into the tile columns with
#      `accum_retile_col`, yielding shape (seq_len, 1, 512).
# 3. Load the projection weight (512 × 512) from off‑chip memory, broadcasting
#    it across the sequence dimension:
#    - `offchip_load` with `stride=(0,)` (same tile for every position) and
#      `out_shape_tiled=(seq_len,)` creates a stream shape (1, seq_len) tile
#      (512,512).
#    - `flatten` merges the leading singleton stream dimension with the seq_len
#      dimension, giving shape (seq_len, 512, 512).
# 4. Perform the matrix multiplication between the flattened attention and the
#    broadcast weight using `binary_matmul`, resulting in (seq_len, 1, 512).
# 5. Load the residual tensor (seq_len × 512) similarly, broadcasting each row
#    across the stream:
#    - `offchip_load` with `stride=(1,)` maps each stream element to the
#      corresponding row tile.
#    - `flatten` removes the leading singleton stream dim, yielding
#      shape (seq_len, 1, 512).
# 6. Add the residual to the projected attention with `binary_add` and return
#    the final stream tensor.

def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Attention block (vanilla shape (seq_len, num_heads, head_dim))
    attn = attention(
        Q, K, V,
        out_shapes=((Q.shape[0], Q.shape[1], Q.shape[2]),),
        out_perms=(None,),
    )  # stream: (seq_len, num_heads, head_dim) tile (16,32)

    # 2. Flatten tile (num_heads, head_dim) → (1, num_heads*head_dim)
    attn = retile_streamify(attn, chunk=1, split_row=True)   # → stream( seq_len*num_heads ) tile(1,32)
    attn = retile_streamify(attn, chunk=1, split_row=False)  # → stream( seq_len*num_heads*head_dim ) tile(1,1)
    attn = reshape_stream(
        attn,
        chunk_size=Q.shape[1] * Q.shape[2],   # 16*32 = 512
        rank=0,
    )  # → stream( seq_len, 512 ) tile(1,1)
    attn = accum_retile_col(attn, rank=1)  # → stream( seq_len ) tile(1,512)

    # 3. Load and broadcast the projection weight (512×512) across the sequence
    proj_w_loaded = offchip_load(
        o_proj_weight,
        stride=(0,),
        out_shape_tiled=(Q.shape[0],),   # seq_len = 64
        tile_row=512,
        tile_col=512,
        transposed=False,
    )
    proj_w = flatten(proj_w_loaded, 0, 1)  # merge leading 1 with seq_len → stream(seq_len) tile(512,512)

    # 4. Matrix multiplication: (seq_len, 1, 512) × (seq_len, 512, 512)
    projected = binary_matmul(attn, proj_w)  # → stream(seq_len) tile(1,512)

    # 5. Load and broadcast the residual (seq_len × 512)
    resid_loaded = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(Q.shape[0],),   # seq_len = 64
        tile_row=1,
        tile_col=512,
        transposed=False,
    )
    resid = flatten(resid_loaded, 0, 1)   # → stream(seq_len) tile(1,512)

    # 6. Add residual
    out = binary_add(projected, resid)    # → stream(seq_len) tile(1,512)

    return out

# RMSNorm (post‑attention) implemented with the DSL.
# -------------------------------------------------
# The input `res_add_0` is already an on‑chip stream tensor with shape
# (64, 1, 512) → stream dim 64, tile (1, 512).
# RMSNorm = x * rsqrt( mean(x²) + eps )
#
# 1. Square the tensor (element‑wise) – `unary_square`.
# 2. Sum across the column dimension of the tile (i.e. over the hidden dim).  
#    `unary_rowwise_sum` reduces dim=-1, keeping the tile‑row dim (1) and
#    producing a (64, 1, 1) stream.
# 3. Divide by the number of columns (512) to obtain the mean.  This is a
#    scalar multiplication with a constant, performed by `unary_mul_imm`.
# 4. Add epsilon (1e‑6) – `unary_add_imm`.
# 5. Compute the reciprocal square‑root – `unary_rsqrt`.
# 6. Multiply the original tensor by the rsqrt factor – `binary_mul`.
#
# The final tensor has the same stream and tile shape as the input,
# matching the required output shape (64, 1, 512).

def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # 1) x²
    sq = unary_square(res_add_0)
    # 2) sum over hidden dimension (tile columns)
    sum_sq = unary_rowwise_sum(sq)               # shape (64, 1, 1)
    # 3) mean = sum / 512  (512 = tile column size)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / 512.0)  # divide by constant
    # 4) add epsilon
    mean_eps = unary_add_imm(mean_sq, 1e-6)
    # 5) rsqrt
    rsqrt = unary_rsqrt(mean_eps)
    # 6) x * rsqrt
    out = binary_mul(res_add_0, rsqrt)
    return out

def moe_dispatch__root_moe_moe_dispatch(
    normed_2,
    w_gate,
    w_up,
    w_down,
    expert_weights,
    expert_onehot,
    *,
    out_shapes,
    out_perms=None,
):
    # -----------------------------------------------------------------
    # 1️⃣  Routing selectors (one‑hot for masking, Index for address calc)
    # -----------------------------------------------------------------
    control = select_gen(expert_onehot, is_multihot=True, n=8)   # MultiHot(8)
    expert_idx = select_gen(expert_onehot, is_multihot=False, n=8)  # Index(8)

    # -----------------------------------------------------------------
    # 2️⃣  Address streams for the three expert weight tensors
    # -----------------------------------------------------------------
    gate_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    up_addr   = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)
    down_addr = expert_addr_gen(expert_idx, expert_addr_base=0, num_tile_per_expert=1)

    # -----------------------------------------------------------------
    # 3️⃣  Load the per‑expert weight tiles (one tile per expert)
    # -----------------------------------------------------------------
    gate_tile = random_offchip_load(w_gate, gate_addr, tile_row=512, tile_col=1792)
    up_tile   = random_offchip_load(w_up,   up_addr,   tile_row=512, tile_col=1792)
    down_tile = random_offchip_load(w_down, down_addr, tile_row=1792, tile_col=512)

    # -----------------------------------------------------------------
    # 4️⃣  Broadcast the normalized activations to the address‑stream shape:
    #     (1, seq_len, 2, 1, 1) with tile (1, 512)
    # -----------------------------------------------------------------
    # normed_2 is already on‑chip: stream (64) × tile (1,512)
    act = promote_outer(normed_2)          # → (1, 64) × tile(1,512)
    act = promote(act, rank=0)             # → (1, 64, 1) × tile(1,512)
    act = promote(act, rank=0)             # → (1, 64, 1, 1) × tile(1,512)
    act = promote(act, rank=0)             # → (1, 64, 1, 1, 1) × tile(1,512)
    act = expand_ref(act, gate_addr, expand_rank=3)  # → (1,64,2,1,1) × tile(1,512)

    # -----------------------------------------------------------------
    # 5️⃣  Gate and up projections
    # -----------------------------------------------------------------
    gate_out = binary_matmul(act, gate_tile)   # tile (1,1792)
    up_out   = binary_matmul(act, up_tile)     # tile (1,1792)

    # -----------------------------------------------------------------
    # 6️⃣  SiLU non‑linearity and hidden computation
    # -----------------------------------------------------------------
    gate_act = unary_silu(gate_out)
    hidden   = binary_mul(gate_act, up_out)

    # -----------------------------------------------------------------
    # 7️⃣  Down projection
    # -----------------------------------------------------------------
    down_out = binary_matmul(hidden, down_tile)   # tile (1,512)

    # -----------------------------------------------------------------
    # 8️⃣  Load per‑token expert‑weight scalars and broadcast them
    # -----------------------------------------------------------------
    w = offchip_load(
        expert_weights,
        stride=[2, 1],
        out_shape_tiled=[normed_2.shape[0], 2],
        tile_row=1,
        tile_col=1,
    )
    w = promote(w, rank=0)   # → (1, seq_len, 2, 1) × tile(1,1)
    w = promote(w, rank=0)   # → (1, seq_len, 2, 1, 1) × tile(1,1)

    # -----------------------------------------------------------------
    # 9️⃣  Apply the expert weights
    # -----------------------------------------------------------------
    weighted_down = binary_mul(down_out, w)   # tile (1,512)

    # -----------------------------------------------------------------
    # 🔟  Reduce over top‑position dimension (and the two singletons)
    # -----------------------------------------------------------------
    summed = accum_add(weighted_down, rank=3)   # stream (1, seq_len)

    # -----------------------------------------------------------------
    # 1️⃣1️⃣  Flatten leading singleton → final shape (seq_len, 1, 512)
    # -----------------------------------------------------------------
    moe_output = flatten(summed, min_rank=0, max_rank=1)

    return moe_output

# The MoE node first normalises the residual stream with RMS‑Norm, then
# forwards the normalised tensor together with the raw MoE parameters to
# the `moe_dispatch__root_moe_moe_dispatch` child.  Both children expect
# vanilla tensors; the black‑box stubs internally flatten the input
# streams to vanilla shape and re‑tile the outputs according to the
# `out_shapes`/`out_perms` supplied by the parent.  Since the weight
# tensors (`w_gate`, `w_up`, `w_down`, `expert_weights`,
# `expert_onehot`) are marked RAW they can be passed directly to the MoE
# child without an off‑chip load.  The required output stream shape is
# (64, 1, 512), which we forward unchanged to the children.
def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot,
                 *, out_shapes, out_perms=None):
    # Apply RMS‑Norm to the on‑chip residual stream.
    normed_2 = rms_norm(
        res_add_0,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Dispatch through the mixture‑of‑experts layer using the raw expert
    # parameters.  The child returns the final streamed tensor.
    result = moe_dispatch__root_moe_moe_dispatch(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return result

# The MoE node simply forwards its inputs to the `moe_dispatch` child,
# requesting the output stream shape defined by the parent's contract
# (a tiled shape (64, 1, 512) in this case).  The child handles loading the
# RAW weight and routing tensors internally, so we do not apply any
# `offchip_load` here.  After obtaining the dispatched tensor we add the
# residual (`res_add_0`) using the DSL `binary_add` operator, which
# preserves the stream metadata and yields the final tensor matching the
# required output shape.
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # Compute the MoE dispatch output with the expected tiling.
    moe_out = moe_dispatch(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Add the residual connection.
    return binary_add(moe_out, res_add_0)

# Implementation reasoning:
# - The root node receives all raw model tensors in the `tensors` dict.
# - Each pipeline stage is delegated to its corresponding blackbox:
#   * `pre_attention` produces the Q, K, V streams.
#   * `attention_o_proj` consumes Q, K, V together with the output projection
#     weight and the original input tensor (as residual) and produces the
#     intermediate residual stream.
#   * `moe` consumes the residual stream along with the MoE parameters and
#     yields the final transformer output.
# - All blackboxes expect *tile‑stream* shapes for their outputs:
#   the trailing two dimensions are tile rows/cols, the leading dimensions
#   form the streaming shape.  Using the vanilla shapes directly satisfies
#   this contract (e.g. Q has vanilla shape (64,16,32) → stream shape
#   (64,16,32) with tile 16×32).
# - No tensor‑method transformations are performed between raw inputs and
#   blackbox calls, satisfying the “no transform” rule.
# - The final stream is written off‑chip via `offchip_store`, which returns a
#   raw torch.Tensor as required for the root.
def tiled_reference(dims, tensors):
    # Raw off‑chip inputs
    input_tensor   = tensors["input_tensor"]
    q_proj         = tensors["q_proj"]
    k_proj         = tensors["k_proj"]
    v_proj         = tensors["v_proj"]
    cos            = tensors["cos"]
    sin            = tensors["sin"]
    o_proj_weight  = tensors["o_proj_weight"]
    w_gate         = tensors["w_gate"]
    w_up           = tensors["w_up"]
    w_down         = tensors["w_down"]
    expert_weights = tensors["expert_weights"]
    expert_onehot  = tensors["expert_onehot"]

    # 1️⃣ Pre‑attention: produce Q, K, V streams.
    Q, K, V = pre_attention(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(
            (64, 16, 32),   # Q: stream dim 64, tile 16×32
            (64,  4, 32),   # K: stream dim 64, tile  4×32
            (64,  4, 32),   # V: stream dim 64, tile  4×32
        ),
        out_perms=(None, None, None),
    )

    # 2️⃣ Attention + O‑projection + residual addition.
    #   Output shape (64, 512) is expressed as stream (64, 1, 512).
    res_add_0 = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
        out_shapes=((64, 1, 512),),
        out_perms=(None,),
    )

    # 3️⃣ MoE (Mixture‑of‑Experts) final layer.
    out = moe(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=((64, 1, 512),),
        out_perms=(None,),
    )

    # Off‑chip write of the final result.
    return offchip_store(out)