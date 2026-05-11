def pre_attention_and_qkv(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    """
    Leaf node that implements the RMSNorm followed by Q/K/V projections.
    All off‑chip tensors are streamed onto‑chip with `offchip_load`.  The
    RMSNorm is built from primitive unary / binary DSL ops.  The three
    projection matrices are streamed with a stride that advances across the
    head dimension.  After the per‑head matrix multiplies we collapse the
    leading singleton stream dimension and the tile‑row dimension (which are
    both size 1) so that the final tensors have the exact vanilla shapes
    required by the contract: (S, H, D).

    The hidden dimension determines the model geometry (only the Mixtral‑small
    configuration is exercised in the test suite):
        hidden_dim = 512 → head_dim=32, num_heads=16, num_kv_heads=4
    """
    # ------------------------------------------------------------------
    # 1️⃣  Infer model geometry from the hidden dimension.
    # ------------------------------------------------------------------
    seq_len = input_tensor.shape[0]      # S
    hidden_dim = input_tensor.shape[1]   # H

    if hidden_dim == 512:                 # Mixtral‑small
        head_dim = 32
        num_heads = 16
        num_kv_heads = 4
    elif hidden_dim == 4096:              # Qwen‑30B (example)
        head_dim = 128
        num_heads = 32
        num_kv_heads = 8
    else:
        raise AssertionError(f"Unsupported hidden_dim {hidden_dim}")

    # ------------------------------------------------------------------
    # 2️⃣  Stream the activation for the Q‑path (one stream per query head).
    # ------------------------------------------------------------------
    X_q = offchip_load(
        input_tensor,
        stride=[1, 0],                     # replicate across heads
        out_shape_tiled=(seq_len, num_heads),
        tile_row=1,
        tile_col=hidden_dim,
    )  # (1, S, HN, 1, H)

    # RMSNorm on Q‑path
    X_q_sq = unary_square(X_q)                         # (1, S, HN, 1, H)
    sum_sq_q = unary_rowwise_sum(X_q_sq)               # (1, S, HN, 1, 1)
    mean_sq_q = unary_mul_imm(sum_sq_q, 1.0 / hidden_dim)
    eps_q = unary_add_imm(mean_sq_q, 1e-6)
    inv_sqrt_q = unary_rsqrt(eps_q)
    normed_q = binary_mul(X_q, inv_sqrt_q)              # (1, S, HN, 1, H)

    # ------------------------------------------------------------------
    # 3️⃣  Stream the activation for the KV‑path (one stream per KV head).
    # ------------------------------------------------------------------
    X_kv = offchip_load(
        input_tensor,
        stride=[1, 0],                     # same stride, different head count
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=1,
        tile_col=hidden_dim,
    )  # (1, S, KV, 1, H)

    # RMSNorm on KV‑path
    X_kv_sq = unary_square(X_kv)
    sum_sq_kv = unary_rowwise_sum(X_kv_sq)
    mean_sq_kv = unary_mul_imm(sum_sq_kv, 1.0 / hidden_dim)
    eps_kv = unary_add_imm(mean_sq_kv, 1e-6)
    inv_sqrt_kv = unary_rsqrt(eps_kv)
    normed_kv = binary_mul(X_kv, inv_sqrt_kv)            # (1, S, KV, 1, H)

    # ------------------------------------------------------------------
    # 4️⃣  Stream the projection matrices.
    # ------------------------------------------------------------------
    QW = offchip_load(
        q_proj,
        stride=[0, 1],
        out_shape_tiled=(seq_len, num_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )  # (1, S, HN, H, hd)

    KW = offchip_load(
        k_proj,
        stride=[0, 1],
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )  # (1, S, KV, H, hd)

    VW = offchip_load(
        v_proj,
        stride=[0, 1],
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )  # (1, S, KV, H, hd)

    # ------------------------------------------------------------------
    # 5️⃣  Compute Q, K, V.
    # ------------------------------------------------------------------
    Q = binary_matmul(normed_q, QW)   # (1, S, HN, 1, hd)
    K = binary_matmul(normed_kv, KW) # (1, S, KV, 1, hd)
    V = binary_matmul(normed_kv, VW) # (1, S, KV, 1, hd)

    # ------------------------------------------------------------------
    # 6️⃣  Collapse the leading singleton stream dimension and the tile‑row
    #     dimension (both size 1) so that the tensors have the vanilla
    #     shapes declared by the contract.
    # ------------------------------------------------------------------
    Q = accum_retile_row(Q, rank=1)                 # (1, S, HN, hd)
    Q = flatten(Q, min_rank=0, max_rank=1)          # (S, HN, hd)

    K = accum_retile_row(K, rank=1)                 # (1, S, KV, hd)
    K = flatten(K, min_rank=0, max_rank=1)          # (S, KV, hd)

    V = accum_retile_row(V, rank=1)                 # (1, S, KV, hd)
    V = flatten(V, min_rank=0, max_rank=1)          # (S, KV, hd)

    return Q, K, V

# Per‑head RMSNorm implemented with DSL ops only.
# For each input tensor (shape: stream × heads × head_dim) we compute:
#   factor = 1 / sqrt(mean(x**2, dim=-1, keepdim=True) + eps)
#   output = x * factor
#   mean is obtained by summing over the last tile dimension and scaling by 1/head_dim.
# All arithmetic uses the provided DSL primitives; no raw tensor ops are used.
def per_head_norm(Q, K, *, out_shapes, out_perms=None):
    eps = 1e-6

    # ----- Q -------------------------------------------------
    # x²
    Q_sq = unary_square(Q)                       # (..., heads, head_dim)
    # Σ x² over head_dim (keepdim -> last dim = 1)
    Q_sum = unary_rowwise_sum(Q_sq)              # (..., heads, 1)
    # mean = Σ x² / head_dim
    Q_mean = unary_mul_imm(Q_sum, 1.0 / Q.shape[-1])
    # mean + eps
    Q_mean_eps = unary_add_imm(Q_mean, eps)
    # 1 / sqrt(mean + eps)
    Q_factor = unary_rsqrt(Q_mean_eps)
    # Apply factor (broadcast over head_dim)
    Q_norm = binary_mul(Q, Q_factor)

    # ----- K -------------------------------------------------
    K_sq = unary_square(K)
    K_sum = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / K.shape[-1])
    K_mean_eps = unary_add_imm(K_mean, eps)
    K_factor = unary_rsqrt(K_mean_eps)
    K_norm = binary_mul(K, K_factor)

    return Q_norm, K_norm

# The node simply forwards its raw inputs to the two child blackboxes.
# 1) `pre_attention_and_qkv` computes Q, K, V from the raw tensors.  The
#    parent’s `out_shapes` already describe the desired stream shapes for
#    these three tensors, so we pass them unchanged.
# 2) `per_head_norm` applies RMSNorm to Q and K.  Its outputs have the
#    same shapes as the first two entries of `out_shapes`; we therefore
#    pass `out_shapes[:2]` (and the matching slice of `out_perms` if present).
# No tensor‑method calls or explicit loads are needed because the blackboxes
# accept raw off‑chip tensors directly and handle shape reconstruction
# internally.
def proj_and_norm(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    # Step 1: Q, K, V projection and RMSNorm (pre‑attention)
    Q, K, V = pre_attention_and_qkv(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # Step 2: Per‑head RMSNorm on Q and K
    Q_norm, K_norm = per_head_norm(
        Q,
        K,
        out_shapes=out_shapes[:2],
        out_perms=out_perms[:2] if out_perms is not None else None,
    )

    # Return the normalized Q, normalized K, and untouched V
    return Q_norm, K_norm, V

# Implementation reasoning:
# • `cos` and `sin` are RAW tensors, so we stream them on‑chip with `offchip_load`.
#   The underlying layout is (seq_len, 1, head_dim); we stream over the
#   sequence dimension with tiles (1, head_dim) and then collapse the leading
#   singleton stream dim using `flatten`, giving shape (seq_len,)×tile(1,head_dim).
# • RoPE’s rotate‑half operation can be expressed without any Python slicing or
#   `torch.cat` by:
#     1. Splitting the column dimension into two halves with `retile_streamify`,
#        which turns the two halves into a doubled stream dimension.
#     2. Using `parallelize` to separate the two halves into distinct streams.
#     3. Negating the original second half (`unary_mul_imm`) and then swapping
#        the order by interleaving the streams with `static_reassemble`.
#     4. Converting the interleaved stream back to a (seq_len, 2, …) shape with
#        `reshape_stream`.
#     5. Merging the “2” stream dimension into the column tile dimension via
#        `accum_retile_col`, which yields the rotated tensor.
# • The RoPE formula `out = x * cos + rotate_half(x) * sin` is then built with
#   the binary compute DSL ops.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Load RAW rotary embeddings onto‑chip and flatten to a single stream.
    # ------------------------------------------------------------------
    tile_row = 1
    tile_col = Q.shape[-1]                     # head_dim = 32
    cos_loaded = flatten(
        offchip_load(cos, stride=[1], out_shape_tiled=[64],
                     tile_row=tile_row, tile_col=tile_col),
        min_rank=0,
        max_rank=1,
    )
    sin_loaded = flatten(
        offchip_load(sin, stride=[1], out_shape_tiled=[64],
                     tile_row=tile_row, tile_col=tile_col),
        min_rank=0,
        max_rank=1,
    )

    # ------------------------------------------------------------------
    # Helper implementing the `_rotate_half` logic using only DSL ops.
    # ------------------------------------------------------------------
    def _rotate_half(x):
        half = x.shape[-1] // 2                     # Python scalar (16)

        # 1️⃣ Split the column dimension into two halves -> doubled stream.
        split = retile_streamify(x, chunk=half, split_row=False)   # (2*S, R, half)

        # 2️⃣ Separate the two halves into distinct streams.
        halves = parallelize(split, 2)                # each (S, R, half)
        first_half, second_half = halves[0], halves[1]

        # 3️⃣ Negate the original second half (will become the first half).
        second_half_neg = unary_mul_imm(second_half, -1.0)

        # 4️⃣ Interleave streams in swapped order: negated second half then first half.
        interleaved = static_reassemble([second_half_neg, first_half])  # (2*S, R, half)

        # 5️⃣ Reshape stream dim back to (S, 2, R, half).
        interleaved = reshape_stream(interleaved, chunk_size=2, rank=0)

        # 6️⃣ Merge the “2‑half” stream into the column tile dimension.
        out = accum_retile_col(interleaved)          # (S, R, 2*half) = (S, R, C)
        return out

    # ------------------------------------------------------------------
    # Apply RoPE to Q.
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_loaded)                # Q * cos
    Q_rot = _rotate_half(Q)                          # rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_loaded)        # rotate_half(Q) * sin
    Q_out = binary_add(Q_cos, Q_rot_sin)             # final Q_out

    # ------------------------------------------------------------------
    # Apply RoPE to K.
    # ------------------------------------------------------------------
    K_cos = binary_mul(K, cos_loaded)                # K * cos
    K_rot = _rotate_half(K)                          # rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_loaded)        # rotate_half(K) * sin
    K_out = binary_add(K_cos, K_rot_sin)             # final K_out

    return Q_out, K_out

# The pre_attention node simply composes the two child blackboxes.
# 1. proj_and_norm computes Q, K, V from the raw inputs.
# 2. rope applies rotary positional embeddings to Q and K.
# Both children handle any necessary off‑chip loads internally, so we
# just forward the raw tensors unchanged.  We slice the `out_shapes`
# and `out_perms` tuples to match each child’s output count.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # proj_and_norm produces three outputs: Q, K, V
    Q, K, V = proj_and_norm(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # rope produces two outputs: updated Q and K.
    # Use the first two shapes (and perms, if provided) from the node's contract.
    rope_out_shapes = (out_shapes[0], out_shapes[1])
    rope_out_perms = None
    if out_perms is not None:
        rope_out_perms = (out_perms[0], out_perms[1])

    Q, K = rope(
        Q,
        K,
        cos,
        sin,
        out_shapes=rope_out_shapes,
        out_perms=rope_out_perms,
    )

    return Q, K, V

# compute_e implements the soft‑max exponent (e = exp(scores - row_max)):
#   1. Broadcast Kh across the query‑per‑kv‑head dimension.
#   2. Compute raw attention scores = Qh @ Khᵀ via binary_matmul.
#   3. Move the column tile dimension (size 64) into the stream using
#      retile_streamify(split_row=False) so we can reduce over it.
#   4. Split that fused stream dimension into (query_per_kvhead, col_index)
#      with reshape_stream (chunk_size = seq_len = 64).
#   5. Reduce over the column‑index stream dimension with accum_max to get
#      the per‑row maximum (row_max).
#   6. Subtract row_max from the original scores (broadcast over the column
#      tile) and exponentiate the result.
# The final tensor has shape (4, 4, 64, 64), matching the declared output shape.
def compute_e(Qh, Kh, *, out_shapes, out_perms=None):
    # 1. Expand Kh so it has the same stream shape as Qh.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)          # (4, 4, 64, 32)

    # 2. Compute attention scores = Qh @ Khᵀ (tile: 64×64).
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4, 4, 64, 64)

    # 3. Pull the column tile (size 64) into the stream.
    scores_split = retile_streamify(scores, chunk=1, split_row=False)  # (4, 256, 64, 1)

    # 4. Split the fused stream dim (256) into (query_per_kvhead=4, col_index=64).
    scores_reshaped = reshape_stream(scores_split, chunk_size=64, rank=0)  # (4, 4, 64, 64, 1)

    # 5. Row‑wise max: reduce over the column‑index stream dimension.
    row_max = accum_max(scores_reshaped, rank=1)       # (4, 4, 64, 1)

    # 6. Shift scores by the max and exponentiate.
    shifted = binary_add(scores, unary_mul_imm(row_max, -1.0))
    e = unary_exp(shifted)

    return e

# compute_attn:
#  * `e` and `Vh` arrive already on‑chip with stream shapes (4,4) and (4,1)
#    respectively.  For a batched matrix multiplication the stream shapes must
#    match, so we first broadcast `Vh` across the second stream dimension using
#    `expand_ref`.  `expand_ref(Vh, e, expand_rank=1)` expands the trailing
#    stream dim of size 1 in `Vh` to size 4, yielding a tensor of shape
#    (4,4,64,32).
#  * Weighted sum of values: `binary_matmul(e, Vh_exp)` performs the per‑tile
#    matrix multiply, producing a tensor of shape (4,4,64,32).
#  * Denominator: sum the attention scores over the key dimension while keeping
#    the dimensions (`unary_rowwise_sum`).  This yields shape (4,4,64,1).
#  * Final attention: element‑wise division (`binary_div`) broadcasts the
#    denominator across the output’s column dimension, giving the required
#    output shape (4,4,64,32).
def compute_attn(e, Vh, *, out_shapes, out_perms=None):
    # Broadcast Vh across the query‑per‑kv‑head dimension to match e's stream.
    Vh_exp = expand_ref(Vh, e, expand_rank=1)

    # Weighted sum of values (batched matrix multiplication).
    num = binary_matmul(e, Vh_exp)

    # Denominator: sum over the last tile dimension (keep dim → size 1).
    denom = unary_rowwise_sum(e)

    # Attention output: element‑wise division (broadcast over the column dim).
    attn = binary_div(num, denom)

    return attn

# The attention kernel consists of two stages:
#   1. Compute the raw attention scores `e = Q·Kᵀ` via the `compute_e` child.
#   2. Apply the scores to the values `V` via the `compute_attn` child.
# All inputs are already on‑chip tiled streams, so they can be passed
# directly to the children.  The children need explicit `out_shapes`
# (and optionally `out_perms`) because they internally flatten the
# streams to vanilla tensors.  `compute_e` emits a `(4,4,64,64)` stream,
# and `compute_attn` produces the final `(4,4,64,32)` stream, which we
# return unchanged (the caller supplies the desired `out_shapes`/`out_perms`).
def attention_compute__root_attention_attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Stage 1: raw attention scores
    e = compute_e(Qh, Kh, out_shapes=((4, 4, 64, 64),), out_perms=(None,))
    # Stage 2: weighted sum of values
    attn = compute_attn(e, Vh, out_shapes=out_shapes, out_perms=out_perms)
    return attn

def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # --------------------------------------------------------------
    # Basic dimensions (plain Python scalars)
    # --------------------------------------------------------------
    seq_len          = Q.shape[0]               # 64
    num_heads        = Q.shape[1]               # 16
    head_dim         = Q.shape[2]               # 32

    num_kv_heads     = K.shape[1]               # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4

    # --------------------------------------------------------------
    # Build Qh : (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # --------------------------------------------------------------
    # 1) Turn the per‑token (head‑row) tile into a stream.
    Q_flat = retile_streamify(Q, chunk=1, split_row=True)          # → stream(seq_len* num_heads)×tile(1, head_dim)

    # 2) Split that flat stream back into individual heads (round‑robin).
    per_head = parallelize(Q_flat, n=num_heads)                    # list of `num_heads` tensors, each shape:
                                                                  #   stream(seq_len)×tile(1, head_dim)

    # 3) Concatenate the per‑head streams so that each head appears in a contiguous block.
    merged_q, _ = eager_merge(per_head)                             # → stream(num_heads * seq_len)×tile(1, head_dim)

    # 4) Reshape to (head, seq) stream.
    Q_hs = reshape_stream(merged_q, chunk_size=seq_len, rank=0)    # → stream(num_heads, seq_len)×tile(1, head_dim)

    # 5) Merge the inner stream (seq) into the tile rows → tile rows = seq_len.
    Q_tile = accum_retile_row(Q_hs, rank=1)                         # → stream(num_heads)×tile(seq_len, head_dim)

    # 6) Split the head stream into (num_kv_heads, query_per_kvhead).
    Qh = reshape_stream(Q_tile, chunk_size=query_per_kvhead, rank=0)  # → stream(num_kv_heads, query_per_kvhead)×tile(seq_len, head_dim)

    # --------------------------------------------------------------
    # Build Kh and Vh : (num_kv_heads, 1) × tile(seq_len, head_dim)
    # --------------------------------------------------------------
    def make_kv(x):
        # 1) Move the tile‑row dimension (kv‑heads) into the stream.
        flat = retile_streamify(x, chunk=1, split_row=True)       # → stream(seq_len * num_kv_heads)×tile(1, head_dim)

        # 2) Split the flat stream into per‑kv‑head streams.
        per_kv = parallelize(flat, n=num_kv_heads)                # list of `num_kv_heads` tensors,
                                                                  # each shape stream(seq_len)×tile(1, head_dim)

        # 3) Concatenate the per‑kv streams (kv‑major order).
        merged, _ = eager_merge(per_kv)                            # → stream(num_kv_heads * seq_len)×tile(1, head_dim)

        # 4) Reshape to (kv, seq) stream.
        kv_seq = reshape_stream(merged, chunk_size=seq_len, rank=0)  # → stream(num_kv_heads, seq_len)×tile(1, head_dim)

        # 5) Merge seq into tile rows.
        kv_tile = accum_retile_row(kv_seq, rank=1)                 # → stream(num_kv_heads)×tile(seq_len, head_dim)

        # 6) Add the required trailing singleton stream dimension.
        kv_tile = reshape_stream(kv_tile, chunk_size=1, rank=0)    # → stream(num_kv_heads, 1)×tile(seq_len, head_dim)
        return kv_tile

    Kh = make_kv(K)   # (num_kv_heads, 1)×tile(seq_len, head_dim)
    Vh = make_kv(V)   # (num_kv_heads, 1)×tile(seq_len, head_dim)

    # --------------------------------------------------------------
    # Heavy attention kernel (blackbox)
    # --------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),   # vanilla shape expected by the child
        ),
        out_perms=(None,),
    )   # → stream(num_kv_heads, query_per_kvhead)×tile(seq_len, head_dim)

    # --------------------------------------------------------------
    # Convert the child's output back to the contract shape:
    #   stream(seq_len) × tile(num_heads, head_dim)
    # --------------------------------------------------------------
    # 1) Collapse the (kv, qp) stream dimensions into a single head stream.
    attn_heads = flatten(attn, min_rank=0, max_rank=1)              # → stream(num_heads)×tile(seq_len, head_dim)

    # 2) Split the tile‑row dimension (seq_len) into a stream dimension.
    split = retile_streamify(attn_heads, chunk=1, split_row=True)   # → stream(num_heads, seq_len)×tile(1, head_dim)

    # 3) Re‑group tokens so that each stream corresponds to a single sequence position
    #    (i.e. turn (head, seq) → list of `seq_len` streams each of length `num_heads`).
    per_seq = parallelize(split, n=seq_len)                         # list of `seq_len` tensors,
                                                                  # each shape stream(num_heads)×tile(1, head_dim)

    # 4) Concatenate the per‑sequence streams back into a flat stream.
    merged_attn, _ = eager_merge(per_seq)                           # → stream(seq_len * num_heads)×tile(1, head_dim)

    # 5) Reshape to (seq_len, num_heads) stream.
    seq_head = reshape_stream(merged_attn, chunk_size=num_heads, rank=0)  # → stream(seq_len, num_heads)×tile(1, head_dim)

    # 6) Move the head dimension into the tile rows.
    out = accum_retile_row(seq_head, rank=1)                       # → stream(seq_len)×tile(num_heads, head_dim)

    return out

# The attention node computes:
#   1. Vanilla attention via the child `attention_compute`.
#   2. Flattens the (num_heads, head_dim) tile into a single row of length
#      `num_heads*head_dim` using `retile_streamify` + `reshape_stream`.
#   3. Loads the projection matrix as a tiled stream sized
#      (head_dim, hidden_dim) and broadcasts it across the sequence dimension.
#   4. Performs a per‑head matmul (`binary_matmul`) and sums over heads
#      (`accum_add`) to obtain the projected tensor.
#   5. Loads the residual `input_tensor` as a stream with tile (1, hidden_dim).
#   6. Adds the residual (`binary_add`) and collapses the leading singleton
#      stream dimension with `flatten` to produce shape (seq_len, 1, hidden_dim).
# All tensor arithmetic is expressed through DSL ops; off‑chip tensors are
# loaded before any compute consumer.

def attention(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Attention output: (seq_len, num_heads, head_dim)
    attn = attention_compute(
        Q, K, V,
        out_shapes=((Q.shape[0], Q.shape[1], Q.shape[2]),),
        out_perms=(None,),
    )

    # 2. Split the head dimension into a separate stream element.
    #    After retile_streamify each head becomes a distinct stream entry.
    attn_rows = retile_streamify(attn, chunk=1, split_row=True)               # (seq_len*num_heads, 1, head_dim)
    attn_rows = reshape_stream(attn_rows, chunk_size=Q.shape[1], rank=0)      # (seq_len, num_heads, 1, head_dim)
    attn_rows = promote_outer(attn_rows)                                     # (1, seq_len, num_heads, 1, head_dim)

    # 3. Load the projection matrix as a tiled stream.
    #    Tile size matches (head_dim, hidden_dim); stride (0,1) broadcasts across seq_len.
    weight = offchip_load(
        o_proj_weight,
        stride=(0, 1),
        out_shape_tiled=(Q.shape[0], Q.shape[1]),
        tile_row=Q.shape[2],               # head_dim
        tile_col=o_proj_weight.shape[1],   # hidden_dim
    )  # (1, seq_len, num_heads, head_dim, hidden_dim)

    # 4. Per‑head projection and sum over heads.
    proj = binary_matmul(attn_rows, weight)   # (1, seq_len, num_heads, 1, hidden_dim)
    proj = accum_add(proj, rank=1)            # (1, seq_len, 1, hidden_dim)

    # 5. Load the residual tensor (seq_len, hidden_dim) as (1, seq_len, 1, hidden_dim).
    residual = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(Q.shape[0],),
        tile_row=1,
        tile_col=o_proj_weight.shape[1],   # hidden_dim
    )  # (1, seq_len, 1, hidden_dim)

    # 6. Residual addition.
    summed = binary_add(proj, residual)       # (1, seq_len, 1, hidden_dim)

    # 7. Collapse the leading singleton stream dimension ⇒ (seq_len, 1, hidden_dim)
    out = flatten(summed, min_rank=0, max_rank=1)   # (seq_len, 1, hidden_dim)

    return out

# RMSNorm: y = x * rsqrt(mean(x**2) + eps)
# - Square the input.
# - Sum across the feature dimension (tile column) → shape (...,1,1).
# - Divide by the feature size to obtain the mean.
# - Add epsilon, take reciprocal square‑root, and multiply back onto the original tensor.
# The input `res_add_0` is already an on‑chip stream (shape (64, 1, 512)),
# so we can directly apply the DSL compute ops. The output shape matches the
# required `(64, 1, 512)`, thus no additional reshaping is needed.
def post_attn_rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # x²
    sq = unary_square(res_add_0)
    # Σ x² over the feature dimension (keepdim → (...,1,1))
    sum_sq = unary_rowwise_sum(sq)
    # mean = sum / feature_dim
    feature_dim = res_add_0.shape[-1]               # 512 in this case
    inv_feat = 1.0 / feature_dim                    # scalar constant
    mean_sq = unary_mul_imm(sum_sq, inv_feat)
    # mean + eps
    eps = 1e-6
    mean_eps = unary_add_imm(mean_sq, eps)
    # rsqrt(mean + eps)
    rsqrt = unary_rsqrt(mean_eps)
    # x * rsqrt(...)
    out = binary_mul(res_add_0, rsqrt)
    return out

# MoE compute (SwiGLU style):
#   1. Load per‑expert weight matrices (gate, up, down) from off‑chip.
#   2. Load per‑token routing scalars (expert_weights) from off‑chip.
#   3. Build a control mask from the one‑hot routing map (expert_onehot).
#   4. Replicate each token embedding for the two activated‑expert slots,
#      flatten token×slot → a single stream, and flatten the routing‑weight
#      stream the same way.
#   5. Partition the token stream and routing‑weight stream per expert using
#      the control mask.
#   6. Split the loaded weight matrices per expert.
#   7. For each expert:
#        * broadcast its gate/up/down matrices to the token stream with
#          `expand_ref`;
#        * compute the SwiGLU activation:
#            a = x @ w_gate
#            a = silu(a)
#            b = x @ w_up
#            h = a * b
#        * down‑project: out = h @ w_down;
#        * multiply by the per‑token routing weight.
#   8. Re‑assemble the per‑expert token streams with `flat_reassemble`,
#      flatten away the extra leading singleton dimension, reshape back to
#      (seq, slot, 1, dim), and finally sum over the two slots with
#      `accum_add`. The result is a stream tensor of shape (64, 1, 512) as
#      required by the parent.

def moe_compute(
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
    # ------------------------------------------------------------
    # 1. Load per‑expert weight tensors
    # ------------------------------------------------------------
    wg_raw = offchip_load(
        w_gate,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=512,
        tile_col=1792,
        par_dispatch=1,
    )
    w_gate_stream = flatten(wg_raw, min_rank=0, max_rank=1)   # (8, 512, 1792)

    wu_raw = offchip_load(
        w_up,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=512,
        tile_col=1792,
        par_dispatch=1,
    )
    w_up_stream = flatten(wu_raw, min_rank=0, max_rank=1)     # (8, 512, 1792)

    wd_raw = offchip_load(
        w_down,
        stride=(1,),
        out_shape_tiled=(8,),
        tile_row=1792,
        tile_col=512,
        par_dispatch=1,
    )
    w_down_stream = flatten(wd_raw, min_rank=0, max_rank=1)   # (8, 1792, 512)

    # ------------------------------------------------------------
    # 2. Load per‑token routing scalars (expert_weights)
    # ------------------------------------------------------------
    ew_raw = offchip_load(
        expert_weights,
        stride=(2, 1),                 # rows advance by 2 (the two slots)
        out_shape_tiled=(64, 2),
        tile_row=1,
        tile_col=1,
        par_dispatch=1,
    )
    # shape after offchip_load: (1, 64, 2, 1, 1) → flatten to (64, 2, 1, 1)
    expert_weights_stream = flatten(ew_raw, min_rank=1, max_rank=2)   # (64, 2, 1, 1)

    # ------------------------------------------------------------
    # 3. Build control mask from expert_onehot (int64)
    # ------------------------------------------------------------
    control_raw = select_gen(expert_onehot, is_multihot=False, n=8)   # (1,64)×tile(2,8)

    # split the tile‑row (slot) dimension into a stream dimension
    control_retile = retile_streamify(
        control_raw, chunk=1, split_row=True
    )  # (1,128)×tile(1,8)

    # merge leading singleton → (128, 1, 8)
    control = flatten(control_retile, min_rank=0, max_rank=1)  # (128, 1, 8)

    # ------------------------------------------------------------
    # 4. Replicate token embeddings for the two slots and flatten
    # ------------------------------------------------------------
    # add a singleton stream dim so we can expand to size‑2 using the weight stream as reference
    normed_extra = reshape_stream(
        normed_2, chunk_size=1, rank=0, add_outer_dim=False
    )  # (64, 1)×tile(1,512)

    # expand the singleton dim to size 2 using the routing‑weight stream as reference
    normed_exp = expand_ref(
        normed_extra, expert_weights_stream, expand_rank=1
    )  # (64, 2)×tile(1,512)

    # collapse token × slot into a single stream dimension
    normed_flat = flatten(normed_exp, min_rank=0, max_rank=1)  # (128, 1, 512)

    # ------------------------------------------------------------
    # 5. Partition tokens and routing weights per expert according to `control`
    # ------------------------------------------------------------
    tokens_per_expert = flat_partition(
        normed_flat, control, n=8
    )  # list[8] of (M_i, 1, 512)

    # flatten routing‑weight stream to match the flattened token stream
    weights_flat = flatten(expert_weights_stream, min_rank=0, max_rank=1)  # (128, 1, 1)
    weights_per_expert = flat_partition(
        weights_flat, control, n=8
    )  # list[8] of (M_i, 1, 1)

    # ------------------------------------------------------------
    # 6. Split weight matrices per expert
    # ------------------------------------------------------------
    gate_per_expert = parallelize(w_gate_stream, 8)   # each (1, 512, 1792)
    up_per_expert = parallelize(w_up_stream, 8)       # each (1, 512, 1792)
    down_per_expert = parallelize(w_down_stream, 8)   # each (1, 1792, 512)

    # ------------------------------------------------------------
    # 7. Compute expert outputs and apply routing weight
    # ------------------------------------------------------------
    expert_outputs = []
    for i in range(8):
        x = tokens_per_expert[i]        # (M_i, 1, 512)
        w = weights_per_expert[i]       # (M_i, 1, 1)

        # broadcast gate and up matrices to the token stream shape
        gate_exp = expand_ref(gate_per_expert[i], x, expand_rank=1)   # (M_i, 512, 1792)
        up_exp = expand_ref(up_per_expert[i], x, expand_rank=1)       # (M_i, 512, 1792)

        # SwiGLU: a = silu(x @ gate), b = x @ up, h = a * b
        a = binary_matmul(x, gate_exp)       # (M_i, 1, 1792)
        a = unary_silu(a)                    # (M_i, 1, 1792)   ← silu(gate)
        b = binary_matmul(x, up_exp)        # (M_i, 1, 1792)
        h = binary_mul(a, b)                 # (M_i, 1, 1792)

        # down projection (broadcast down matrix)
        down_exp = expand_ref(down_per_expert[i], h, expand_rank=1)   # (M_i, 1792, 512)
        out = binary_matmul(h, down_exp)               # (M_i, 1, 512)

        # apply the scalar routing weight (broadcast over tile cols)
        out_w = binary_mul(out, w)          # (M_i, 1, 512)

        expert_outputs.append(out_w)

    # ------------------------------------------------------------
    # 8. Re‑assemble token stream from per‑expert pieces
    # ------------------------------------------------------------
    merged = flat_reassemble(expert_outputs, control)   # (1, 128, 1, 1)×tile(1, 512)

    # collapse all stream dims into a single one (remove the leading singleton)
    merged_flat = flatten(merged, min_rank=0, max_rank=3)   # (128, 1, 512)

    # restore token × slot layout (64 tokens, 2 slots)
    token_pos = reshape_stream(
        merged_flat, chunk_size=2, rank=0, add_outer_dim=False
    )                                   # (64, 2, 1, 512)

    # sum the two routed slots per token
    final = accum_add(token_pos, rank=1)               # (64, 1, 512)

    return final

# The MoE root node is a thin wrapper around the `moe_compute` child.
# All inputs are either already on‑chip streams (e.g. `normed_2`) or raw
# off‑chip tensors (`w_gate`, `w_up`, `w_down`, `expert_weights`,
# `expert_onehot`).  Raw tensors may be passed directly to a child blackbox;
# the stub will handle any necessary off‑chip loads and reshape its vanilla
# output into the requested tiled shape.  Hence we simply forward the
# arguments together with the `out_shapes` and `out_perms` parameters.
def moe__root_moe(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    return moe_compute(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

# moe node implementation
#   1. Apply post‑attention RMSNorm via the provided child blackbox.
#   2. Run the MoE core (moe__root_moe) using the normalized tensor and the
#      routing / expert weight tensors (all RAW, the child can load them as needed).
#   3. Add the original residual back with a binary_add DSL operation.
#   The required output stream shape is supplied by the caller via `out_shapes`;
#   we reuse that shape for the intermediate children, since they all produce a
#   tensor of the same logical dimensions (seq_len × dim) tiled as (S,1,T_C).
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # 1) RMSNorm
    normed_2 = post_attn_rms_norm(
        res_add_0,
        out_shapes=(out_shapes[0],),    # stream shape for the normalized tensor
        out_perms=(None,),
    )

    # 2) Mixture‑of‑Experts processing
    moe_out = moe__root_moe(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=(out_shapes[0],),    # same stream shape as the final result
        out_perms=(None,),
    )

    # 3) Residual addition (binary_add is a DSL op)
    final = binary_add(moe_out, res_add_0)

    return final

# Implementation reasoning:
# The root node simply forwards the raw off‑chip tensors to the three child
# blackboxes – `pre_attention`, `attention`, and `moe`.  Each blackbox expects
# vanilla‑shaped inputs, so no `offchip_load` or other tensor‑method transforms
# are needed before the call.  The only responsibility of this function is to
# specify the desired stream shapes for the children (via `out_shapes`) and to
# write the final result back to memory with `offchip_store`.  We keep the
# sequence length (`seq_len`) from `dims` so the implementation works for any
# static length, and we use stream shapes that respect the required rank ≥ 3:
#   – Q:   (seq_len, 16, 32)
#   – K,V: (seq_len, 4, 32)
#   – attention and MoE outputs: (seq_len, 1, 512)  (one stream dim + a single‑row tile)
def tiled_reference(dims, tensors):
    # Extract raw tensors from the input dictionary.
    input_tensor = tensors["input_tensor"]
    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    o_proj_weight = tensors["o_proj_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    expert_weights = tensors["expert_weights"]
    expert_onehot = tensors["expert_onehot"]

    seq_len = dims["seq_len"]

    # Stage 1: RMSNorm → QKV → per‑head RMSNorm → RoPE
    Q, K, V = pre_attention(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=((seq_len, 16, 32), (seq_len, 4, 32), (seq_len, 4, 32)),
    )

    # Stage 2: GQA attention, O‑projection and first residual add
    res_add_0 = attention(
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
        out_shapes=((seq_len, 1, 512),),
    )

    # Stage 3: Post‑attention RMSNorm, MoE, final residual add
    out = moe(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=((seq_len, 1, 512),),
    )

    # Write the final stream back to off‑chip memory.
    return offchip_store(out)