# Implementation reasoning:
# Load the RAW input via `offchip_load` with tile size 1×hidden_dim (tile_row=1,
# tile_col=512) and stream over the sequence length dimension. The load emits a
# stream with an extra leading singleton dimension (shape (1, 64, 1, 512)).
# We flatten the two stream dimensions into a single one to obtain the required
# shape (64, 1, 512). The RMS‑norm is expressed entirely with DSL ops:
#   * square the input (`binary_mul`),
#   * sum across the hidden dimension (`unary_rowwise_sum`),
#   * compute the mean by scaling with 1/hidden_dim (`unary_mul_imm`),
#   * add epsilon (`unary_add_imm`),
#   * take the reciprocal square‑root (`unary_rsqrt`),
#   * finally multiply the original tensor by this factor (`binary_mul`).
# All tensor manipulations use DSL primitives, preserving the streaming shape.

def pre_attention_norm(input_tensor, *, out_shapes, out_perms=None):
    # Load the off‑chip tensor as a stream of 1×hidden_dim tiles.
    # stride=1 advances across the sequence dimension.
    x = offchip_load(
        input_tensor,
        stride=[1],
        out_shape_tiled=[64],
        tile_row=1,
        tile_col=512,
    )
    # Merge the leading singleton stream dimension with the sequence stream.
    x = flatten(x, min_rank=0, max_rank=1)   # shape: (64, 1, 512)

    # x²
    x_sq = binary_mul(x, x)

    # Sum over hidden dimension (tile column) → shape (64, 1, 1)
    sum_sq = unary_rowwise_sum(x_sq)

    # Mean = sum / hidden_dim (512)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / 512.0)

    # Add epsilon for numerical stability.
    eps = 1e-6
    mean_eps = unary_add_imm(mean_sq, eps)

    # rsqrt(mean + eps)
    rsqrt_mean = unary_rsqrt(mean_eps)

    # RMS‑norm: x * rsqrt(mean + eps)
    out = binary_mul(x, rsqrt_mean)

    return out

def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Pure‑Python dimension extraction (scalar math – allowed)
    # ----------------------------------------------------------------------
    seq_len    = normed.shape[0]          # 64
    hidden_dim = normed.shape[-1]         # 512
    head_dim   = cos.shape[-1]            # 32

    # Number of heads for Q and for KV (derived from weight shapes)
    num_q_heads  = q_proj.shape[1] // head_dim   # 512//32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim   # 128//32 = 4

    # ----------------------------------------------------------------------
    # Put the on‑chip 'normed' tensor into a 2‑D stream (adds a trailing
    # stream dimension of size 1).  Result: stream shape (seq_len, 1),
    # tile shape (1, hidden_dim).
    # ----------------------------------------------------------------------
    normed_stream = reshape_stream(normed, chunk_size=1, rank=0)   # (S,1,1,H)

    # ----------------------------------------------------------------------
    # Load the three projection matrices from off‑chip and reshape them
    # into streams of shape (seq_len, num_heads) with tile shape
    # (hidden_dim, head_dim).  The stride (0, 1) broadcasts each head’s
    # tile across the entire sequence dimension.
    # ----------------------------------------------------------------------
    # Q‑projection
    q_weight_raw = offchip_load(
        q_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_q_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    q_weight = flatten(q_weight_raw, min_rank=1, max_rank=2)      # (S, Hq, hidden, head_dim)

    # K‑projection
    k_weight_raw = offchip_load(
        k_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    k_weight = flatten(k_weight_raw, min_rank=1, max_rank=2)      # (S, Hkv, hidden, head_dim)

    # V‑projection
    v_weight_raw = offchip_load(
        v_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    v_weight = flatten(v_weight_raw, min_rank=1, max_rank=2)      # (S, Hkv, hidden, head_dim)

    # ----------------------------------------------------------------------
    # Broadcast the normed tensor across the head dimension so that its
    # stream shape matches the weight streams.
    # expand_ref adds the extra head‑wise stream dimension (size 1 → num_*)
    # while preserving the tile layout.
    # ----------------------------------------------------------------------
    q_norm = expand_ref(normed_stream, q_weight, expand_rank=1)   # (S, Hq, 1, hidden)
    k_norm = expand_ref(normed_stream, k_weight, expand_rank=1)  # (S, Hkv, 1, hidden)
    v_norm = expand_ref(normed_stream, v_weight, expand_rank=1)  # (S, Hkv, 1, hidden)

    # ----------------------------------------------------------------------
    # Matrix multiplication (per‑head): (1 × hidden) @ (hidden × head_dim)
    # ----------------------------------------------------------------------
    Q_mat = binary_matmul(q_norm, q_weight)   # (S, Hq, 1, head_dim)
    K_mat = binary_matmul(k_norm, k_weight)   # (S, Hkv, 1, head_dim)
    V_mat = binary_matmul(v_norm, v_weight)   # (S, Hkv, 1, head_dim)

    # ----------------------------------------------------------------------
    # Merge the per‑head stream dimension into the tile‑row dimension,
    # yielding the requested vanilla shapes:
    #   Q: (seq_len, num_q_heads, head_dim)
    #   K, V: (seq_len, num_kv_heads, head_dim)
    # ----------------------------------------------------------------------
    Q = accum_retile_row(Q_mat, rank=1)   # (S,)×tile(num_q_heads, head_dim)
    K = accum_retile_row(K_mat, rank=1)   # (S,)×tile(num_kv_heads, head_dim)
    V = accum_retile_row(V_mat, rank=1)   # (S,)×tile(num_kv_heads, head_dim)

    return Q, K, V

# per_head_norm: apply RMS (root‑mean‑square) normalization per head.
# For each tensor (Q and K) we compute:
#   1. Square the values (unary_square)
#   2. Sum over the last (head_dim) axis (unary_rowwise_sum)
#   3. Divide by the head dimension size → mean of squares (unary_mul_imm with 1/head_dim)
#   4. Add a small epsilon for numerical stability (unary_add_imm)
#   5. Take the reciprocal square‑root (unary_rsqrt) → 1/√(mean + eps)
#   6. Multiply the original tensor by this factor (binary_mul)
# Steps 1‑5 produce a (seq_len, heads, 1) scaling tensor that broadcasts over the
# head‑dimension when multiplied in step 6. The same sequence is performed for Q
# and K independently, preserving the required output shapes (64,16,32) and
# (64,4,32) and leaving the data on‑chip.
def per_head_norm(Q, K, *, out_shapes, out_perms=None):
    # ---- Normalize Q ----
    Q_sq = unary_square(Q)                                 # Q²
    Q_sum = unary_rowwise_sum(Q_sq)                        # Σ Q² over head_dim → (seq, heads, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / Q.shape[-1])       # mean = Σ Q² / head_dim
    Q_mean_eps = unary_add_imm(Q_mean, 1e-6)               # mean + ε
    Q_scale = unary_rsqrt(Q_mean_eps)                      # 1 / √(mean + ε)
    Q_norm = binary_mul(Q, Q_scale)                        # Q * scale (broadcast over head_dim)

    # ---- Normalize K ----
    K_sq = unary_square(K)                                 # K²
    K_sum = unary_rowwise_sum(K_sq)                        # Σ K² over head_dim → (seq, heads, 1)
    K_mean = unary_mul_imm(K_sum, 1.0 / K.shape[-1])       # mean = Σ K² / head_dim
    K_mean_eps = unary_add_imm(K_mean, 1e-6)               # mean + ε
    K_scale = unary_rsqrt(K_mean_eps)                      # 1 / √(mean + ε)
    K_norm = binary_mul(K, K_scale)                        # K * scale (broadcast)

    return Q_norm, K_norm

# The rope operation is implemented using only STeP DSL primitives.
#   * Raw cosine/sine tensors are loaded with `offchip_load` and flattened to a
#     regular stream (shape (seq_len, 1, dim)).
#   * For each input tensor (Q or K) we:
#       1. Move the per‑head dimension (tile rows) into the stream with
#          `retile_streamify(..., split_row=True)`.
#       2. Split the feature dimension into two halves using
#          `retile_streamify(..., split_row=False)`.
#       3. Introduce an explicit “half‑index” stream dimension with
#          `reshape_stream(..., chunk_size=2)`.
#       4. Flatten the two stream axes and use `parallelize` to obtain the
#          two halves as separate streams.
#   * Cosine and sine are broadcast to the same head‑as‑stream shape using
#     `repeat_ref` with a reference built from the current input tensor.
#   * After broadcasting, cosine and sine are split the same way as the
#     input tensor, flattened, and parallelized.
#   * The four products required by RoPE are computed with `binary_mul`,
#     `binary_add` and a negation via `unary_mul_imm`.
#   * The two output halves are interleaved with `static_reassemble`,
#     the half‑index stream is merged back into the column dimension with
#     `reshape_stream` + `accum_retile_col`, and finally the head dimension
#     is restored with another `reshape_stream` + `accum_retile_row`.
#   * The transformed Q and K tensors are returned.

def apply_rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Load and flatten the raw cosine / sine tables.
    # After `flatten` we have shape (seq_len, 1, dim).
    seq_len = cos.shape[0]                 # Python int
    dim = cos.shape[-1]                    # Python int
    half_dim = dim // 2                    # Python int, dim is even

    cos_stream = flatten(
        offchip_load(
            cos,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=1,
            tile_col=dim,
        ),
        min_rank=0,
        max_rank=1,
    )
    sin_stream = flatten(
        offchip_load(
            sin,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=1,
            tile_col=dim,
        ),
        min_rank=0,
        max_rank=1,
    )

    # ----------------------------------------------------------------------
    def _rope_one(x):
        # x: (seq_len, heads, dim)   where tile rows = heads, tile cols = dim
        heads = x.shape[-2]                     # Python int
        # 1) Move heads into the stream dimension.
        x_stream = retile_streamify(x, chunk=1, split_row=True)   # (seq_len*heads, 1, dim)

        # 2) Split the feature dimension into two halves.
        x_split = retile_streamify(x_stream, chunk=half_dim, split_row=False)  # (seq_len*heads*2, 1, half)

        # 3) Introduce an explicit half‑index stream dimension.
        x_split2 = reshape_stream(x_split, chunk_size=2, rank=0)   # (seq_len*heads, 2, 1, half)

        # 4) Flatten the two stream axes and separate the halves.
        x_flat = flatten(x_split2, min_rank=0, max_rank=1)        # (seq_len*heads*2, 1, half)
        x_parts = parallelize(x_flat, 2)                          # [x0, x1], each (seq_len*heads, 1, half)

        # ------------------------------------------------------------------
        # Broadcast cosine and sine to match the head‑as‑stream layout.
        # Build a reference with stream shape (seq_len, heads).
        ref = reshape_stream(
            retile_streamify(x, chunk=1, split_row=True),
            chunk_size=heads,
            rank=0,
        )  # (seq_len, heads, 1, dim)

        # Repeat cos / sin across the head dimension.
        cos_ref = repeat_ref(cos_stream, ref)   # (seq_len, heads, 1, dim)
        sin_ref = repeat_ref(sin_stream, ref)   # (seq_len, heads, 1, dim)

        # Collapse the two stream axes so we can reuse the same split logic as for x.
        cos_flat = flatten(cos_ref, min_rank=0, max_rank=1)   # (seq_len*heads, 1, dim)
        sin_flat = flatten(sin_ref, min_rank=0, max_rank=1)   # (seq_len*heads, 1, dim)

        # Split cosine / sine the same way as the data tensor.
        cos_split = retile_streamify(cos_flat, chunk=half_dim, split_row=False)   # (seq_len*heads*2, 1, half)
        sin_split = retile_streamify(sin_flat, chunk=half_dim, split_row=False)   # (seq_len*heads*2, 1, half)

        # Introduce the half‑index stream dimension.
        cos_split2 = reshape_stream(cos_split, chunk_size=2, rank=0)   # (seq_len*heads, 2, 1, half)
        sin_split2 = reshape_stream(sin_split, chunk_size=2, rank=0)   # (seq_len*heads, 2, 1, half)

        # Flatten and separate the halves.
        cos_flat2 = flatten(cos_split2, min_rank=0, max_rank=1)       # (seq_len*heads*2, 1, half)
        sin_flat2 = flatten(sin_split2, min_rank=0, max_rank=1)       # (seq_len*heads*2, 1, half)
        cos_parts = parallelize(cos_flat2, 2)                         # [c0, c1]
        sin_parts = parallelize(sin_flat2, 2)                         # [s0, s1]

        # ------------------------------------------------------------------
        # Compute RoPE: out0 = Q0 * cos0 - Q1 * sin0
        #               out1 = Q1 * cos1 + Q0 * sin1
        term0 = binary_mul(x_parts[0], cos_parts[0])                # Q0 * cos0
        term1 = binary_mul(x_parts[1], cos_parts[1])                # Q1 * cos1

        cross0 = binary_mul(x_parts[1], sin_parts[0])               # Q1 * sin0
        cross1 = binary_mul(x_parts[0], sin_parts[1])               # Q0 * sin1

        cross0_neg = unary_mul_imm(cross0, -1.0)                     # -Q1 * sin0

        out0 = binary_add(term0, cross0_neg)                        # first half
        out1 = binary_add(term1, cross1)                            # second half

        # ------------------------------------------------------------------
        # Interleave the two halves back together.
        merged_half = static_reassemble([out0, out1])               # (2*seq_len*heads, 1, half)

        # Merge half‑index stream into the column dimension.
        merged_half = reshape_stream(merged_half, chunk_size=2, rank=0)  # (seq_len*heads, 2, 1, half)
        merged = accum_retile_col(merged_half, rank=1)                 # (seq_len*heads, 1, dim)

        # Restore the original head‑as‑stream layout.
        merged = reshape_stream(merged, chunk_size=heads, rank=0)     # (seq_len, heads, 1, dim)
        merged = accum_retile_row(merged, rank=1)                     # (seq_len, heads, dim)

        return merged

    # Apply the rope transformation to both Q and K.
    Q_out = _rope_one(Q)
    K_out = _rope_one(K)

    return Q_out, K_out

# QKV preparation using only DSL ops.
#   Q: (seq_len, heads, dim) → (kv, q_per_kv, seq_len, dim)
#   K/V: (seq_len, kv_heads, dim) → (kv_heads, 1, seq_len, dim)
# The plan:
#   1. Move the original tile‑row dimension (heads) into the stream with
#      `retile_streamify(chunk=1)`.  Tile rows become 1.
#   2. Split the combined stream dimension into separate stream axes:
#        – For Q: (seq_len, heads) → (seq_len, kv, q_per_kv)
#        – For K/V: (seq_len, kv)   → (seq_len, kv)
#   3. Use a Buffered + `streamify` pair to permute the stream axes so that
#      the sequence dimension becomes the innermost stream axis.
#      This is done by treating the stream axes as the buffer grid and
#      providing a stride that reproduces the original linear index.
#   4. Finally merge the innermost stream axis (the sequence) into the tile‑row
#      dimension with `accum_retile_row`.  For K/V we also add a singleton stream
#      dimension after the kv axis with `reshape_stream(chunk_size=1)`.
#
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------
    # Q → (kv, q_per_kv, seq_len, dim)
    # 1. Move heads into the stream.
    q0 = retile_streamify(Q, chunk=1)                       # stream(1024) × tile(1,32)

    # 2. Split the combined stream into (seq_len, heads).
    q1 = reshape_stream(q0, chunk_size=16, rank=0)          # stream(64,16) × tile(1,32)

    # 3. Split heads (16) into (kv, q_per_kv) = (4,4).
    q2 = reshape_stream(q1, chunk_size=4, rank=0)           # stream(64,4,4) × tile(1,32)

    # 4. Bufferize the three stream dimensions so we can reorder them.
    q_buf = bufferize(q2, rank=3)                           # buffer grid = (64,4,4)

    # 5. Reorder to (kv, q_per_kv, seq_len) via streamify.
    #    Linear index = seq·16 + kv·4 + q  (original layout)
    #    For out shape (kv=4, q=4, seq=64) we need stride = [4, 1, 16].
    q3 = streamify(q_buf, stride=[4, 1, 16], out_shape_tiled=(4, 4, 64))
                                                             # stream(4,4,64) × tile(1,32)

    # 6. Merge the innermost stream (seq_len) into the tile‑row dimension.
    Qh = accum_retile_row(q3, rank=1)                       # stream(4,4) × tile(64,32)

    # ------------------------------
    # K → (kv, 1, seq_len, dim)
    # 1. Move kv heads into the stream.
    k0 = retile_streamify(K, chunk=1)                       # stream(256) × tile(1,32)

    # 2. Split into (seq_len, kv) = (64,4).
    k1 = reshape_stream(k0, chunk_size=4, rank=0)           # stream(64,4) × tile(1,32)

    # 3. Bufferize both stream axes.
    k_buf = bufferize(k1, rank=2)                           # buffer grid = (64,4)

    # 4. Reorder to (kv, seq_len).  Original linear index = seq·4 + kv.
    #    For out shape (kv=4, seq=64) stride = [1, 4].
    k2 = streamify(k_buf, stride=[1, 4], out_shape_tiled=(4, 64))
                                                             # stream(4,64) × tile(1,32)

    # 5. Merge seq_len into tile rows.
    k3 = accum_retile_row(k2, rank=1)                       # stream(4) × tile(64,32)

    # 6. Add a singleton stream dimension after kv.
    Kh = reshape_stream(k3, chunk_size=1, rank=0)           # stream(4,1) × tile(64,32)

    # ------------------------------
    # V → (kv, 1, seq_len, dim)  (identical to K)
    v0 = retile_streamify(V, chunk=1)                       # stream(256) × tile(1,32)
    v1 = reshape_stream(v0, chunk_size=4, rank=0)           # stream(64,4) × tile(1,32)
    v_buf = bufferize(v1, rank=2)                           # buffer grid = (64,4)
    v2 = streamify(v_buf, stride=[1, 4], out_shape_tiled=(4, 64))
                                                             # stream(4,64) × tile(1,32)
    v3 = accum_retile_row(v2, rank=1)                       # stream(4) × tile(64,32)
    Vh = reshape_stream(v3, chunk_size=1, rank=0)           # stream(4,1) × tile(64,32)

    return Qh, Kh, Vh

# attention_weights:
#   1. Broadcast Kh over the query‑per‑kv‑head dimension (expand_ref).
#   2. Compute raw attention scores with a transposed matmul.
#   3. Obtain the per‑row maximum (stable softmax) by
#        a) moving the column dimension (tile_c) into the stream via retile_streamify,
#        b) separating the combined stream dimension back into (query‑head, column) using reshape_stream,
#        c) reducing over the column stream dimension with accum_max.
#   4. Subtract the row‑wise max from the scores (implemented as addition with a negated max).
#   5. Apply exp to obtain the softmax numerator.
#   6. Sum over the column dimension to get the denominator.
#   7. Divide numerator by denominator → final attention weights.
# The resulting tensor has shape (4, 4, 64, 64), matching the required output.
def attention_weights(Qh, Kh, *, out_shapes, out_perms=None):
    # 1. Broadcast Kh to match Qh's query‑per‑kv‑head dimension.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)                # (4, 4, 64, 32)

    # 2. Compute Q · Kᵀ.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4, 4, 64, 64)

    # 3. Compute row‑wise max for numerical stability.
    #    a) Move tile columns into the stream (split_col) – chunk=1 leaves tile_c=1.
    scores_split = retile_streamify(scores, chunk=1, split_row=False)   # (4, 256, 64, 1)
    #    b) Split the combined stream dimension (256 = 4 * 64) back into
    #       (query‑head=4, column=64).
    #    The column size equals the original tile_c dimension.
    col_dim = scores.shape[-1]                               # 64
    scores_reshaped = reshape_stream(scores_split, chunk_size=col_dim, rank=0)  # (4, 4, 64, 64, 1)
    #    c) Reduce over the column stream dimension to get the max per row.
    row_max = accum_max(scores_reshaped, rank=1)             # (4, 4, 64, 1)

    # 4. Subtract the max from the scores (a + (‑max)).
    neg_max = unary_mul_imm(row_max, -1.0)                    # (4, 4, 64, 1)
    shifted = binary_add(scores, neg_max)                    # (4, 4, 64, 64)

    # 5. Exponential for the softmax numerator.
    e = unary_exp(shifted)                                    # (4, 4, 64, 64)

    # 6. Row‑wise sum to obtain the denominator.
    denom = unary_rowwise_sum(e)                              # (4, 4, 64, 1)

    # 7. Softmax: divide numerator by denominator (broadcast across columns).
    attn_weights = binary_div(e, denom)                       # (4, 4, 64, 64)

    return attn_weights

# This node receives on‑chip tensors:
#   attn_weights: shape (4, 4, 64, 64) → stream (4,4) tile (64,64)
#   Vh:          shape (4, 1, 64, 32) → stream (4,1) tile (64,32)
# The reference computes a batched matmul followed by a reshape that swaps the
# two head‑related stream dimensions with the sequence‑length tile dimension,
# yielding a vanilla shape (seq_len, num_heads, head_dim) = (64, 16, 32).
# We reproduce this using only DSL primitives:
#   1. Broadcast `Vh` across the second KV‑head stream dimension (`expand_ref`).
#   2. Batched matmul (`binary_matmul`) → stream (4,4) tile (64,32).
#   3. Allocate a zero‑filled destination tensor of shape (seq_len,)×tile (num_heads, head_dim)
#      by flatten‑ing, retile‑ing, and promoting a zero tensor.
#   4. Collapse the two head‑related stream dimensions into one (`flatten`) and split
#      this merged stream into per‑head sub‑streams (`parallelize`).
#   5. For each head, turn its tile rows into a stream (`retile_streamify`) and
#      write those rows into the destination at the appropriate row offset using
#      `binary_set_offset` + `binary_row_wise_append`.
#   6. The final tensor has stream shape (64,) and tile shape (16,32) – exactly the
#      required output.
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Vh across the second KV‑head dimension.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)

    # 2. Weighted sum over values (batched matmul).
    weighted = binary_matmul(attn_weights, Vh_exp)          # stream (4,4) tile (64,32)

    # -----------------------------------------------------------------
    # 3. Allocate a zero‑filled tensor of shape (seq_len,) × tile (num_heads, head_dim)
    #    (i.e. (64,)×(16,32)).
    # -----------------------------------------------------------------
    # Create a tensor of all ones with the same shape as `weighted` and subtract 1
    # to obtain zeros (avoids disallowed torch.zeros).
    ones   = binary_is_equal(weighted, weighted)            # all 1.0, shape (4,4,64,32)
    zeros  = unary_sub_imm(ones, 1.0)                       # all 0.0, same shape
    # Merge the two head‑related stream dimensions → stream (16,) tile (64,32)
    zeros_flat = flatten(zeros, min_rank=0, max_rank=1)
    # Fold that stream dimension into the tile‑row dimension → tile (1024,32)
    zeros_tile = accum_retile_row(zeros_flat, rank=1)
    # Add a leading singleton stream dimension so we can split the tile‑row
    zeros_tile = promote(zeros_tile, rank=0)                # stream (1,)×tile (1024,32)

    # Number of heads = Hkv * Q_per_KV.
    num_heads = attn_weights.shape[0] * attn_weights.shape[1]   # 4 * 4 = 16
    # Split the combined tile‑row (seq_len * num_heads) back into a stream
    # (seq_len) and a tile‑row (num_heads).
    result = retile_streamify(zeros_tile,
                              chunk=num_heads,
                              split_row=True)           # stream (64,)×tile (16,32)

    # -----------------------------------------------------------------
    # 4. Split the weighted result into per‑head streams.
    # -----------------------------------------------------------------
    merged = flatten(weighted, min_rank=0, max_rank=1)      # stream (16,) tile (64,32)
    head_streams = parallelize(merged, num_heads)          # list of 16 tensors,
                                                             # each shape (1,)×tile (64,32)

    # -----------------------------------------------------------------
    # 5. Write each head’s rows into the destination at the correct offset.
    # -----------------------------------------------------------------
    for h_idx, head in enumerate(head_streams):
        # a) Convert the head’s tile rows into a stream: shape (64,)×tile (1,32)
        head_rows = retile_streamify(head,
                                     chunk=1,
                                     split_row=True)

        # b) Build a constant‑offset tensor of shape (seq_len,)×tile (1,1)
        #    whose value equals the current head index.
        offset_scalar = torch.tensor(float(h_idx), dtype=torch.float32)
        offset_tile   = metadata_gen(offset_scalar)          # stream (1,)×tile (1,1)
        offsets = expand_ref(offset_tile,
                             result,
                             expand_rank=1)               # stream (64,)×tile (1,1)

        # c) Write the rows into the destination at the computed offsets.
        result = binary_row_wise_append(
                    binary_set_offset(result, offsets),
                    head_rows)

    # `result` now has stream shape (64,) and tile shape (16,32),
    # i.e. the vanilla shape (64, 16, 32) required by the contract.
    return result

# The attention node simply wires together the three child blackboxes that
# implement the full attention computation.
#   1. `prepare_qkv` splits the on‑chip Q, K, V streams into per‑expert
#      sub‑streams (Qh, Kh, Vh).  Its outputs have the fixed tile‑stream
#      shapes (4,4,64,32), (4,1,64,32) and (4,1,64,32) respectively.
#   2. `attention_weights` computes the attention matrix from Qh and Kh,
#      yielding a stream of shape (4,4,64,64).
#   3. `apply_weights_and_reshape` applies the attention matrix to Vh and
#      reshapes the result back to the original layout.  The desired output
#      shape for this node is supplied via `out_shapes` (and optionally
#      `out_perms`), so we forward those directly.
# No tensor‑method transforms are performed between the calls; all shape
# handling is delegated to the child stubs via their `out_shapes` arguments.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # 1. Split Q, K, V into per‑expert streams.
    Qh, Kh, Vh = prepare_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),   # Qh
            (4, 1, 64, 32),   # Kh
            (4, 1, 64, 32),   # Vh
        ),
    )

    # 2. Compute attention weights.
    attn_weights = attention_weights(
        Qh,
        Kh,
        out_shapes=((4, 4, 64, 64),),
    )

    # 3. Apply weights to V and reshape to the requested output shape.
    attn = apply_weights_and_reshape(
        attn_weights,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn

# Implementation reasoning:
# * `attn` is already on‑chip with shape (64, 16, 32) → stream dim=64, tile (16, 32).
#   We need to flatten the tile into (1, 32) so it can be multiplied with the weight.
#   We split the tile rows into a new stream dimension using `retile_streamify`
#   (chunk=1 makes each row a separate stream element) and then reshape that
#   stream back into (64, 16) using `reshape_stream`.
# * `o_proj_weight` is raw (off‑chip)  (512, 512).  We tile it as (32, 512) and stream
#   it over (seq_len=64, num_heads=16).  Stride [0, 1] replicates the same head slice
#   across all sequence positions.  `offchip_load` yields a leading singleton stream
#   dimension; we merge it with the sequence dimension via `flatten`.
# * Perform per‑head matmul with `binary_matmul`; result shape (64, 16, 1, 512).
# * Sum over the head stream dimension with `accum_add(rank=1)` → (64, 1, 512).
# * Load the residual `input_tensor` (64, 512) as a tiled stream (1, 512) over the
#   sequence dimension, then flatten the leading singleton → (64, 1, 512).
# * Finally, add the residual via `binary_add` and return the stream.
def o_proj_residual(attn, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Convert attn (64,16,32) → (64,16,1,32)
    attn_split = retile_streamify(attn, chunk=1, split_row=True)          # (1024, 1, 32)
    attn_reshaped = reshape_stream(attn_split, chunk_size=16, rank=0)    # (64, 16, 1, 32)

    # 2. Load and tile the projection weight (512,512) → (64,16,32,512)
    #    stride [0,1]: repeat same head slice across sequence positions.
    weight_loaded = offchip_load(
        o_proj_weight,
        stride=[0, 1],
        out_shape_tiled=(64, 16),
        tile_row=32,
        tile_col=512,
    )                                                                    # (1, 64, 16, 32, 512)
    weight_tiled = flatten(weight_loaded, min_rank=1, max_rank=2)       # (64, 16, 32, 512)

    # 3. Per‑head matrix multiplication
    proj = binary_matmul(attn_reshaped, weight_tiled)                    # (64, 16, 1, 512)

    # 4. Sum over heads → (64, 1, 512)
    proj_sum = accum_add(proj, rank=1)

    # 5. Load residual input_tensor (64,512) → (64, 1, 512)
    input_loaded = offchip_load(
        input_tensor,
        stride=[1],
        out_shape_tiled=(64,),
        tile_row=1,
        tile_col=512,
    )                                                                    # (1, 64, 1, 512)
    input_reshaped = flatten(input_loaded, min_rank=0, max_rank=1)       # (64, 1, 512)

    # 6. Add residual
    output = binary_add(proj_sum, input_reshaped)                        # (64, 1, 512)
    return output

# The attention block is built by chaining the provided blackboxes.
# Each blackbox is given the exact stream shape we need for its output,
# expressed as a tuple of integers (stream_dim, tile_rows, tile_cols).
# All raw off‑chip tensors are passed directly to the children – the
# children internally perform the off‑chip load and reshape to the
# vanilla shapes they expect. No explicit tensor arithmetic or PyTorch
# methods are used; only tuple construction and blackbox calls appear.
def attention_block(input_tensor, q_proj, k_proj, v_proj, cos, sin,
                    o_proj_weight, *, out_shapes, out_perms=None):
    # 1️⃣ Pre‑attention RMSNorm: produce a stream (64, 1, 512) so the
    #    subsequent projection sees the expected vanilla shape (64, 512).
    normed = pre_attention_norm(
        input_tensor,
        out_shapes=((64, 1, 512),),
        out_perms=(None,),
    )

    # 2️⃣ QKV projections: each head dimension is emitted as a separate stream.
    Q, K, V = qkv_projection(
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=(
            (64, 16, 32),   # Q
            (64, 4, 32),    # K
            (64, 4, 32),    # V
        ),
        out_perms=(None, None, None),
    )

    # 3️⃣ Per‑head RMSNorm
    Q, K = per_head_norm(
        Q,
        K,
        out_shapes=((64, 16, 32), (64, 4, 32)),
        out_perms=(None, None),
    )

    # 4️⃣ Apply RoPE
    Q, K = apply_rope(
        Q,
        K,
        cos,
        sin,
        out_shapes=((64, 16, 32), (64, 4, 32)),
        out_perms=(None, None),
    )

    # 5️⃣ GQA attention
    attn = attention(
        Q,
        K,
        V,
        out_shapes=((64, 16, 32),),
        out_perms=(None,),
    )

    # 6️⃣ O‑projection + residual addition.
    #    The required output shape for this node is (64, 1, 512);
    #    we forward the caller‑provided shape/permutation.
    res = o_proj_residual(
        attn,
        o_proj_weight,
        input_tensor,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return res

# Implementation respects all DSL constraints.
# Off‑chip tensors are loaded before any DSL consumer sees them.
# `expert_onehot` is turned into a stream via `select_gen`.
# Per‑expert weight matrices are loaded once with `offchip_load`,
# flattened to collapse the leading singleton stream dimensions,
# and then broadcast across the token stream with `expand_ref`.
# The MoE routing uses `flat_partition`/`flat_reassemble`,
# and the final residual addition is performed with `binary_add`.
# The root ends with `offchip_store`.

def tiled_reference(dims, tensors):
    # ------------------------------------------------------------------
    # 1️⃣  Attention block (child)
    # ------------------------------------------------------------------
    seq_len = tensors["input_tensor"].shape[0]          # 64
    hidden_dim = tensors["input_tensor"].shape[1]      # 512

    # Child returns a tiled stream: (seq_len, 1, hidden_dim)
    att_out = attention_block(
        tensors["input_tensor"],
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        tensors["o_proj_weight"],
        out_shapes=((seq_len, 1, hidden_dim),),
        out_perms=(None,),
    )  # shape: (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 2️⃣  Post‑attention RMSNorm (x * rsqrt(mean(x²) + eps))
    # ------------------------------------------------------------------
    eps = 1e-6

    x_sq = unary_square(att_out)                                 # (seq_len, 1, hidden_dim)
    sum_sq = unary_rowwise_sum(x_sq)                             # (seq_len, 1, 1)
    hidden_const = unary_to_const_int(sum_sq, hidden_dim)        # (seq_len, 1, 1)
    mean = binary_div(sum_sq, hidden_const)                      # (seq_len, 1, 1)
    mean_eps = unary_add_imm(mean, eps)                          # (seq_len, 1, 1)
    inv_std = unary_rsqrt(mean_eps)                              # (seq_len, 1, 1)
    normed = binary_mul(att_out, inv_std)                        # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 3️⃣  Mixture‑of‑Experts (top‑2 routing)
    # ------------------------------------------------------------------
    num_experts = tensors["w_gate"].shape[0]          # 8
    inter_dim = tensors["w_gate"].shape[2]           # 1792

    # Duplicate tokens for the two top‑k positions → (seq_len, 2, 1, hidden_dim)
    token_top = repeat_static(normed, factor=2)

    # Convert the one‑hot routing mask into a tiled stream
    expert_onehot_stream = select_gen(
        tensors["expert_onehot"],
        is_multihot=False,
        n=num_experts,
    )  # shape: (1, seq_len, 2, num_experts)

    # Partition tokens per expert using the routing mask
    tokens_per_expert = flat_partition(
        token_top,
        expert_onehot_stream,
        n=num_experts,
    )  # list of length num_experts, each (T_e, 1, hidden_dim)

    # Load scalar expert weights (shape (seq_len, 2)) as a tiled stream
    exp_weights_stream = offchip_load(
        underlying=tensors["expert_weights"],
        stride=(2, 1),                     # walk the (seq_len, 2) grid row‑major
        out_shape_tiled=(seq_len, 2),
        tile_row=1,
        tile_col=1,
    )  # shape: (1, seq_len, 2, 1, 1)

    # Partition the scalar weights with the same routing mask
    weights_per_expert = flat_partition(
        exp_weights_stream,
        expert_onehot_stream,
        n=num_experts,
    )  # each (T_e, 1, 1)

    # ------------------------------------------------------------------
    # Per‑expert computation
    # ------------------------------------------------------------------
    moe_expert_outputs = []  # will hold (T_e, 1, hidden_dim) tensors

    for e_idx in range(num_experts):
        # Tokens routed to this expert
        tok = tokens_per_expert[e_idx]            # (T_e, 1, hidden_dim)

        # --------------------------------------------------------------
        # Gate and up projection (load once, broadcast across tokens)
        # --------------------------------------------------------------
        gate_tile = offchip_load(
            underlying=tensors["w_gate"][e_idx],
            stride=(0,),
            out_shape_tiled=(1,),
            tile_row=hidden_dim,
            tile_col=inter_dim,
        )  # (1, 1, hidden_dim, inter_dim)
        gate_tile = flatten(gate_tile, min_rank=0, max_rank=1)    # (1, hidden_dim, inter_dim)
        gate_w = expand_ref(gate_tile, tok, expand_rank=1)      # (T_e, hidden_dim, inter_dim)

        up_tile = offchip_load(
            underlying=tensors["w_up"][e_idx],
            stride=(0,),
            out_shape_tiled=(1,),
            tile_row=hidden_dim,
            tile_col=inter_dim,
        )  # (1, 1, hidden_dim, inter_dim)
        up_tile = flatten(up_tile, min_rank=0, max_rank=1)        # (1, hidden_dim, inter_dim)
        up_w = expand_ref(up_tile, tok, expand_rank=1)          # (T_e, hidden_dim, inter_dim)

        gate_out = binary_matmul(tok, gate_w)      # (T_e, 1, inter_dim)
        up_out   = binary_matmul(tok, up_w)       # (T_e, 1, inter_dim)

        hidden = binary_mul(unary_silu(gate_out), up_out)   # (T_e, 1, inter_dim)

        # --------------------------------------------------------------
        # Down projection (load once, broadcast across tokens)
        # --------------------------------------------------------------
        down_tile = offchip_load(
            underlying=tensors["w_down"][e_idx],
            stride=(0,),
            out_shape_tiled=(1,),
            tile_row=inter_dim,
            tile_col=hidden_dim,
        )  # (1, 1, inter_dim, hidden_dim)
        down_tile = flatten(down_tile, min_rank=0, max_rank=1)  # (1, inter_dim, hidden_dim)
        down_w = expand_ref(down_tile, hidden, expand_rank=1)   # (T_e, inter_dim, hidden_dim)

        down_out = binary_matmul(hidden, down_w)   # (T_e, 1, hidden_dim)

        # --------------------------------------------------------------
        # Weight by the per‑token scalar (expert_weights)
        # --------------------------------------------------------------
        w_scalar = weights_per_expert[e_idx]       # (T_e, 1, 1)
        weighted_down = binary_mul(down_out, w_scalar)   # (T_e, 1, hidden_dim)

        moe_expert_outputs.append(weighted_down)

    # ------------------------------------------------------------------
    # Re‑assemble per‑expert contributions back to (seq_len, 2, 1, hidden_dim)
    # ------------------------------------------------------------------
    moe_assembled = flat_reassemble(moe_expert_outputs, expert_onehot_stream)
    # Sum over the top‑k dimension (size 2) and the leading singleton added by flat_reassemble
    moe_summed = accum_add(moe_assembled, rank=2)   # (1, seq_len)×tile(1, hidden_dim)
    # Collapse the leading singleton stream dimension
    moe_output = flatten(moe_summed, min_rank=0, max_rank=1)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # 4️⃣  Final residual addition
    # ------------------------------------------------------------------
    final = binary_add(moe_output, att_out)   # (seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # Off‑chip store (root must end with offchip_store)
    # ------------------------------------------------------------------
    return offchip_store(final)