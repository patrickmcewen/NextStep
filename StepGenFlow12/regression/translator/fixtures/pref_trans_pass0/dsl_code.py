# pre_attn_norm_and_proj
# ----------------------
# Implements the pre‑attention RMSNorm followed by Q, K, V linear projections.
# All RAW tensors are first streamed onto‑chip with `offchip_load`.  RMSNorm is
# built from unary/binary ops (square → row‑wise sum → mean → ε‑add → rsqrt →
# scale).  Each weight matrix is broadcast across the sequence dimension,
# multiplied with the normalized activations, and then reshaped from
# (seq_len, heads*head_dim) → (seq_len, heads, head_dim).  The extra leading
# singleton stream dimension introduced by `offchip_load` is removed with
# `flatten`, merging the (1, seq_len) stream into a single `seq_len` dim.
# The resulting tensors have the exact shapes declared by the parent:
#   Q → (64, 16, 32)   K → (64,  4, 32)   V → (64,  4, 32)

def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos,
                          *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalar shape constants (pure Python arithmetic is allowed)
    # ------------------------------------------------------------------
    seq_len    = input_tensor.shape[0]          # 64
    hidden_dim = input_tensor.shape[1]          # 512
    head_dim   = cos.shape[-1]                  # 32
    num_heads  = q_proj.shape[1] // head_dim    # 16
    num_kv_heads = k_proj.shape[1] // head_dim  # 4
    eps = 1e-6

    # ------------------------------------------------------------------
    # Load the raw tensors into the on‑chip stream format.
    # offchip_load always adds a leading singleton stream dimension.
    # ------------------------------------------------------------------
    # input_tensor: stream over the sequence, tile = (1, hidden_dim)
    inp = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=hidden_dim,
    )  # shape: stream(1, seq_len) × tile(1, hidden_dim)

    # ------------------------------------------------------------------
    # RMSNorm = x * rsqrt(mean(x²) + eps)
    # ------------------------------------------------------------------
    inp_sq   = unary_square(inp)                         # tile(1, hidden_dim)
    sum_sq   = unary_rowwise_sum(inp_sq)                 # tile(1,1)
    mean_sq  = unary_mul_imm(sum_sq, 1.0 / hidden_dim)   # tile(1,1)
    mean_eps = unary_add_imm(mean_sq, eps)               # tile(1,1)
    scale    = unary_rsqrt(mean_eps)                     # tile(1,1)
    normed   = binary_mul(inp, scale)                    # tile(1, hidden_dim)

    # ------------------------------------------------------------------
    # Helper: project `normed` with a weight matrix and reshape to
    # (seq_len, heads, head_dim).  Returns a tensor of shape
    # stream(1, seq_len) × tile(heads, head_dim).
    # ------------------------------------------------------------------
    def proj_and_reshape(weight, out_dim, heads):
        # Broadcast weight across the sequence dimension.
        w = offchip_load(
            weight,
            stride=(0,),                # same tile for every seq position
            out_shape_tiled=(seq_len,),
            tile_row=hidden_dim,
            tile_col=out_dim,
        )  # shape: stream(1, seq_len) × tile(hidden_dim, out_dim)

        # MatMul: (seq_len, hidden_dim) @ (hidden_dim, out_dim) → (seq_len, out_dim)
        raw = binary_matmul(normed, w)               # stream(1, seq_len) × tile(1, out_dim)

        # Promote to insert a dummy stream dim before the tile‑col.
        raw_promoted = promote(raw, rank=0)          # stream(1, seq_len, 1) × tile(1, out_dim)

        # Split the tile‑col (out_dim = heads * head_dim) into a new stream dim.
        raw_split = retile_streamify(
            raw_promoted,
            chunk=head_dim,
            split_row=False,                         # split column dimension
        )  # shape: stream(1, seq_len, heads) × tile(1, head_dim)

        # Merge the new stream dim (heads) into the tile‑row dimension.
        final = accum_retile_row(raw_split, rank=1)  # stream(1, seq_len) × tile(heads, head_dim)
        return final

    # ------------------------------------------------------------------
    # Compute Q, K, V and remove the leading singleton stream dim.
    # ------------------------------------------------------------------
    Q_raw = proj_and_reshape(q_proj, q_proj.shape[1], num_heads)      # (1, seq_len) × (16,32)
    K_raw = proj_and_reshape(k_proj, k_proj.shape[1], num_kv_heads)   # (1, seq_len) × (4,32)
    V_raw = proj_and_reshape(v_proj, v_proj.shape[1], num_kv_heads)   # (1, seq_len) × (4,32)

    # Flatten the leading singleton with the sequence dimension.
    Q = flatten(Q_raw, min_rank=0, max_rank=1)   # stream(seq_len) × tile(16,32) → (64,16,32)
    K = flatten(K_raw, min_rank=0, max_rank=1)   # (64,4,32)
    V = flatten(V_raw, min_rank=0, max_rank=1)   # (64,4,32)

    return Q, K, V

# per_head_norm_and_rope
# ------------------------------------------------------------
# 1. Load the RAW cosine / sine tensors from off‑chip, flatten the leading
#    singleton dimension that `offchip_load` adds, and keep them as streams.
# 2. Perform per‑head RMSNorm using only DSL primitives.
# 3. Apply RoPE.  The rotation‐half operation is implemented by splitting the
#    head dimension into two halves (using `retile_streamify`), separating the
#    halves into independent streams (`flatten` + `parallelize`), applying the
#    RoPE formulas
#        first_half  = a * cos_a  –  b * sin_a
#        second_half = b * cos_b  +  a * sin_b
#    where *a* and *b* are the two halves of the input, and then merging the
#    halves back (using `static_reassemble`, `reshape_stream`,
#    `accum_retile_col`).  This reproduces the exact behaviour of the
#    reference `_rotate_half` helper without using any Python indexing.
# 4. V is passed through unchanged.
#
# All arithmetic is expressed via DSL calls; only scalar Python operations
# (e.g. computing the half‑size) are used.
def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # -------------------------------------------------
    # Load RAW cosine / sine tensors
    # -------------------------------------------------
    cos_loaded = offchip_load(
        cos, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32
    )
    sin_loaded = offchip_load(
        sin, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32
    )
    # Remove the leading singleton added by offchip_load
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)  # stream(64,)×tile(1,32)
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)  # stream(64,)×tile(1,32)

    # -------------------------------------------------
    # RMS‑Norm for Q
    # -------------------------------------------------
    Q_sq   = unary_square(Q)
    Q_sum  = unary_rowwise_sum(Q_sq)                 # (..., tile_r, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / 32.0)         # divide by head_dim
    Q_eps  = unary_add_imm(Q_mean, 1e-6)
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q_norm = binary_mul(Q, Q_rsqrt)

    # -------------------------------------------------
    # RMS‑Norm for K
    # -------------------------------------------------
    K_sq   = unary_square(K)
    K_sum  = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / 32.0)
    K_eps  = unary_add_imm(K_mean, 1e-6)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # -------------------------------------------------
    # Helper: apply RoPE to a (stream, tile) tensor
    # -------------------------------------------------
    def apply_rope(x, cos_t, sin_t):
        # x, cos_t, sin_t all have shape stream(64,)×tile(R, head_dim)
        head_dim = x.shape[-1]          # Python int
        half = head_dim // 2            # split point

        # ---- Split the head dimension into two halves ----
        # retile column => new stream dim (seq_len*2) and half‑sized tile_c
        x_split = retile_streamify(x, chunk=half, split_row=False)
        cos_split = retile_streamify(cos_t, chunk=half, split_row=False)
        sin_split = retile_streamify(sin_t, chunk=half, split_row=False)

        # reshape stream (seq_len*2) → (seq_len, 2)
        x_resh = reshape_stream(x_split, chunk_size=2, rank=0)
        cos_resh = reshape_stream(cos_split, chunk_size=2, rank=0)
        sin_resh = reshape_stream(sin_split, chunk_size=2, rank=0)

        # flatten the two stream dimensions and parallelize to obtain the two halves
        x_flat = flatten(x_resh, min_rank=0, max_rank=1)      # stream(64*2,)×tile(R,half)
        cos_flat = flatten(cos_resh, min_rank=0, max_rank=1)
        sin_flat = flatten(sin_resh, min_rank=0, max_rank=1)

        x_parts = parallelize(x_flat, n=2)    # [a, b]
        cos_parts = parallelize(cos_flat, n=2)  # [cos_a, cos_b]
        sin_parts = parallelize(sin_flat, n=2)  # [sin_a, sin_b]

        a, b = x_parts
        cos_a, cos_b = cos_parts
        sin_a, sin_b = sin_parts

        # ---- RoPE formulas for the two halves ----
        a_cos = binary_mul(a, cos_a)                         # a * cos_a
        b_cos = binary_mul(b, cos_b)                         # b * cos_b
        b_sin = binary_mul(b, sin_a)                         # b * sin_a
        a_sin = binary_mul(a, sin_b)                         # a * sin_b

        first_half  = binary_add(a_cos, unary_mul_imm(b_sin, -1.0))  # a*cos_a - b*sin_a
        second_half = binary_add(b_cos, a_sin)                       # b*cos_b + a*sin_b

        # ---- Re‑assemble the halves back into the original head dimension ----
        # interleave the two streams (a token’s first half, then its second half)
        interleaved = static_reassemble([first_half, second_half])

        # reshape stream (seq_len*2) → (seq_len, 2) again
        inter_resh = reshape_stream(interleaved, chunk_size=2, rank=0)

        # merge the half‑index stream dimension into the tile column dimension
        out = accum_retile_col(inter_resh, rank=1)

        return out

    # -------------------------------------------------
    # Apply RoPE to Q and K
    # -------------------------------------------------
    Q_out = apply_rope(Q_norm, cos_stream, sin_stream)
    K_out = apply_rope(K_norm, cos_stream, sin_stream)

    # V is unchanged
    V_out = V

    return Q_out, K_out, V_out

# Implementation notes:
# This node simply forwards its inputs to the two child blackboxes that
# implement the full pre‑attention computation.  All inputs are RAW
# (off‑chip), but blackboxes are allowed to receive RAW tensors directly;
# they internally handle any required loading.  No further DSL operations
# are needed here, so we just call the children with the requested
# `out_shapes`/`out_perms` and return the final Q, K, V streams.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # First stage: RMSNorm + Q/K/V projection
    Q, K, V = pre_attn_norm_and_proj(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Second stage: per‑head RMSNorm and RoPE
    Q, K, V = per_head_norm_and_rope(
        Q,
        K,
        V,
        cos,
        sin,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return Q, K, V

# Implementation reasoning:
# Q, K, V are on‑chip streams with shapes:
#   Q : stream(64,) × tile(16, 32)
#   K : stream(64,) × tile(4, 32)
#   V : stream(64,) × tile(4, 32)
#
# The required transformation is a view+permute that swaps the stream
# dimension (seq_len) with the tile‑row dimension (heads).  This can be
# expressed with the available DSL primitives as follows:
#
# 1. Merge the stream dimension into the tile‑row dimension
#    (accum_retile_row).
# 2. Promote the result so we have an explicit stream dim of size 1.
# 3. Split every row into its own tile (retile_streamify with chunk=1);
#    we now have a stream of single‑row tiles.
# 4. Use `parallelize` (round‑robin) to distribute those rows into
#    `num_groups` sub‑streams, where `num_groups` is the number of heads
#    for the tensor (16 for Q, 4 for K/V).  Each sub‑stream now contains
#    all rows belonging to a particular head, ordered by sequence.
# 5. For each sub‑stream, merge its stream dimension back into the tile‑row
#    dimension (accum_retile_row) to obtain a single tile of shape
#    (seq_len, dim).  Add a leading singleton stream dimension with
#    `promote` so that all sub‑streams have the same shape.
# 6. Re‑assemble the per‑head tiles into one stream using
#    `static_reassemble`; because each input has exactly one stream element,
#    this simply concatenates the tiles, yielding a stream of length
#    `num_groups` with tile rows = seq_len.
# 7. Finally split the outer stream dimension into the two required stream
#    axes with `reshape_stream`:
#       – Q : split 16 → (kv_heads=4, query_per_kvhead=4)
#       – K/V : split 4 → (kv_heads=4, 1)
#
# All operations are pure DSL calls; no raw tensor arithmetic is used.

def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # scalar parameters
    num_heads = Q.shape[-2]                 # 16
    kv_heads = K.shape[-2]                  # 4
    query_per_kvhead = num_heads // kv_heads  # 4

    # ------------------------------------------------------------------------
    # Helper: reorder a tensor of shape (seq_len, heads, dim) into a stream of
    #         (num_heads, seq_len, dim) where each head's rows are grouped.
    # ------------------------------------------------------------------------
    def _reorder(tensor, num_groups):
        # 1. merge stream (seq_len) into tile rows (heads)
        merged = accum_retile_row(tensor, rank=1)          # tile(num_groups*seq_len, dim)

        # 2. add a leading stream dim of size 1
        promoted = promote(merged, rank=0)                # stream(1,)×tile(...)

        # 3. split each row into its own tile (chunk=1)
        rows = retile_streamify(promoted, chunk=1)        # stream(N,)×tile(1, dim)

        # 4. round‑robin split the rows into `num_groups` sub‑streams
        substreams = parallelize(rows, n=num_groups)      # list of tensors, each stream(seq_len,)

        # 5. turn each sub‑stream back into a single tile (seq_len rows)
        tiles = []
        for sub in substreams:
            # merge the sub‑stream into tile rows
            tiled = accum_retile_row(sub, rank=1)         # tile(seq_len, dim)
            # add a leading singleton stream dim so that all tiles have identical shape
            tiled = promote(tiled, rank=0)                # stream(1,)×tile(seq_len, dim)
            tiles.append(tiled)

        # 6. concatenate the per‑head tiles into one stream (length = num_groups)
        combined = static_reassemble(tiles)                # stream(num_groups,)×tile(seq_len, dim)
        return combined

    # ------------------------------------------------------------------------
    # Apply the helper to Q, K, V
    # ------------------------------------------------------------------------
    Q_comb = _reorder(Q, num_groups=num_heads)   # stream(16,)×tile(64,32)
    K_comb = _reorder(K, num_groups=kv_heads)    # stream(4,)×tile(64,32)
    V_comb = _reorder(V, num_groups=kv_heads)    # stream(4,)×tile(64,32)

    # ------------------------------------------------------------------------
    # Final reshape of the leading stream dimension to match the contract
    # ------------------------------------------------------------------------
    Qh = reshape_stream(Q_comb, chunk_size=query_per_kvhead, rank=0)   # stream(4,4)×tile(64,32)
    Kh = reshape_stream(K_comb, chunk_size=1, rank=0)                  # stream(4,1)×tile(64,32)
    Vh = reshape_stream(V_comb, chunk_size=1, rank=0)                  # stream(4,1)×tile(64,32)

    # ------------------------------------------------------------------------
    # Verify that we produced exactly what the parent expects
    # ------------------------------------------------------------------------
    assert Qh.shape == out_shapes[0], f"Qh shape mismatch: {Qh.shape} != {out_shapes[0]}"
    assert Kh.shape == out_shapes[1], f"Kh shape mismatch: {Kh.shape} != {out_shapes[1]}"
    assert Vh.shape == out_shapes[2], f"Vh shape mismatch: {Vh.shape} != {out_shapes[2]}"

    return Qh, Kh, Vh

# Implementation notes:
# - Qh, Kh, Vh are already on‑chip streams. We expand Kh and Vh along their
#   trailing singleton stream dimension so they share Qh's (kv_head, query_per_kvhead)
#   stream shape.
# - Stable softmax is implemented by subtracting the per‑row maximum before the
#   exponentiation. The row‑wise max is obtained by turning the tile‑column
#   dimension into a stream dimension (via `retile_streamify` + `reshape_stream`)
#   and then reducing with `accum_max`.
# - After the attention computation we reshape the result into the required
#   vanilla layout (seq_len=64, num_heads=16, head_dim=32) by:
#   1) Splitting each row into its own tile (`retile_streamify`).
#   2) Bufferizing and re‑ordering the stream with `streamify` (using a stride
#      that maps (seq_len, num_heads) onto the underlying tile grid).
#   3) Merging the singleton tile‑row dimension back into the stream with
#      `accum_retile_row`, producing the final shape (64, 16, 32).
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Expand KV tensors to match Qh's stream shape (kv_head, query_per_kvhead)
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # ----- GQA attention (stable softmax) -----
    # scores = Q @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # Compute per‑row max for numerical stability:
    # 1) Turn the tile‑column dimension into a stream dimension.
    scores_col_split = retile_streamify(scores, chunk=1, split_row=False)
    # 2) Reshape the combined stream dimension (query_per_kv * tile_c) into
    #    (query_per_kv, tile_c) so the column index becomes its own stream dim.
    tile_c = scores.shape[-1]            # = 64
    scores_reshaped = reshape_stream(scores_col_split,
                                     chunk_size=tile_c,
                                     rank=0)
    # 3) Reduce across the column‑index stream dimension to obtain the max.
    row_max = accum_max(scores_reshaped, rank=1)

    # Subtract the max (broadcast across tile columns) and exponentiate.
    centered_scores = binary_add(scores,
                                unary_mul_imm(row_max, -1.0))
    e = unary_exp(centered_scores)

    # Numerator = e @ V
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # Denominator = sum over the last tile dimension (softmax normalizer)
    denom = unary_rowwise_sum(e)

    # Attention = num / denom   (broadcasts denom across the tile‑col dim)
    attn = binary_div(num, denom)

    # ------------------------------------------------------------------
    # Reshape to the required vanilla layout (seq_len, num_heads, head_dim)
    # ------------------------------------------------------------------
    # Split each row of the attention matrix into a separate tile.
    attn_split = retile_streamify(attn, chunk=1, split_row=True)  # (4,256,1,32)

    # Bufferize so we can index into the stream with a custom stride.
    buf = bufferize(attn_split, rank=2)

    # Desired output shape (seq_len, num_heads, head_dim)
    target_seq, target_heads, target_dim = out_shapes[0]   # (64, 16, 32)

    # Mapping stride: linear_idx = seq * 1 + head * seq_len
    stride = [1, target_seq]                # [1, 64]
    out_shape_tiled = (target_seq, target_heads)  # (64, 16)

    # Reorder the stream dimensions using the stride mapping.
    reordered = streamify(buf,
                          stride,
                          out_shape_tiled)    # (64,16,1,32)

    # Merge the singleton tile‑row dimension into the stream axis,
    # producing tile shape (16,32) and stream shape (64).
    result = accum_retile_row(reordered)   # (64,16,32)

    return result

# The `attention` node receives on‑chip streams `Q`, `K`, `V`.  
# It first expands those into the three head‑specific streams expected by the model  
# (`Qh`, `Kh`, `Vh`) via the `compute_qkv` blackbox.  The vanilla shapes of those
# outputs are:
#   * Qh : (4, 4, 64, 32)   – 4 “group” heads, 4 heads per group, seq_len=64, head_dim=32  
#   * Kh : (4, 1, 64, 32)   – 4 groups, 1 KV‑head per group, seq_len=64, head_dim=32  
#   * Vh : (4, 1, 64, 32)   – same layout as `Kh`  
# For each output we pass a stream shape that mirrors its vanilla shape; the last two
# dimensions are the tile dimensions, so the shapes are valid tile‑streams (rank ≥ 3).  
# The `out_perms` entries are `None` because no permutation is needed.
#  
# The three streams are then fed to `attention_compute`, which produces the final
# attention output with the shape requested by the caller (`out_shapes`, typically
# `(64, 16, 32)`).  We simply forward the caller‑provided `out_shapes` and `out_perms`
# to the child.  The resulting stream is returned directly – the root node will
# handle the off‑chip store.  

def attention(Q, K, V, *, out_shapes, out_perms=None):
    # Compute Q, K, V projections per head.
    Qh, Kh, Vh = compute_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),  # Qh stream shape
            (4, 1, 64, 32),  # Kh stream shape
            (4, 1, 64, 32),  # Vh stream shape
        ),
        out_perms=(None, None, None),
    )
    # Perform attention on the projected heads.
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn

# Implementation reasoning:
# 1. Compute the attention output using the child `attention`. Its output
#    has tile shape (seq_len, num_heads, head_dim) i.e. (64,16,32).
# 2. Turn the per‑token (num_heads×head_dim) tile into a flat (1,512)
#    tile:
#    a) `retile_streamify(..., chunk=1)` splits the tile rows (num_heads)
#       into a separate stream dimension, yielding shape (seq_len*num_heads, 1, head_dim).
#    b) `reshape_stream(..., chunk_size=num_heads, rank=0)` splits that combined
#       stream back into (seq_len, num_heads) streams, giving shape (seq_len, num_heads, 1, head_dim).
#    c) `accum_retile_col(..., rank=1)` merges the innermost stream dimension
#       (num_heads) into the tile‑column dimension, producing (seq_len, 1, num_heads*head_dim)
#       i.e. (64, 1, 512).
# 3. Load the projection matrix (512×512) from off‑chip and broadcast it over the
#    seq_len stream. `offchip_load` introduces a leading singleton stream dimension,
#    which we remove with `flatten(..., min_rank=0, max_rank=1)`, yielding shape
#    (seq_len, 512, 512).
# 4. Perform the matrix multiplication per token with `binary_matmul`,
#    resulting in (seq_len, 1, 512).
# 5. Load the residual `input_tensor` (shape (seq_len, 512)) as a (1, 512) tile per token.
#    After `offchip_load` we flatten the leading singleton stream dimension to obtain
#    (seq_len, 1, 512) to match the projection output.
# 6. Add the residual via `binary_add` and return the result, which matches the
#    required output shape (64, 1, 512).

def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Attention: returns (seq_len, num_heads, head_dim) tile stream.
    attn = attention(
        Q,
        K,
        V,
        out_shapes=((Q.shape[0], Q.shape[1], Q.shape[2]),),
        out_perms=(None,),
    )

    # 2a. Split tile rows (num_heads) into a stream dimension.
    attn_split = retile_streamify(attn, chunk=1)  # (seq_len*num_heads, 1, head_dim)

    # 2b. Reshape the combined stream back into (seq_len, num_heads) streams.
    num_heads = Q.shape[1]
    attn_reshaped = reshape_stream(attn_split, chunk_size=num_heads, rank=0)  # (seq_len, num_heads, 1, head_dim)

    # 2c. Merge the innermost stream dimension (num_heads) into tile columns.
    attn_flat = accum_retile_col(attn_reshaped, rank=1)  # (seq_len, 1, num_heads*head_dim) → (64,1,512)

    # 3. Load the O‑projection weight and broadcast over the seq_len stream.
    w_stream = offchip_load(
        o_proj_weight,
        stride=(0,),
        out_shape_tiled=(Q.shape[0],),   # same seq_len as attention output
        tile_row=512,
        tile_col=512,
    )
    # Remove the leading singleton stream dimension introduced by offchip_load.
    W = flatten(w_stream, min_rank=0, max_rank=1)  # (seq_len, 512, 512)

    # 4. Matrix multiplication: (seq_len,1,512) @ (seq_len,512,512) → (seq_len,1,512)
    proj = binary_matmul(attn_flat, W)

    # 5. Load the residual input tensor and align its shape.
    inp_stream = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(Q.shape[0],),   # seq_len
        tile_row=1,
        tile_col=512,
    )
    inp = flatten(inp_stream, min_rank=0, max_rank=1)  # (seq_len, 1, 512)

    # 6. Residual addition.
    out = binary_add(proj, inp)

    return out

# RMSNorm = x * rsqrt(mean(x**2) + eps)
# The input `res_add_0` is already a tile‑stream of shape (seq_len, 1, hidden_dim).
# We square the tensor, sum across the feature dimension (the column tile), divide
# by the hidden size to obtain the mean, add epsilon, take the reciprocal square‑root,
# and finally multiply the original tensor by this scaling factor.
def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # eps used in the original PyTorch reference
    eps = 1e-6

    # Hidden dimension = product of the two tile dimensions.
    # For the given tiling this is 1 * 512 = 512.
    hidden_dim = res_add_0.shape[-2] * res_add_0.shape[-1]

    # x²
    sq = unary_square(res_add_0)

    # Sum of squares across the hidden dimension (column tile).
    # After this reduction the tile shape becomes (1, 1).
    sum_sq = unary_rowwise_sum(sq)

    # Mean of squares: divide by hidden_dim (implemented as multiplication by its reciprocal).
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)

    # Add epsilon and take rsqrt.
    rsqrt = unary_rsqrt(unary_add_imm(mean_sq, eps))

    # Scale the original tensor.
    out = binary_mul(res_add_0, rsqrt)

    return out

# MoE dispatch (root node) expressed with STeP DSL primitives.
#   1️⃣ Convert the int64 routing mask into a tile‑stream with `select_gen`.
#   2️⃣ Load the per‑slot scalar expert weights (float) as a 1×1‑tile stream via `offchip_load`.
#   3️⃣ Broadcast the normalized token stream across the two activation slots:
#        – `promote_outer` adds a leading singleton stream dimension.
#        – `promote` inserts a trailing singleton stream dimension.
#        – `expand_ref` expands that trailing 1 into the slot dimension using the
#          shape of the weight‑stream as a reference.
#   4️⃣ Turn the one‑hot routing mask into integer tile addresses with `expert_addr_gen`.
#   5️⃣ Fetch the three expert weight matrices for the selected expert of each token‑slot
#      using `random_offchip_load`.  The loaded tensors contain extra singleton stream
#      dimensions; they are collapsed to the proper stream shape with `flatten`.
#   6️⃣ Perform the MoE forward pass:
#        gate → SiLU → up → elementwise mul → down, using `binary_matmul`,
#        `unary_silu` and `binary_mul`.
#   7️⃣ Multiply the down‑projected results by the scalar expert weights
#      (`binary_mul` broadcasts the 1×1 weight tile across the hidden dimension).
#   8️⃣ Sum the contributions from the two slots with `accum_add(rank=1)`.
#   9️⃣ Collapse the two stream dimensions into a single stream dimension,
#        yielding the required shape (seq_len = 64, tile = 1 × 512).
#   🔟 Write the result off‑chip (side‑effect) and return the stream itself
#        so that its shape matches the contract‑declared `(64, 1, 512)`.
def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down,
                                         expert_weights, expert_onehot,
                                         *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 1. Routing mask: int64 one‑hot → tile stream (slots × experts)
    # ------------------------------------------------------------------
    onehot_stream = select_gen(
        expert_onehot,
        is_multihot=False,
        n=expert_onehot.shape[-1],
    )  # stream(1, seq_len) × tile(2, n_routed_experts)

    # ------------------------------------------------------------------
    # 2. Per‑slot scalar expert weights (float) → 1×1‑tile stream
    # ------------------------------------------------------------------
    w_shape = expert_weights.shape                     # (seq_len, n_activated_experts)
    stride_w = (w_shape[-1], 1)                       # (cols, 1)
    expert_weights_stream = offchip_load(
        expert_weights,
        stride=stride_w,
        out_shape_tiled=w_shape,
        tile_row=1,
        tile_col=1,
    )  # stream(1, seq_len, n_activated) × tile(1,1)

    # ------------------------------------------------------------------
    # 3. Replicate the normalized token stream across the two activation slots
    # ------------------------------------------------------------------
    # a) add a leading singleton stream dimension
    normed_outer = promote_outer(normed_2)            # stream(1, seq_len) × tile(1, hidden)
    # b) add a trailing singleton stream dimension
    normed_with_slot = promote(normed_outer, rank=0)  # stream(1, seq_len, 1) × tile(1, hidden)
    # c) expand the trailing 1 → slot dimension (2) using the weight stream as reference
    normed_rep = expand_ref(
        normed_with_slot,
        expert_weights_stream,
        expand_rank=1,
    )  # stream(1, seq_len, 2) × tile(1, hidden)

    # ------------------------------------------------------------------
    # 4. Convert one‑hot mask to per‑expert tile addresses
    # ------------------------------------------------------------------
    addr = expert_addr_gen(onehot_stream, expert_addr_base=0, num_tile_per_expert=1)

    # ------------------------------------------------------------------
    # 5. Load expert weight tiles for the selected expert of each token‑slot
    #    (collapse the extra singleton dimensions produced by random_offchip_load)
    # ------------------------------------------------------------------
    gate_raw = random_offchip_load(
        w_gate,
        addr,
        tile_row=w_gate.shape[1],
        tile_col=w_gate.shape[2],
    )
    gate_tile = flatten(gate_raw, min_rank=0, max_rank=2)   # stream(1, seq_len, 2) × tile(512,1792)

    up_raw = random_offchip_load(
        w_up,
        addr,
        tile_row=w_up.shape[1],
        tile_col=w_up.shape[2],
    )
    up_tile = flatten(up_raw, min_rank=0, max_rank=2)      # stream(1, seq_len, 2) × tile(512,1792)

    down_raw = random_offchip_load(
        w_down,
        addr,
        tile_row=w_down.shape[1],
        tile_col=w_down.shape[2],
    )
    down_tile = flatten(down_raw, min_rank=0, max_rank=2)  # stream(1, seq_len, 2) × tile(1792,512)

    # ------------------------------------------------------------------
    # 6. MoE forward pass (gate → SiLU → up → elementwise mul → down)
    # ------------------------------------------------------------------
    gate_out = binary_matmul(normed_rep, gate_tile)     # stream(1, seq_len, 2) × tile(1,1792)
    gate_act = unary_silu(gate_out)                     # stream(1, seq_len, 2) × tile(1,1792)
    up_out = binary_matmul(normed_rep, up_tile)         # stream(1, seq_len, 2) × tile(1,1792)
    hidden = binary_mul(gate_act, up_out)               # stream(1, seq_len, 2) × tile(1,1792)
    down_out = binary_matmul(hidden, down_tile)         # stream(1, seq_len, 2) × tile(1,512)

    # ------------------------------------------------------------------
    # 7. Apply scalar expert weights (broadcast across hidden dim)
    # ------------------------------------------------------------------
    weighted = binary_mul(down_out, expert_weights_stream)  # stream(1, seq_len, 2) × tile(1,512)

    # ------------------------------------------------------------------
    # 8. Sum over the two activation slots
    # ------------------------------------------------------------------
    summed = accum_add(weighted, rank=1)   # stream(1, seq_len) × tile(1,512)

    # ------------------------------------------------------------------
    # 9. Collapse the two stream dimensions into a single one
    # ------------------------------------------------------------------
    final = flatten(summed, min_rank=0, max_rank=1)  # stream(seq_len) × tile(1,512)

    # ------------------------------------------------------------------
    # 10. Write the result off‑chip (side‑effect) and return the stream.
    # ------------------------------------------------------------------
    offchip_store(final)      # side‑effect: store the tensor off‑chip
    return final              # shape matches contract: (64, 1, 512)

# Implementation reasoning:
# The node consists of two sequential blackbox calls:
#   1. RMSNorm on the on‑chip stream `res_add_0` producing `normed_2`.
#   2. MoE dispatch that consumes `normed_2` together with the RAW weight tensors.
# No DSL operations are needed between the calls because the blackboxes handle
# all necessary shape transformations (they flatten/reshape the incoming streams
# to vanilla shapes internally). The RAW tensors are passed directly to the MoE
# child without an off‑chip load, which the child stub will perform if required.
# The output shape requested by the parent is `(64, 1, 512)`; we forward this
# shape (and any permutation) to both children so that the final stream has the
# correct tile‑stream layout.
def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # Apply RMSNorm to the residual addition stream.
    normed_2 = rms_norm(
        res_add_0,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Perform the Mixture‑of‑Experts dispatch using the normalized tensor and
    # the (still off‑chip) weight tensors.
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

# The `moe` node consists of two steps:
# 1. Dispatch the input tokens to the mixture‑of‑experts sub‑module.  
#    This logic lives in the blackbox `moe_dispatch`.  We forward all
#    inputs unchanged (RAW tensors may be passed directly to a blackbox)
#    and propagate the parent‑provided `out_shapes` / `out_perms` so the
#    child returns a stream whose shape matches the parent contract
#    (64, 1, 512).
# 2. Apply the final residual connection that the original PyTorch
#    model performed (`moe_output + res_add_0`).  Both tensors are now
#    tile‑streams with identical stream shape, so we can use the DSL
#    binary addition `binary_add`.
def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # MoE routing & expert computation.
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
    # Residual addition (stream‑wise).
    return binary_add(res_add_0, moe_out, compute_bw=1)

# Implementation reasoning:
# - The root node simply orchestrates three blackbox sub‑models: pre_attention,
#   attention_o_proj, and moe.  All raw inputs are passed directly to these
#   blackboxes; they handle any off‑chip loading internally.
# - Each blackbox requires an `out_shapes` specification describing the
#   desired tile‑stream shape of its outputs.  For Q, K, V we keep the natural
#   (seq_len, num_heads, head_dim) layout, which already satisfies the
#   rank‑≥‑3 requirement (stream dim + two tile dims).
# - The attention output and the final MoE output have vanilla shape (64, 512).
#   To satisfy the stream‑rank constraint we expose them as a stream of 64
#   tokens with tile shape (1, 512); i.e. stream shape (64, 1, 512).  This way
#   `offchip_store` will flatten the leading stream dimension into rows,
#   yielding the expected off‑chip shape (64, 512).
# - The final result is written off‑chip via `offchip_store`.
def tiled_reference(dims, tensors):
    # Unpack raw tensors provided by the caller.
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

    # 1️⃣ Pre‑attention: produce Q, K, V streams.
    Q, K, V = pre_attention(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(
            (64, 16, 32),   # Q: stream=64, tile=(16,32)
            (64, 4, 32),    # K: stream=64, tile=(4,32)
            (64, 4, 32),    # V: stream=64, tile=(4,32)
        ),
        out_perms=(None, None, None),
    )

    # 2️⃣ Attention + O‑projection + residual addition.
    res_add_0 = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
        out_shapes=((64, 1, 512),),  # stream=64, tile=(1,512)
        out_perms=(None,),
    )

    # 3️⃣ MoE block.
    out = moe(
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=((64, 1, 512),),  # stream=64, tile=(1,512)
        out_perms=(None,),
    )

    # Write the final result off‑chip.  `offchip_store` will reshape the
    # (64,1,512) stream into the expected vanilla shape (64,512).
    return offchip_store(out)