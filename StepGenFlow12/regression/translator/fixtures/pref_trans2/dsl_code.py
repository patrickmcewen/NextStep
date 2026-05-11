# Implementation notes
# --------------------
# The main source of error was the rotation matrix used for `rotate_half`.
# The matrix must satisfy   x @ R == rotate_half(x) = [-x[..., half:], x[..., :half]].
# The correct constant matrix M has the property:
#   - for output column i < half:   M[i+half, i] = -1
#   - for output column i >= half:  M[i-half, i] =  1
# (All other entries are 0).  With this matrix `binary_matmul`
# (which performs a row‑vector‑by‑matrix multiply) yields the exact
# rotate‑half operation.
#
# Additionally, `cos` and `sin` must be streamed token‑wise, not broadcast,
# so they are loaded with stride=(1,).
#
# All other steps (loading, linear projection, reshaping, per‑head RMSNorm,
# and the final RoPE combination) follow the reference PyTorch logic
# using only DSL calls.

def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Extract dimensions (pure Python)
    # ------------------------------------------------------------------
    seq_len      = normed.shape[1]                     # 64
    head_dim     = cos.shape[-1]                       # 32
    num_heads    = q_proj.shape[1] // head_dim         # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim         # 128 // 32 = 4
    half         = head_dim // 2
    eps          = 1e-6

    # ------------------------------------------------------------------
    # Load RAW tensors (broadcast where appropriate)
    # ------------------------------------------------------------------
    q_weight = offchip_load(
        q_proj,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=512,
        tile_col=512,
        transposed=False,
    )
    k_weight = offchip_load(
        k_proj,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=512,
        tile_col=128,
        transposed=False,
    )
    v_weight = offchip_load(
        v_proj,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=512,
        tile_col=128,
        transposed=False,
    )
    # cos and sin vary per token → stride = 1
    cos_tile = offchip_load(
        cos,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
        transposed=False,
    )
    sin_tile = offchip_load(
        sin,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
        transposed=False,
    )

    # ------------------------------------------------------------------
    # Linear projections
    # ------------------------------------------------------------------
    Q = binary_matmul(normed, q_weight)   # (1, S) × tile (1, 512)
    K = binary_matmul(normed, k_weight)   # (1, S) × tile (1, 128)
    V = binary_matmul(normed, v_weight)   # (1, S) × tile (1, 128)

    # ------------------------------------------------------------------
    # Reshape projections into per‑head tiles
    # ------------------------------------------------------------------
    # Q → (1, S) × tile (num_heads, head_dim)
    Q = retile_streamify(Q, chunk=head_dim, split_row=False)
    Q = reshape_stream(Q, chunk_size=num_heads, rank=0)
    Q = accum_retile_row(Q, rank=1)

    # K → (1, S) × tile (num_kv_heads, head_dim)
    K = retile_streamify(K, chunk=head_dim, split_row=False)
    K = reshape_stream(K, chunk_size=num_kv_heads, rank=0)
    K = accum_retile_row(K, rank=1)

    # V → (1, S) × tile (num_kv_heads, head_dim)
    V = retile_streamify(V, chunk=head_dim, split_row=False)
    V = reshape_stream(V, chunk_size=num_kv_heads, rank=0)
    V = accum_retile_row(V, rank=1)

    # ------------------------------------------------------------------
    # Per‑head RMSNorm for Q and K (V left unchanged)
    # ------------------------------------------------------------------
    # Q RMSNorm
    Q_sq   = binary_mul(Q, Q)
    Q_sum  = unary_rowwise_sum(Q_sq)                     # sum over head_dim → (heads, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / head_dim)       # divide by head_dim
    Q_eps  = unary_add_imm(Q_mean, eps)                 # + epsilon
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q_norm = binary_mul(Q, Q_rsqrt)

    # K RMSNorm
    K_sq   = binary_mul(K, K)
    K_sum  = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / head_dim)
    K_eps  = unary_add_imm(K_mean, eps)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # ------------------------------------------------------------------
    # Rotation matrix for `rotate_half` (constant head_dim × head_dim)
    # ------------------------------------------------------------------
    # Build matrix M such that x @ M == rotate_half(x)
    rot_rows = []
    for row in range(head_dim):
        rot_rows.append([0.0] * head_dim)   # placeholder for each row
    for i in range(head_dim):
        if i < half:
            # output column i takes -x[i+half] → M[i+half, i] = -1
            rot_rows[i + half][i] = -1.0
        else:
            # output column i takes  x[i‑half] → M[i‑half, i] =  1
            rot_rows[i - half][i] =  1.0
    rot_const = torch.tensor(rot_rows, dtype=torch.float32)   # (head_dim, head_dim)

    rot_weight = offchip_load(
        rot_const,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=head_dim,
        tile_col=head_dim,
        transposed=False,
    )

    # ------------------------------------------------------------------
    # Apply rotate_half via matrix multiplication
    # ------------------------------------------------------------------
    Q_rot = binary_matmul(Q_norm, rot_weight)   # (1, S) × tile (num_heads, head_dim)
    K_rot = binary_matmul(K_norm, rot_weight)   # (1, S) × tile (num_kv_heads, head_dim)

    # ------------------------------------------------------------------
    # RoPE: x * cos + rotate_half(x) * sin
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q_norm, cos_tile)
    Q_sin = binary_mul(Q_rot, sin_tile)
    Q_out = binary_add(Q_cos, Q_sin)

    K_cos = binary_mul(K_norm, cos_tile)
    K_sin = binary_mul(K_rot, sin_tile)
    K_out = binary_add(K_cos, K_sin)

    V_out = V  # V unchanged after head‑splitting

    return Q_out, K_out, V_out

# Compute scaled‑dot‑product attention with max‑sub stability using only DSL ops.
#   • Qh, Kh, Vh are already on‑chip streams.
#   • Kh and Vh have a singleton qpkv dimension; we broadcast them to Qh's qpkv dimension.
#   • scores   = Qh @ Khᵀ  (shape: …×tile(64,64))
#   • row_max  = max over the column dimension of scores.
#       – Split the column tile into a stream dimension (retile_streamify + reshape_stream).
#       – Reduce that stream dimension with accum_max.
#   • scores_centered = scores - row_max   (broadcasted subtraction via binary_add + unary_mul_imm).
#   • e = exp(scores_centered)
#   • num    = e @ Vh
#   • denom  = sum over columns of e (unary_rowwise_sum)
#   • attn   = num / denom
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Broadcast Kh and Vh across the qpkv stream dimension of Qh.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # Raw attention scores.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # ---- Row‑wise max for numerical stability ----
    # Move the column tile dimension into a stream dimension.
    scores_retiled = retile_streamify(scores, chunk=1, split_row=False)
    # Separate the original qpkv dimension from the column index.
    scores_reshaped = reshape_stream(scores_retiled, chunk_size=64, rank=0)
    # Reduce over the column stream dimension to obtain max per row.
    row_max = accum_max(scores_reshaped, rank=1)          # shape: stream(4,4)×tile(64,1)

    # Subtract max from scores (broadcasted across the column tile).
    scores_centered = binary_add(scores, unary_mul_imm(row_max, -1.0))

    # Exponential of the stabilized scores.
    e = unary_exp(scores_centered)

    # Numerator of softmax.
    num = binary_matmul(e, Vh_exp)

    # Denominator (sum over the column dimension).
    denom = unary_rowwise_sum(e)

    # Final attention output.
    attn = binary_div(num, denom)

    return attn

def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Transform Q from (batch=1, seq_len)×tile(heads, dim) to
    # (kv_head=4, q_per_kv=4, seq_len, head_dim) → stream(4,4)×tile(64,32)
    Qh = restream(
        Q,
        stride=(128, 32, 512, 1),          # (kv_head, q_per_kv, seq_len, head_dim) strides
        out_shape_tiled=(4, 4, 64, 32),
    )

    # Transform K and V from (batch=1, seq_len)×tile(kv_heads, dim)
    # to (kv_head, 1, seq_len, head_dim) → stream(4,1)×tile(64,32)
    Kh = restream(
        K,
        stride=(32, 0, 128, 1),             # (kv_head, broadcast, seq_len, head_dim) strides
        out_shape_tiled=(4, 1, 64, 32),
    )
    Vh = restream(
        V,
        stride=(32, 0, 128, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Core attention computation (blackbox)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),      # child‑specific output shape
        out_perms=(None,),
    )

    # Inverse transform: (kv, q_per_kv)×tile(seq, dim) → (batch=1, seq_len)×tile(heads, dim)
    out = restream(
        attn,
        stride=(0, 32, 2048, 1),            # (batch, seq_len, heads, head_dim) strides
        out_shape_tiled=(1, 64, 16, 32),
    )

    return out

# Implementation reasoning:
# 1. Load the projection weight from off‑chip memory, broadcasting it over the
#    sequence dimension.  The weight has shape (512, 512).  We tile it with
#    tile_row=32 and tile_col=512, yielding tiles of shape (32, 512).  The
#    underlying grid has 16 row‑tiles, so we stream over (seq_len, num_heads) =
#    (64, 16) with stride (0, 1) to broadcast each head’s tile across all 64
#    sequence positions:
#        weight = offchip_load(o_proj_weight,
#                               stride=(0, 1),
#                               out_shape_tiled=(64, 16),
#                               tile_row=32,
#                               tile_col=512)
#
# 2. The input `attn` has tile shape (16, 32) where the 16 rows encode the
#    `num_heads`.  We need the heads to become a separate stream dimension so
#    that they line up with the streamed weight.  First move the tile‑row
#    dimension into a stream dimension with `retile_streamify(split_row=True,
#    chunk=1)`, turning (1, 64, 16, 32) → (1, 1024, 1, 32).  Then split that
#    combined stream dimension back into (seq_len, num_heads) using
#    `reshape_stream(chunk_size=16, rank=0)`, yielding a stream of shape
#    (1, 64, 16, 1, 32).  Now the stream shape matches that of `weight`.
#
# 3. Perform a batched matrix multiplication: each tile (1 × 32) from `attn`
#    multiplies the corresponding weight tile (32 × 512), producing tiles of
#    shape (1 × 512).  This is done with `binary_matmul`.
#
# 4. The result still has a separate head stream dimension (size 16).  The
#    original flatten operation corresponds to summing over this dimension,
#    which we achieve with `accum_add` (rank = 1) that reduces the innermost
#    stream axis.
#
# 5. The final tensor has shape (1, 64, 1, 512) as required.

def o_proj(attn, o_proj_weight, *, out_shapes, out_perms=None):
    # Step 1: load weight and broadcast over sequence length (64)
    weight = offchip_load(
        o_proj_weight,
        stride=(0, 1),
        out_shape_tiled=(64, 16),
        tile_row=32,
        tile_col=512,
    )
    # Step 2: move the head dimension (tile rows = 16) to a stream axis
    attn_stream = retile_streamify(attn, chunk=1, split_row=True)
    # Split the combined stream (1024) into (seq_len=64, heads=16)
    attn_stream = reshape_stream(attn_stream, chunk_size=16, rank=0)
    # Step 3: batched matrix multiplication (1×32) @ (32×512) → (1×512)
    proj = binary_matmul(attn_stream, weight)
    # Step 4: sum over the head stream dimension to obtain (1, 64, 1, 512)
    out = accum_add(proj, rank=1)
    return out

# Implementation reasoning:
# This node composes the two child blackboxes `attention_core` and `o_proj`.
# The inputs `Q`, `K`, `V` are already on‑chip streams with shape (1,64,16,32).
# We first invoke `attention_core` to obtain the attention tensor, preserving the
# same stream shape as `Q`.  The child stub will handle the conversion from the
# tiled stream to the vanilla shape it expects.
# The resulting attention stream is then passed, together with the raw weight
# matrix `o_proj_weight`, to the `o_proj` blackbox.  The parent requires the final
# output shape (1,64,1,512); we simply forward the `out_shapes` and `out_perms`
# arguments we received so the child can emit exactly that shape (and identity
# permutation).  No off‑chip load is needed for `o_proj_weight` because it can be
# given directly to the child blackbox, which recovers its vanilla shape
# internally.
def attention_o_proj(Q, K, V, o_proj_weight, *, out_shapes, out_perms=None):
    # Compute attention output (stream shape matches Q).
    attn = attention_core(
        Q,
        K,
        V,
        out_shapes=((1, 64, 16, 32),),
        out_perms=(None,),
    )
    # Project attention to hidden dimension (final required shape).
    out = o_proj(
        attn,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return out

# attention_block implements the two-step attention computation by delegating the
# heavy lifting to the provided child blackboxes.
#   1. `qkv_preprocess` converts the normalized activations into query, key and
#      value streams.  The expected tile‑stream shapes are derived from the model
#      dimensions: (batch=1, seq_len=64, num_q_heads=16, head_dim=32) for Q and
#      (1, 64, 4, 32) for K and V.  These shapes satisfy the DSL requirement of
#      rank ≥ 3 (at least one stream dimension plus the two tile dimensions).
#   2. `attention_o_proj` performs the attention weighting and the final output
#      projection, producing a stream whose shape matches the contract‑specified
#      `out_shapes` ((1, 64, 1, 512),).
# No off‑chip loads or tensor arithmetic are performed here; raw tensors are
# passed directly to the child blackboxes, which internally handle any necessary
# loading.
def attention_block(normed, q_proj, k_proj, v_proj, cos, sin, o_proj_weight, *, out_shapes, out_perms=None):
    # Shapes for the Q, K, V streams expected by the qkv_preprocess blackbox.
    q_shape = (1, 64, 16, 32)   # (batch, seq_len, q_heads, head_dim)
    k_shape = (1, 64, 4, 32)    # (batch, seq_len, kv_heads, head_dim)
    v_shape = (1, 64, 4, 32)    # (batch, seq_len, kv_heads, head_dim)

    # Produce Q, K, V streams.
    Q, K, V = qkv_preprocess(
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(q_shape, k_shape, v_shape),
        out_perms=(None, None, None),
    )

    # Final projection; forward the caller‑provided output shape/permutation.
    o_proj_out = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return o_proj_out

# RMSNorm implementation using only DSL operations.
# The formula is:  y = x * rsqrt(mean(x²) + eps)
# Steps:
#   1. Square the input (`unary_square`).
#   2. Sum over the column dimension (`unary_rowwise_sum`), producing a (1)‑element
#      tile per token (shape ...×1×1).
#   3. Divide by the hidden dimension to obtain the mean (multiply by 1/hidden_dim
#      via `unary_mul_imm` – the constant is a Python scalar, not a tensor).
#   4. Add the epsilon constant (`unary_add_imm`).
#   5. Compute reciprocal square‑root (`unary_rsqrt`).
#   6. Multiply the original input with the scaling factor (`binary_mul`);
#      broadcasting expands the (1)‑column tile to the full hidden dimension.
def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    eps = 1e-6

    # 1. x²
    x_sq = unary_square(res_add_0)

    # 2. sum_{c} x²   → shape (..., 1, 1)
    sum_sq = unary_rowwise_sum(x_sq)

    # 3. mean = sum / hidden_dim
    hidden_dim = res_add_0.shape[-1]            # Python int, safe to use
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)

    # 4. mean + eps
    mean_sq_eps = unary_add_imm(mean_sq, eps)

    # 5. rsqrt(mean + eps)
    scale = unary_rsqrt(mean_sq_eps)

    # 6. x * scale   (broadcast across the hidden dimension)
    out = binary_mul(res_add_0, scale)

    return out

# The expert contribution consists of three matmul projections, a SILU activation,
# and a final weight‑by‑routing step.  
# All inputs are already on‑chip streams. The expert weight matrices are
# per‑expert (stream shape (1, …)); they must be replicated across the token
# dimension (size taken from `routing_weights`) so that the binary_matmul
# operands have identical stream shapes.  This replication is done with
# `repeat_static`.  Afterwards we perform:
#   gate_out = normed_selected @ w_gate_e
#   up_out   = normed_selected @ w_up_e
#   hidden   = silu(gate_out) * up_out
#   down_out = hidden @ w_down_e
#   weighted = down_out * routing_weights
# All operations are expressed using the DSL compute primitives.
def expert_contribute(normed_selected, w_gate_e, w_up_e, w_down_e, routing_weights, *, out_shapes, out_perms=None):
    # Token count (the variable stream dimension) – taken from any token‑wise
    # input, e.g. routing_weights.  This is a plain Python int.
    token_cnt = routing_weights.shape[1]

    # Replicate each expert weight matrix across the token dimension so that
    # the stream shapes of the matrices match `normed_selected` (and later
    # `hidden`).  `repeat_static` inserts a new stream dimension before the
    # existing ones and expands it to `token_cnt`.
    w_gate_rep = repeat_static(w_gate_e, token_cnt)
    w_up_rep   = repeat_static(w_up_e,   token_cnt)
    w_down_rep = repeat_static(w_down_e, token_cnt)

    # Linear projections for the selected tokens.
    gate_out = binary_matmul(normed_selected, w_gate_rep)
    up_out   = binary_matmul(normed_selected, w_up_rep)

    # SILU activation on gate output and element‑wise multiply with up output.
    hidden = binary_mul(unary_silu(gate_out), up_out)

    # Down projection back to hidden dimension.
    down_out = binary_matmul(hidden, w_down_rep)

    # Weight by routing scores (broadcasted across the hidden dimension).
    weighted = binary_mul(down_out, routing_weights)

    return weighted

# This implementation follows the MoE aggregation algorithm using the DSL.
# It loads the raw weight tensors, duplicates each token for the two
# activated experts, partitions tokens (and their routing weights) per
# expert, chunks each expert's token stream into fixed‑size blocks of 8
# (the size expected by the `expert_contribute` blackbox), calls the
# blackbox on each block, merges the block results back into a per‑expert
# stream, and finally re‑assembles and sums the contributions to produce
# the required output stream shape (1, 64, 1, 512).
def moe_aggregation(normed_2, w_gate, w_up, w_down,
                    expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 1) Load the three expert weight tensors from off‑chip memory.
    # ------------------------------------------------------------------
    w_gate_stream = offchip_load(
        w_gate, stride=(1,), out_shape_tiled=(8,), tile_row=512, tile_col=1792
    )
    w_up_stream = offchip_load(
        w_up, stride=(1,), out_shape_tiled=(8,), tile_row=512, tile_col=1792
    )
    w_down_stream = offchip_load(
        w_down, stride=(1,), out_shape_tiled=(8,), tile_row=1792, tile_col=512
    )

    # Collapse the leading singleton, yielding pure per‑expert streams.
    w_gate_flat = flatten(w_gate_stream, min_rank=0, max_rank=1)   # (8,512,1792)
    w_up_flat   = flatten(w_up_stream,   min_rank=0, max_rank=1)   # (8,512,1792)
    w_down_flat = flatten(w_down_stream, min_rank=0, max_rank=1)   # (8,1792,512)

    # Split each weight tensor into a list of per‑expert tensors.
    w_gate_list = parallelize(w_gate_flat, n=8)   # each (1,512,1792)
    w_up_list   = parallelize(w_up_flat,   n=8)   # each (1,512,1792)
    w_down_list = parallelize(w_down_flat, n=8)   # each (1,1792,512)

    # ------------------------------------------------------------------
    # 2) Duplicate every token for the two activated experts.
    # ------------------------------------------------------------------
    normed_rep = repeat_static(normed_2, factor=2)                     # (1,64,2,1,512)
    normed_flat = flatten(normed_rep, min_rank=0, max_rank=2)          # (128,1,512)
    token_stream = reshape_stream(normed_flat, chunk_size=2, rank=0)   # (64,2,1,512)

    # ------------------------------------------------------------------
    # 3) Produce the (multihot) routing mask as a DSL stream.
    # ------------------------------------------------------------------
    control_onehot = select_gen(expert_onehot, is_multihot=True, n=8)   # (1,64,2,8)

    # ------------------------------------------------------------------
    # 4) Partition tokens and routing weights per expert according to the mask.
    # ------------------------------------------------------------------
    token_parts = flat_partition(token_stream, control_onehot, n=8)     # list of 8 tensors, (k_i,1,512)
    routing_stream = metadata_gen(expert_weights)                       # (1,64,2,1,1)
    routing_parts = flat_partition(routing_stream, control_onehot, n=8) # list of 8 tensors, (k_i,1,1)

    # ------------------------------------------------------------------
    # 5) For each expert, further chunk its token stream into blocks of 8,
    #    invoke the blackbox on each block, and merge the block results.
    # ------------------------------------------------------------------
    contribs = []
    for i in range(8):
        # Per‑expert token and routing streams.
        tok_i = token_parts[i]          # (k_i,1,512)
        rout_i = routing_parts[i]       # (k_i,1,1)

        # Chunk into groups of 8 (padding if necessary).
        tok_i_chunks = reshape_stream(tok_i, chunk_size=8, rank=0)   # (C,8,1,512)
        rout_i_chunks = reshape_stream(rout_i, chunk_size=8, rank=0) # (C,8,1,1)

        n_chunks = tok_i_chunks.shape[0]

        if n_chunks == 0:
            # No tokens for this expert – contribution is an empty tensor.
            merged_flat = tok_i
        else:
            # Split the chunked tensors into a list (one per chunk).
            tok_chunks = parallelize(tok_i_chunks, n=n_chunks)      # each (1,8,1,512)
            rout_chunks = parallelize(rout_i_chunks, n=n_chunks)    # each (1,8,1,1)

            # Call the blackbox on every chunk.
            chunk_contribs = []
            for j in range(n_chunks):
                c = expert_contribute(
                    tok_chunks[j],
                    w_gate_list[i],
                    w_up_list[i],
                    w_down_list[i],
                    rout_chunks[j],
                    out_shapes=(tok_chunks[j].shape,),
                    out_perms=(None,),
                )
                chunk_contribs.append(c)

            # Merge the per‑chunk contributions back into a single stream.
            merged, _ = eager_merge(chunk_contribs)                 # (C,8,1,512)

            # Collapse the two stream dimensions (C and 8) into one.
            merged_flat = flatten(merged, min_rank=0, max_rank=1)    # (C*8,1,512)

        contribs.append(merged_flat)

    # ------------------------------------------------------------------
    # 6) Re‑assemble per‑expert contributions to the original token order
    #    and sum over the two active‑expert slots.
    # ------------------------------------------------------------------
    reassembled = flat_reassemble(contribs, control_onehot)   # (1,64,2,n_active,1,512)
    final = accum_add(reassembled, rank=2)                    # (1,64,1,512)

    return final

# Implementation reasoning:
# The `moe_block` node only needs to compose two child blackboxes:
#   1. `rms_norm` – applies RMS normalization to the residual stream.
#   2. `moe_aggregation` – mixes the normalized representation using the
#      MoE expert weights.
# The inputs `w_gate`, `w_up`, `w_down`, `expert_weights`, and `expert_onehot`
# are RAW off‑chip tensors, but they are only consumed by the `moe_aggregation`
# blackbox, which accepts vanilla‑shape tensors directly.  Therefore we do **not**
# load them with `offchip_load`; we simply forward them to the child.
# Both children produce a single stream tensor, and the caller already
# tells us the required output stream shape via `out_shapes`.  We propagate
# that shape (and any output permutation) unchanged to each child, then
# return the final MoE result.  No tensor‑method transforms are used, satisfying
# the call‑site rules.
def moe_block(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # RMSNorm on the residual stream.
    normed_2 = rms_norm(
        res_add_0,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # MoE aggregation using the normalized representation and raw expert tensors.
    moe_output = moe_aggregation(
        normed_2,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return moe_output

# Implementation reasoning:
# 1. Load the raw input tensor as a stream of tiles.  We tile each token as a
#    1×feature tile, i.e. tile_row=1, tile_col=feature_dim (512).  The stream
#    dimension is the sequence length (64) and we keep a leading singleton
#    dimension added by `offchip_load`.
# 2. Perform RMSNorm using DSL compute ops only:
#       - square the input (binary_mul)
#       - sum across the feature dimension (unary_rowwise_sum)
#       - compute the mean by multiplying with 1/feature_dim (unary_mul_imm)
#       - add epsilon (unary_add_imm)
#       - take reciprocal sqrt (unary_rsqrt)
#       - multiply the original input by the scaling factor (binary_mul)
# 3. Call the attention blackbox with the normalized stream and raw weight /
#    positional arguments.  We request the output stream shape to match the
#    input stream shape ((1, seq_len, 1, feature_dim)).
# 4. Add the residual connection (binary_add) between the attention output and
#    the original input stream.
# 5. Call the MoE blackbox on the residual, again asking for the same stream
#    shape.
# 6. Add the final residual (binary_add) and write the result off‑chip with
#    `offchip_store`, which automatically reshapes the stream back to the
#    vanilla (seq_len, feature_dim) layout.
def tiled_reference(dims, tensors):
    # ---------------------------
    # 1) Load the model input as a tiled stream.
    #    Each token is a tile of shape (1, feature_dim).
    # ---------------------------
    seq_len = tensors["input_tensor"].shape[0]          # 64
    feature_dim = tensors["input_tensor"].shape[1]     # 512
    tile_row = 1
    tile_col = feature_dim
    # stride = (grid_c,) = (1,) because we tile only across rows.
    input_stream = offchip_load(
        tensors["input_tensor"],
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=tile_row,
        tile_col=tile_col,
    )  # shape (1, seq_len, tile_row, tile_col) -> (1,64,1,512)

    # ---------------------------
    # 2) RMSNorm on the input stream.
    #    normed = x * rsqrt(mean(x**2) + eps)
    # ---------------------------
    # x**2
    sq = binary_mul(input_stream, input_stream)
    # sum across feature dimension (tile_col)
    sum_sq = unary_rowwise_sum(sq)                     # (1, seq_len, tile_row, 1)
    # mean = sum / feature_dim
    mean_sq = unary_mul_imm(sum_sq, 1.0 / feature_dim)
    # add epsilon
    eps = 1e-6
    mean_eps = unary_add_imm(mean_sq, eps)
    # rsqrt
    inv_sqrt = unary_rsqrt(mean_eps)
    # normalized output
    normed = binary_mul(input_stream, inv_sqrt)

    # ---------------------------
    # 3) Attention block.
    # ---------------------------
    att_out = attention_block(
        normed,
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        tensors["o_proj_weight"],
        out_shapes=((1, seq_len, tile_row, tile_col),),  # (1,64,1,512)
    )

    # ---------------------------
    # 4) First residual addition: o_proj_out + input_tensor
    # ---------------------------
    res_add_0 = binary_add(att_out, input_stream)

    # ---------------------------
    # 5) MoE block.
    # ---------------------------
    moe_out = moe_block(
        res_add_0,
        tensors["w_gate"],
        tensors["w_up"],
        tensors["w_down"],
        tensors["expert_weights"],
        tensors["expert_onehot"],
        out_shapes=((1, seq_len, tile_row, tile_col),),  # (1,64,1,512)
    )

    # ---------------------------
    # 6) Final residual addition and write back off-chip.
    # ---------------------------
    final = binary_add(moe_out, res_add_0)

    # Store the result off‑chip (produces a vanilla (seq_len, feature_dim) tensor)
    return offchip_store(final)