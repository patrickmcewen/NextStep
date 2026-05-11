# Implementation notes:
# This node receives the raw input activation and the three projection matrices.
# The activation must be loaded from off‑chip, RMS‑normalized, and then multiplied
# by each projection matrix.  The projection matrices are also raw off‑chip tensors,
# but they may already be buffered on‑chip in some calling contexts, so we first
# check their rank.  If they are still 2‑D (vanilla) we load them with `offchip_load`;
# otherwise we treat them as an existing stream.  In either case we ensure the
# weight stream has at least two trailing stream dimensions of size 1 (required
# by `expand_ref`) and then expand it to match the activation stream shape.
#
# After the matmul, each result has tile shape (1, out_dim).  We split the column
# tile into `head_dim`‑sized chunks, merge the new stream dimension with the tile
# rows via `accum_retile_row`, and finally flatten the leading singleton stream
# dimension together with the sequence dimension to obtain the contract‑required
# shapes (S, num_heads, head_dim) for Q and (S, num_kv_heads, head_dim) for K/V.
#
# No raw PyTorch arithmetic is used – all arithmetic and shape manipulation are
# performed via DSL calls.

def pre_attention_and_qkv(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0. Static dimensions from the vanilla tensors.
    # ------------------------------------------------------------------
    seq_len = input_tensor.shape[0]          # 64 for the given config
    hidden_dim = input_tensor.shape[1]       # 512
    head_dim = out_shapes[0][2]              # 32 (from the required Q shape)

    # ------------------------------------------------------------------
    # 1. Load the input activation and apply RMSNorm.
    # ------------------------------------------------------------------
    inp = offchip_load(
        input_tensor,
        stride=(1, 1),                # 1‑tile streaming over the batch dimension
        out_shape_tiled=(seq_len, 1),
        tile_row=1,
        tile_col=hidden_dim,
    )
    sq = unary_square(inp)                                 # x²
    sum_sq = unary_rowwise_sum(sq)                         # sum over hidden dim → (…,1,1)
    denom = unary_to_const_int(sum_sq, constant=hidden_dim)  # hidden_dim as float tensor
    mean_sq = binary_div(sum_sq, denom)                    # mean(x²)
    mean_eps = unary_add_imm(mean_sq, constant=1e-6)       # + ε
    inv_rms = unary_rsqrt(mean_eps)                        # rsqrt
    normed = binary_mul(inp, inv_rms)                      # scale

    # ------------------------------------------------------------------
    # 2. Helper to turn a projection matrix into a stream that matches `normed`.
    # ------------------------------------------------------------------
    def prepare_weight(weight):
        # a) If still a raw 2‑D tensor, load it from off‑chip.
        if weight.ndim == 2:
            w = offchip_load(
                weight,
                stride=(1, 1),
                out_shape_tiled=(1, 1),
                tile_row=hidden_dim,
                tile_col=weight.shape[1],
            )
        else:
            w = weight   # already a stream

        # b) Ensure at least two stream dimensions (required by expand_ref).
        stream_rank = len(w.shape) - 2
        if stream_rank == 0:
            w = promote_outer(w)          # → stream rank 1
            w = promote(w, rank=1)        # → stream rank 2
        elif stream_rank == 1:
            w = promote(w, rank=1)        # → stream rank 2

        # c) Broadcast over the sequence dimension to match `normed`.
        return expand_ref(w, normed, expand_rank=2)

    q_stream = prepare_weight(q_proj)   # (1, 64, 1) × tile(512, 512)
    k_stream = prepare_weight(k_proj)   # (1, 64, 1) × tile(512, 128)
    v_stream = prepare_weight(v_proj)   # (1, 64, 1) × tile(512, 128)

    # ------------------------------------------------------------------
    # 3. Compute Q, K, V and reshape to (seq_len, heads, head_dim).
    # ------------------------------------------------------------------
    def compute_projection(proj_stream):
        raw = binary_matmul(normed, proj_stream)                     # (…,1,out_dim)
        split = retile_streamify(raw, chunk=head_dim, split_row=False)  # (…,heads,1,head_dim)
        merged = accum_retile_row(split, rank=1)                     # (…,heads,head_dim)
        return flatten(merged, min_rank=0, max_rank=1)               # (seq_len, heads, head_dim)

    Q = compute_projection(q_stream)
    K = compute_projection(k_stream)
    V = compute_projection(v_stream)

    return Q, K, V