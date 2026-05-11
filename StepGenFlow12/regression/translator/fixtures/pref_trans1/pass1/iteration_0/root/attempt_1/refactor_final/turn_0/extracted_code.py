# This implementation follows the reference model step‑by‑step using only
# the provided DSL primitives.  All off‑chip tensors are first loaded via
# `offchip_load`/`offchip_load_ref`.  The attention block is delegated to the
# pre‑existing child `attention_block`.  RMSNorm is expressed with
# `unary_square`, `unary_rowwise_sum`, `binary_div`, `unary_add_imm`,
# `unary_rsqrt` and `binary_mul`.  The MoE routing uses `repeat_static` to
# duplicate tokens for the two top‑k positions and the inter‑dimensional
# block factor, `flat_partition`/`flat_reassemble` to route per‑expert data,
# and per‑expert linear layers built from `offchip_load_ref` together with
# `binary_matmul`, `unary_silu`, `binary_mul` and `accum_add`.  Finally the
# residual add and off‑chip store close the kernel.

def tiled_reference(dims, tensors):
    # ----------------------------------------------------------------------
    # 1️⃣  Attention block (child)
    # ----------------------------------------------------------------------
    # Raw inputs are off‑chip; they can be passed directly to the child.
    # The child returns a tiled stream: (seq_len, 1, hidden_dim).
    seq_len = tensors["input_tensor"].shape[0]          # 64
    hidden_dim = tensors["input_tensor"].shape[1]      # 512

    # Define the output tile shape for the child (one stream dim + tile dims)
    attention_out = attention_block(
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

    # ----------------------------------------------------------------------
    # 2️⃣  Post‑attention RMSNorm (x * rsqrt(mean(x²) + eps))
    # ----------------------------------------------------------------------
    eps = 1e-6

    # x²
    x_sq = unary_square(attention_out)

    # sum over the hidden dimension (tile column).  The tile row is 1, so a
    # single `unary_rowwise_sum` is sufficient.
    sum_sq = unary_rowwise_sum(x_sq)                     # (seq_len, 1, 1)

    # mean = sum_sq / hidden_dim
    hidden_const = unary_to_const_int(sum_sq, hidden_dim)
    mean = binary_div(sum_sq, hidden_const)

    # mean + eps
    mean_eps = unary_add_imm(mean, eps)

    # 1 / sqrt(mean + eps)
    inv_std = unary_rsqrt(mean_eps)

    # Normalized tensor (same shape as attention_out)
    normed_2 = binary_mul(attention_out, inv_std)        # (seq_len, 1, hidden_dim)

    # ----------------------------------------------------------------------
    # 3️⃣  MoE block
    # ----------------------------------------------------------------------
    # Constants
    num_experts = tensors["w_gate"].shape[0]             # 8
    inter_dim   = tensors["w_gate"].shape[2]            # 1792

    # Choose a block size that evenly splits the intermediate dimension.
    # Here we use 64 (512 // 8).  This yields 28 blocks (1792 // 64).
    tile_col_gate = hidden_dim // 8                     # 64
    assert inter_dim % tile_col_gate == 0, "inter_dim not divisible by tile_col_gate"
    num_blocks = inter_dim // tile_col_gate              # 28

    # --------------------------------------------------------------
    # Prepare token stream duplicated for the two top‑k positions.
    # --------------------------------------------------------------
    # (seq_len, 2, 1, hidden_dim)
    token_rep_top = repeat_static(normed_2, factor=2)

    # (seq_len, 2, num_blocks, 1, hidden_dim)
    token_rep = repeat_static(token_rep_top, factor=num_blocks)

    # --------------------------------------------------------------
    # Partition tokens and expert weights per expert using the routing mask.
    # --------------------------------------------------------------
    # expert_onehot : (seq_len, 2, num_experts)   (int64)
    # flat_partition will emit, for each expert e,
    #   a tensor of shape (T_e, 1, hidden_dim) where T_e is the number of
    #   (token, top‑pos) pairs routed to that expert.
    tokens_per_expert = flat_partition(token_rep, tensors["expert_onehot"], n=num_experts)

    # Load expert_weights (vanilla shape (seq_len, 2)) as a tiled stream.
    # We use tile size 1×1 and stride (2,1) so each element gets its own tile.
    exp_weights_stream = offchip_load(
        underlying=tensors["expert_weights"],
        stride=(2, 1),                     # (seq_len dim stride, 2‑dim stride)
        out_shape_tiled=(seq_len, 2),
        tile_row=1,
        tile_col=1,
    )  # shape: (1, seq_len, 2, 1, 1)

    # Partition the scalar weights with the same routing mask.
    weights_per_expert = flat_partition(exp_weights_stream, tensors["expert_onehot"], n=num_experts)

    # --------------------------------------------------------------
    # Per‑expert computation.
    # --------------------------------------------------------------
    moe_expert_outputs = []  # list of tensors, each (T_e, 1, hidden_dim)

    for e_idx in range(num_experts):
        # ---- token slice for this expert (T_e, 1, hidden_dim) ----
        tok = tokens_per_expert[e_idx]                  # (T_e, 1, hidden_dim)

        # ---- repeat over the inter‑dim blocks (T_e, num_blocks, 1, hidden_dim) ----
        tok_rep = repeat_static(tok, factor=num_blocks)

        # ---- Load gate and up matrices for this expert, broadcasting over the
        #      (T_e, num_blocks) stream using offchip_load_ref -----------------
        # Reference stream for gate/up
        ref_gate = tok_rep
        stride_gate = tuple(0 for _ in range(len(ref_gate.shape) - 2))  # (0,0,0)

        gate_w = offchip_load_ref(
            ref=ref_gate,
            underlying=tensors["w_gate"][e_idx],
            stride=stride_gate,
            out_shape_tiled=ref_gate.shape[:-2],
            tile_row=hidden_dim,          # 512
            tile_col=tile_col_gate,       # 64
        )
        up_w = offchip_load_ref(
            ref=ref_gate,
            underlying=tensors["w_up"][e_idx],
            stride=stride_gate,
            out_shape_tiled=ref_gate.shape[:-2],
            tile_row=hidden_dim,
            tile_col=tile_col_gate,
        )

        # ---- Linear projections -------------------------------------------------
        gate_out = binary_matmul(tok_rep, gate_w)          # (T_e, num_blocks, 1, tile_col_gate)
        up_out   = binary_matmul(tok_rep, up_w)           # same shape

        # ---- SiLU activation and elementwise multiplication --------------------
        hidden = binary_mul(unary_silu(gate_out), up_out) # (T_e, num_blocks, 1, tile_col_gate)

        # ---- Load down matrix for this expert (broadcast over same stream) -----
        ref_down = hidden
        stride_down = tuple(0 for _ in range(len(ref_down.shape) - 2))  # (0,0,0)

        down_w = offchip_load_ref(
            ref=ref_down,
            underlying=tensors["w_down"][e_idx],
            stride=stride_down,
            out_shape_tiled=ref_down.shape[:-2],
            tile_row=tile_col_gate,      # 64
            tile_col=hidden_dim,         # 512
        )

        # ---- Down projection ----------------------------------------------------
        down_out = binary_matmul(hidden, down_w)          # (T_e, num_blocks, 1, hidden_dim)

        # ---- Weight by expert scalar (repeat weights over the block dimension) --
        w_scalar = weights_per_expert[e_idx]               # (T_e, 1, 1)
        w_scalar_rep = repeat_static(w_scalar, factor=num_blocks)  # (T_e, num_blocks, 1, 1)

        weighted_down = binary_mul(down_out, w_scalar_rep)   # (T_e, num_blocks, 1, hidden_dim)

        # ---- Sum over the inter‑dim block axis ----------------------------------
        contrib = accum_add(weighted_down, rank=1)          # (T_e, 1, hidden_dim)

        moe_expert_outputs.append(contrib)

    # --------------------------------------------------------------
    # Re‑assemble per‑expert contributions back to (seq_len, 2, 1, hidden_dim)
    # --------------------------------------------------------------
    # flat_reassemble returns shape (seq_len, 2, n_active, 1, hidden_dim)
    # where n_active == 1.  We collapse the extra dimensions with two
    # successive `accum_add` (rank=2) which sums over both.
    moe_assembled = flat_reassemble(moe_expert_outputs, tensors["expert_onehot"])
    moe_output = accum_add(moe_assembled, rank=2)   # (seq_len, 1, hidden_dim)

    # ----------------------------------------------------------------------
    # 4️⃣  Final residual addition
    # ----------------------------------------------------------------------
    final = binary_add(moe_output, attention_out)   # (seq_len, 1, hidden_dim)

    # ----------------------------------------------------------------------
    # Off‑chip store (root must end with offchip_store)
    # ----------------------------------------------------------------------
    return offchip_store(final)