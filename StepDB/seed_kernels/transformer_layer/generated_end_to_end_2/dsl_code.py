# Implementation reasoning:
# The numerical error comes from apply_rope: parallelize(x_split, 2) on stream(64, 2*n_heads)
# splits by the OUTERMOST "64" batch dim, not the innermost "2*n_heads" half-tile dim.
# Fix: flatten x_split to stream(64*2*n_heads,), parallelize by 2, then reassemble properly.
# 
# For Q (n_heads=16): x_split = stream(64, 32) tile(1,16).
# flatten → stream(2048,). parallelize(2048, 2): consumer 0 gets evens=all first halves, 
# consumer 1 gets odds=all second halves. ✓
# After RoPE ops: static_reassemble([out1,out2]) → stream(2048,) interleaved.
# reshape_stream(chunk=2, rank=0) → stream(1024, 2). accum_retile_col(rank=1) → stream(1024,).
# reshape to stream(64, 16) via reshape_stream(chunk=16, rank=0)... 
# but 1024 = 64*16 so reshape_stream(chunk_size=16, rank=0) splits (1024) into (64, 16). ✓
#
# Wait: parallelize on the flat stream means:
# Consumer 0: positions 0, 2, 4, ... (even) = all first-half tiles of every (batch, head) pair.
# Consumer 1: positions 1, 3, 5, ... (odd) = all second-half tiles of every (batch, head) pair.
# For tile at flat pos 2*(b*n_heads+h) = first half of (b,h), and 2*(b*n_heads+h)+1 = second half. ✓
#
# After computing out1, out2 at stream(B*n_heads,):
# static_reassemble([out1, out2]) → stream(2*B*n_heads,) interleaved: [b0h0_half1, b0h0_half2, ...]
# reshape_stream(chunk=2, rank=0) → stream(B*n_heads, 2) tile(1,16).
# accum_retile_col(rank=1) → stream(B*n_heads,) tile(1,32).
# reshape_stream(chunk_size=n_heads, rank=0) → stream(B, n_heads) tile(1,32). ✓
#
# Also fix: the same issue occurs in scores extraction via retile_streamify+parallelize.
# Let me verify: scores_4x4 is stream(1, seq_b) tile(4,4).
# retile_streamify(chunk=1, split_row=False) → stream(1, seq_b*4) tile(4,1).
# flatten(min_rank=0, max_rank=1) → stream(seq_b*4,) dyn.
# parallelize(scores_flat, 4)[kv_h] → takes positions kv_h, kv_h+4, kv_h+8, ...
# Position i in scores_flat: this is the i-th column split of the (4,4) tile at seq pos i//4.
# Columns are in order: col0, col1, col2, col3 for each seq pos.
# So positions kv_h, kv_h+4, kv_h+8, ... = col_kv_h at seq positions 0, 1, 2, ...
# That's the correct column kv_h for each seq pos. ✓
#
# Similarly for V splitting:
# v_b_rows = retile_streamify(v_full_b, chunk=1, split_row=True): rows interleaved.
# stream(1, seq_b*4) tile(1,32). Row order: row0_seq0, row1_seq0, row2_seq0, row3_seq0, row0_seq1,...
# flatten → stream(seq_b*4,). parallelize by 4: consumer kv_h gets positions kv_h, kv_h+4, ...
# = row_kv_h at seq pos 0, 1, 2, ... ✓ 
# So V and scores extraction are both correct.
#
# The main bug is in apply_rope. Let me also fix the reassembly:
# After flat parallelize of x_split (stream(64,2*n_heads) → flatten → stream(64*2*n_heads,)):
# Consumer 0: even positions = first-half tiles in layout order (b=0,h=0), (b=0,h=1), ...
# So out1[b*n_heads+h] = RoPE out for first half of head h, batch b.
# static_reassemble interleaves: pos 0=out1[0] (b0,h0 half1), pos 1=out2[0] (b0,h0 half2),
#   pos 2=out1[1] (b0,h1 half1), pos 3=out2[1] (b0,h1 half2), ...
# reshape_stream(chunk=2) → groups pairs → stream(B*n_heads, 2) tile(1,16).
# accum_retile_col(rank=1) → stream(B*n_heads,) tile(1,32). Each tile = [half1|half2] for (b,h). ✓
# reshape_stream(chunk=n_heads, rank=0) → stream(B, n_heads). ✓

def tiled_reference(dims, tensors):
    B = 64
    D = 512
    HEAD_DIM = 32
    NUM_HEADS = 16
    NUM_KV = 4
    QPK = 4
    N_EXPERTS = 8
    N_ACTIVE = 2
    FFN_DIM = 1792
    FFN_TILE = 32
    FFN_CHUNKS = FFN_DIM // FFN_TILE  # 56
    TILE_N = 32
    W_CHUNKS = D // TILE_N            # 16
    HALF = HEAD_DIM // 2              # 16

    def rms_norm_stream(x, dim):
        sq = unary_square(x)
        s = unary_rowwise_sum(sq)
        m = unary_mul_imm(s, 1.0 / dim)
        e = unary_add_imm(m, 1e-6)
        r = unary_rsqrt(e)
        rr = repeat_static(r, dim)
        rt = accum_retile_col(rr, rank=1)
        return binary_mul(x, rt)

    def promote_outer_static_1(x):
        if hasattr(x, "underlying_tensor"):
            return StepTensor(
                x.underlying_tensor.unsqueeze(0),
                stream_dtype=x.stream_dtype,
                dyn_mask=(False,) + x.dyn_mask,
                dyn_origins=(None,) + x.dyn_origins,
            )
        y = PromoteOuter(graph, x)
        s = _dsl2step_stream(x)
        y._stream = Stream(stream_dtype=s.stream_dtype, shape=(1,) + s.shape)
        return y

    # =========================================================================
    # Step 1: Load input and RMSNorm
    # =========================================================================
    inp_raw = offchip_load(
        tensors["input_tensor"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )
    inp = flatten(inp_raw, min_rank=0, max_rank=1)
    normed = rms_norm_stream(inp, D)

    # =========================================================================
    # Step 2: QKV projections
    # =========================================================================
    q_proj_w = flatten(
        offchip_load(tensors["q_proj"], stride=(0, 1), out_shape_tiled=(B, NUM_HEADS),
                     tile_row=D, tile_col=HEAD_DIM),
        min_rank=1, max_rank=2
    )
    normed_q = repeat_ref(normed, q_proj_w)
    Q_raw = binary_matmul(normed_q, q_proj_w)

    k_proj_w = flatten(
        offchip_load(tensors["k_proj"], stride=(0, 1), out_shape_tiled=(B, NUM_KV),
                     tile_row=D, tile_col=HEAD_DIM),
        min_rank=1, max_rank=2
    )
    normed_k = repeat_ref(normed, k_proj_w)
    K_raw = binary_matmul(normed_k, k_proj_w)

    v_proj_w = flatten(
        offchip_load(tensors["v_proj"], stride=(0, 1), out_shape_tiled=(B, NUM_KV),
                     tile_row=D, tile_col=HEAD_DIM),
        min_rank=1, max_rank=2
    )
    normed_v = repeat_ref(normed, v_proj_w)
    V_raw = binary_matmul(normed_v, v_proj_w)

    # =========================================================================
    # Step 3: Per-head RMSNorm
    # =========================================================================
    Q_normed = rms_norm_stream(Q_raw, HEAD_DIM)
    K_normed = rms_norm_stream(K_raw, HEAD_DIM)

    # =========================================================================
    # Step 4: RoPE — fixed parallelize to split on innermost (head-half) dim
    # =========================================================================
    cos_s = flatten(
        offchip_load(tensors["cos"], stride=(1, 0), out_shape_tiled=(B, 1),
                     tile_row=1, tile_col=HEAD_DIM),
        min_rank=0, max_rank=2
    )
    sin_s = flatten(
        offchip_load(tensors["sin"], stride=(1, 0), out_shape_tiled=(B, 1),
                     tile_row=1, tile_col=HEAD_DIM),
        min_rank=0, max_rank=2
    )

    def apply_rope(x_heads, cos_1d, sin_1d, n_heads):
        """
        x_heads: stream(B, n_heads), tile(1, HEAD_DIM)
        cos_1d, sin_1d: stream(B,), tile(1, HEAD_DIM)
        Returns: stream(B, n_heads), tile(1, HEAD_DIM)
        
        Fix: flatten to 1D then parallelize by 2 to correctly split
        first-half and second-half tiles.
        """
        cos_exp = repeat_ref(cos_1d, x_heads)
        sin_exp = repeat_ref(sin_1d, x_heads)
        # Split into two half-width tiles
        x_split = retile_streamify(x_heads, chunk=HALF, split_row=False)
        cos_split = retile_streamify(cos_exp, chunk=HALF, split_row=False)
        sin_split = retile_streamify(sin_exp, chunk=HALF, split_row=False)
        # x_split: stream(B, n_heads*2), tile(1, HALF)
        # Tile layout: for each (b, h): [half1, half2] → interleaved at innermost dim.
        # Flatten to 1D so parallelize splits on tile position (even=half1, odd=half2)
        x_flat = flatten(x_split, min_rank=0, max_rank=1)
        cos_flat = flatten(cos_split, min_rank=0, max_rank=1)
        sin_flat = flatten(sin_split, min_rank=0, max_rank=1)
        # x_flat: stream(B*n_heads*2,), tile(1, HALF)
        # Tile order: (b0,h0,half1), (b0,h0,half2), (b0,h1,half1), ...
        # Wait: stream(B, n_heads*2) flattened: row-major over (B, n_heads*2).
        # Position 2*(b*n_heads+h) = half1 of (b,h), 2*(b*n_heads+h)+1 = half2. ✓
        x_halves = parallelize(x_flat, 2)
        cos_halves = parallelize(cos_flat, 2)
        sin_halves = parallelize(sin_flat, 2)
        # halves[0]: even positions = all first halves: stream(B*n_heads,) tile(1,HALF)
        # halves[1]: odd positions = all second halves: stream(B*n_heads,) tile(1,HALF)
        x1, x2 = x_halves[0], x_halves[1]
        c1, c2 = cos_halves[0], cos_halves[1]
        s1, s2 = sin_halves[0], sin_halves[1]
        neg_x2 = unary_mul_imm(x2, -1.0)
        out1 = binary_add(binary_mul(x1, c1), binary_mul(neg_x2, s1))
        out2 = binary_add(binary_mul(x2, c2), binary_mul(x1, s2))
        # out1, out2: stream(B*n_heads,), tile(1, HALF)
        # Reassemble: interleave back
        interleaved = static_reassemble([out1, out2])
        # stream(2*B*n_heads,), tile(1, HALF)
        # Tile order: (b0,h0,half1_rope), (b0,h0,half2_rope), (b0,h1,half1_rope), ...
        inter_2d = reshape_stream(interleaved, chunk_size=2, rank=0)
        # stream(B*n_heads, 2), tile(1, HALF)
        merged = accum_retile_col(inter_2d, rank=1)
        # stream(B*n_heads,), tile(1, HEAD_DIM)
        result = reshape_stream(merged, chunk_size=n_heads, rank=0)
        # stream(B, n_heads), tile(1, HEAD_DIM) ✓
        return result

    Q_rope = apply_rope(Q_normed, cos_s, sin_s, NUM_HEADS)
    # Q_rope: stream(B=64, NUM_HEADS=16), tile(1, 32)
    K_rope = apply_rope(K_normed, cos_s, sin_s, NUM_KV)
    # K_rope: stream(B=64, NUM_KV=4), tile(1, 32)

    # =========================================================================
    # Step 5: Prepare K, V for cache insertion
    # =========================================================================
    K_batched = accum_retile_row(K_rope, rank=1)
    V_batched = accum_retile_row(V_raw, rank=1)

    K_by_batch = parallelize(K_batched, B)
    V_by_batch = parallelize(V_batched, B)
    Q_by_batch = parallelize(Q_rope, B)
    # Q_by_batch[b]: stream(1, NUM_HEADS), tile(1, 32)

    # =========================================================================
    # Step 6: Per-batch attention
    # =========================================================================
    attn_outputs_per_batch = []

    for b in range(B):
        idx_b_meta = metadata_gen(tensors["idx"][b])
        seq_b_meta = metadata_gen(tensors["seq_len"][b])
        idx_zero_b = unary_to_const_int(idx_b_meta, 0)

        raddr_b = cache_read_addr_gen(idx_zero_b, seq_b_meta, row_offset=1)
        k_b_cache = random_offchip_load(
            tensors["k_cache"][b], raddr=raddr_b, tile_row=NUM_KV, tile_col=HEAD_DIM
        )
        v_b_cache = random_offchip_load(
            tensors["v_cache"][b], raddr=raddr_b, tile_row=NUM_KV, tile_col=HEAD_DIM
        )

        K_b_new = K_by_batch[b]
        V_b_new = V_by_batch[b]

        last_sel_b = filter_last_tile(seq_b_meta)
        k_last_b, k_prefix_b = flat_partition(k_b_cache, last_sel_b, n=2, partition_rank=0)
        v_last_b, v_prefix_b = flat_partition(v_b_cache, last_sel_b, n=2, partition_rank=0)

        k_last_off = binary_set_offset(k_last_b, idx_zero_b)
        k_last_updated = binary_row_wise_append(k_last_off, K_b_new)
        v_last_off = binary_set_offset(v_last_b, idx_zero_b)
        v_last_updated = binary_row_wise_append(v_last_off, V_b_new)

        k_full_raw = flat_reassemble([k_last_updated, k_prefix_b], last_sel_b, reassemble_rank=0)
        k_full_b = flatten(k_full_raw, min_rank=0, max_rank=1)

        v_full_raw = flat_reassemble([v_last_updated, v_prefix_b], last_sel_b, reassemble_rank=0)
        v_full_b = flatten(v_full_raw, min_rank=0, max_rank=1)

        v_b_rows = retile_streamify(v_full_b, chunk=1, split_row=True)
        v_b_flat = flatten(v_b_rows, min_rank=0, max_rank=1)
        v_b_by_head = parallelize(v_b_flat, NUM_KV)

        Q_b = Q_by_batch[b]  # stream(1, 16), tile(1,32)
        Q_b_flat16 = flatten(Q_b, min_rank=0, max_rank=1)
        Q_b_grouped = reshape_stream(Q_b_flat16, chunk_size=QPK, rank=0)
        Q_b_kv = accum_retile_row(Q_b_grouped, rank=1)
        # stream(4,), tile(4,32) — no flatten needed
        Q_b_by_kv = parallelize(Q_b_kv, NUM_KV)

        kv_results = []
        for kv_h in range(NUM_KV):
            Q_kv_h = Q_b_by_kv[kv_h]
            v_kv = promote_outer_static_1(v_b_by_head[kv_h])

            Q_kv_h_exp = repeat_ref(Q_kv_h, k_full_b)
            scores_4x4 = binary_matmul(Q_kv_h_exp, k_full_b, weight_transposed=True)

            scores_cols = retile_streamify(scores_4x4, chunk=1, split_row=False)
            scores_flat = flatten(scores_cols, min_rank=0, max_rank=1)
            scores_by_kv = parallelize(scores_flat, NUM_KV)
            scores_kv_h = promote_outer_static_1(scores_by_kv[kv_h])

            row_max = accum_max(scores_kv_h, rank=1)
            row_max_exp = repeat_ref(row_max, scores_kv_h)
            scores_shifted = binary_add(scores_kv_h, unary_mul_imm(row_max_exp, -1.0))
            exp_s = unary_exp(scores_shifted)

            context_num = binary_matmul(exp_s, v_kv)
            context_sum = accum_add(context_num, rank=1)

            denom = accum_add(exp_s, rank=1)
            denom_rep = repeat_static(denom, HEAD_DIM)
            denom_tiled = accum_retile_col(denom_rep, rank=1)

            context_kv = binary_div(context_sum, denom_tiled)

            context_kv_1d = promote(context_kv, rank=0)
            heads_split = retile_streamify(context_kv_1d, chunk=1, split_row=True)
            # stream(4,), tile(1,32)

            heads_2d = promote_outer(heads_split)
            heads_group = accum_retile_row(heads_2d, rank=1)
            kv_results.append(heads_group)

        attn_b_groups = static_reassemble(kv_results)
        attn_b_2d = promote_outer(attn_b_groups)
        attn_b_data = accum_retile_row(attn_b_2d, rank=1)
        attn_outputs_per_batch.append(attn_b_data)

    attn_hd = static_reassemble(attn_outputs_per_batch)
    attn_rows = retile_streamify(attn_hd, chunk=1, split_row=True)
    attn_2d = attn_rows

    # =========================================================================
    # Step 7: O-projection
    # =========================================================================
    attn_merged = accum_retile_col(attn_2d, rank=1)

    o_proj_w = flatten(
        offchip_load(tensors["o_proj_weight"], stride=(0, 1), out_shape_tiled=(B, W_CHUNKS),
                     tile_row=D, tile_col=TILE_N),
        min_rank=1, max_rank=2
    )
    attn_for_proj = repeat_ref(attn_merged, o_proj_w)
    o_mm = binary_matmul(attn_for_proj, o_proj_w)
    o_proj_result = accum_retile_col(o_mm, rank=1)

    # =========================================================================
    # Step 8: Residual add
    # =========================================================================
    res_add_0 = binary_add(o_proj_result, inp)

    # =========================================================================
    # Step 9: RMS Norm
    # =========================================================================
    normed_2 = rms_norm_stream(res_add_0, D)

    # =========================================================================
    # Step 10: MoE
    # =========================================================================
    sel_moe = select_gen(tensors["expert_onehot"], is_multihot=False, n=N_EXPERTS)

    normed_2_outer = promote_outer(normed_2)
    normed_2_rep = repeat_static(normed_2_outer, N_ACTIVE)

    parts_moe = flat_partition(normed_2_rep, sel_moe, n=N_EXPERTS, partition_rank=0)

    expert_results = []
    for e in range(N_EXPERTS):
        xi = parts_moe[e]
        gate_w = offchip_load_ref(xi, tensors["w_gate_list"][e],
                                   stride=(1,), out_shape_tiled=(FFN_CHUNKS,),
                                   tile_row=D, tile_col=FFN_TILE)
        up_w = offchip_load_ref(xi, tensors["w_up_list"][e],
                                 stride=(1,), out_shape_tiled=(FFN_CHUNKS,),
                                 tile_row=D, tile_col=FFN_TILE)
        xi_exp = repeat_ref(xi, gate_w)
        gate_out = binary_matmul(xi_exp, gate_w)
        up_out = binary_matmul(xi_exp, up_w)
        hidden = binary_mul(unary_silu(gate_out), up_out)
        down_w = offchip_load_ref(xi, tensors["w_down_list"][e],
                                   stride=(1,), out_shape_tiled=(FFN_CHUNKS,),
                                   tile_row=FFN_TILE, tile_col=D)
        down_out = binary_map_accum(hidden, down_w, rank=1)
        expert_results.append(down_out)

    moe_reassembled = flat_reassemble(expert_results, sel_moe, reassemble_rank=0)
    per_slot_sum = accum_add(moe_reassembled, rank=1)

    ew_stream = offchip_load(
        tensors["expert_weights"],
        stride=(N_ACTIVE, 1),
        out_shape_tiled=(B, N_ACTIVE),
        tile_row=1, tile_col=1,
    )

    per_slot_flat = flatten(per_slot_sum, min_rank=1, max_rank=2)
    ew_flat = flatten(ew_stream, min_rank=1, max_rank=2)

    weighted_moe = binary_mul(per_slot_flat, ew_flat)
    moe_out = accum_add(weighted_moe, rank=1)

    # =========================================================================
    # Step 11: Final residual add
    # =========================================================================
    final = binary_add(moe_out, res_add_0)
    final_out = promote_outer(final)
    return offchip_store(final_out)
