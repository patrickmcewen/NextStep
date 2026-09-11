def attention_block_1(input_tensor, q_proj, k_proj, v_proj, cos, sin, k_cache, v_cache, num_token_list, o_proj_weight, *, out_shapes):
    B = 64
    D = 512
    HEAD_DIM = 32
    HALF = 16
    NUM_HEADS = 16
    NUM_KV_HEADS = 4
    QPK = 4
    MAX_N = 4096
    D_CHUNKS = D // HEAD_DIM
    KV_D_CHUNKS = (NUM_KV_HEADS * HEAD_DIM) // HEAD_DIM
    PAR_FACTOR = 16
    CBW = 16

    # ============================================================
    # 1. Load input & RMSNorm
    # ============================================================
    x_raw = offchip_load(input_tensor, stride=(1,), out_shape_tiled=(B,),
                         tile_row=1, tile_col=D, par_dispatch=8)
    x = flatten(x_raw, min_rank=0, max_rank=1)

    x_sq = unary_square(x)
    x_rsum = unary_rowwise_sum(x_sq)
    x_mean = unary_mul_imm(x_rsum, 1.0 / D, compute_bw=CBW)
    x_eps = unary_add_imm(x_mean, 1e-6, compute_bw=CBW)
    x_inv = unary_rsqrt(x_eps, compute_bw=CBW)
    x_inv_rep = repeat_static(x_inv, D)
    x_inv_tiled = accum_retile_col(x_inv_rep, rank=1)
    normed = binary_mul(x, x_inv_tiled, compute_bw=CBW)

    # ============================================================
    # 2. QKV Projections – aggressive shared parallelism (16-way)
    # ============================================================
    q_w = offchip_load_ref(normed, q_proj, stride=(1,), out_shape_tiled=(D_CHUNKS,),
                           tile_row=D, tile_col=HEAD_DIM, par_dispatch=8)
    kv_w_k = offchip_load_ref(normed, k_proj, stride=(1,), out_shape_tiled=(KV_D_CHUNKS,),
                               tile_row=D, tile_col=HEAD_DIM, par_dispatch=8)
    kv_w_v = offchip_load_ref(normed, v_proj, stride=(1,), out_shape_tiled=(KV_D_CHUNKS,),
                               tile_row=D, tile_col=HEAD_DIM, par_dispatch=8)

    normed_par = parallelize(normed, PAR_FACTOR)
    q_w_par = parallelize(q_w, PAR_FACTOR)
    kv_k_par = parallelize(kv_w_k, PAR_FACTOR)
    kv_v_par = parallelize(kv_w_v, PAR_FACTOR)

    Q_parts = []
    K_parts = []
    V_parts = []
    for i in range(PAR_FACTOR):
        n_q = repeat_ref(normed_par[i], q_w_par[i])
        Q_parts.append(binary_matmul(n_q, q_w_par[i], compute_bw=CBW))

        n_k = repeat_ref(normed_par[i], kv_k_par[i])
        K_parts.append(binary_matmul(n_k, kv_k_par[i], compute_bw=CBW))

        n_v = repeat_ref(normed_par[i], kv_v_par[i])
        V_parts.append(binary_matmul(n_v, kv_v_par[i], compute_bw=CBW))

    Q_full = static_reassemble(Q_parts)
    K_full = static_reassemble(K_parts)
    V_full = static_reassemble(V_parts)

    # ============================================================
    # 3. Per-head RMSNorm on Q and K
    # ============================================================
    def per_head_rms(h_stream, dim):
        sq   = unary_square(h_stream, compute_bw=CBW)
        rs   = unary_rowwise_sum(sq)
        m    = unary_mul_imm(rs, 1.0 / dim, compute_bw=CBW)
        e    = unary_add_imm(m, 1e-6, compute_bw=CBW)
        inv  = unary_rsqrt(e, compute_bw=CBW)
        irep = repeat_static(inv, dim)
        itil = accum_retile_col(irep, rank=1)
        return binary_mul(h_stream, itil, compute_bw=CBW)

    Q_normed = per_head_rms(Q_full, HEAD_DIM)
    K_normed = per_head_rms(K_full, HEAD_DIM)

    # ============================================================
    # 4. RoPE
    # ============================================================
    cos_raw = offchip_load(cos, stride=(1, 0), out_shape_tiled=(B, 1),
                           tile_row=1, tile_col=HEAD_DIM, par_dispatch=8)
    sin_raw = offchip_load(sin, stride=(1, 0), out_shape_tiled=(B, 1),
                           tile_row=1, tile_col=HEAD_DIM, par_dispatch=8)
    cos_s = flatten(cos_raw, min_rank=0, max_rank=2)
    sin_s = flatten(sin_raw, min_rank=0, max_rank=2)

    def apply_rope(heads, n_heads):
        cos_bc = repeat_static(cos_s, n_heads)
        sin_bc = repeat_static(sin_s, n_heads)

        h_sp = retile_streamify(heads,  chunk=HALF, split_row=False)
        c_sp = retile_streamify(cos_bc, chunk=HALF, split_row=False)
        s_sp = retile_streamify(sin_bc, chunk=HALF, split_row=False)

        h_f = flatten(h_sp, min_rank=0, max_rank=1)
        c_f = flatten(c_sp, min_rank=0, max_rank=1)
        s_f = flatten(s_sp, min_rank=0, max_rank=1)

        hh = parallelize(h_f, 2)
        ch = parallelize(c_f, 2)
        sh = parallelize(s_f, 2)
        x1, x2 = hh[0], hh[1]
        c1, c2 = ch[0], ch[1]
        s1, s2 = sh[0], sh[1]

        neg_x2 = unary_mul_imm(x2, -1.0, compute_bw=CBW)
        r1 = binary_add(binary_mul(x1, c1, compute_bw=CBW), binary_mul(neg_x2, s1, compute_bw=CBW), compute_bw=CBW)
        r2 = binary_add(binary_mul(x2, c2, compute_bw=CBW), binary_mul(x1, s2, compute_bw=CBW), compute_bw=CBW)

        reassembled = static_reassemble([r1, r2])
        rs2 = reshape_stream(reassembled, chunk_size=2, rank=0)
        merged = accum_retile_col(rs2, rank=1)
        restored = reshape_stream(merged, chunk_size=n_heads, rank=0)
        return restored

    Q_rope = apply_rope(Q_normed, NUM_HEADS)
    K_rope = apply_rope(K_normed, NUM_KV_HEADS)

    # ============================================================
    # 5. KV Cache write
    # ============================================================
    K_new = accum_retile_row(K_rope, rank=1)
    V_new = accum_retile_row(V_full, rank=1)

    ntl_meta = metadata_gen(num_token_list)
    ntl = flatten(ntl_meta, min_rank=0, max_rank=1)

    K_bwr  = parallelize(K_new, B)
    V_bwr  = parallelize(V_new, B)
    ntl_wr = parallelize(ntl, B)

    for b in range(B):
        nb   = ntl_wr[b]
        bidx = unary_to_const_int(nb, b)
        kwa  = binary_cache_write_addr_gen(bidx, nb, row_offset=MAX_N)
        random_offchip_store(k_cache, wdata=K_bwr[b], waddr=kwa,
                             tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, par_dispatch=8)
        vwa  = binary_cache_write_addr_gen(bidx, nb, row_offset=MAX_N)
        random_offchip_store(v_cache, wdata=V_bwr[b], waddr=vwa,
                             tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, par_dispatch=8)

    # ============================================================
    # 6. GQA Attention per batch item
    # ============================================================
    Q_for_attn = accum_retile_row(Q_rope, rank=1)
    Q_grp_flat = retile_streamify(Q_for_attn, chunk=QPK, split_row=True)
    Q_groups   = reshape_stream(Q_grp_flat, chunk_size=NUM_KV_HEADS, rank=0)

    Q_batched  = parallelize(Q_groups, B)
    K_new_bat  = parallelize(K_new, B)
    V_new_bat  = parallelize(V_new, B)
    ntl_bat    = parallelize(ntl, B)

    attn_out_list = []
    for b in range(B):
        Q_kv = Q_batched[b]
        K_b  = K_new_bat[b]
        V_b  = V_new_bat[b]
        nb   = ntl_bat[b]

        one  = unary_to_const_int(nb, 1)
        sl   = binary_cache_write_addr_gen(nb, one, row_offset=1)

        bidx2 = unary_to_const_int(nb, b)
        raddr = cache_read_addr_gen(bidx2, sl, row_offset=MAX_N)

        k_full = random_offchip_load(k_cache, raddr=raddr,
                                     tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, par_dispatch=8)
        v_full = random_offchip_load(v_cache, raddr=raddr,
                                     tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, par_dispatch=8)

        last_sel = filter_last_tile(sl)
        k_last, k_notlast = flat_partition(k_full, last_sel, n=2, partition_rank=0)
        v_last, v_notlast = flat_partition(v_full, last_sel, n=2, partition_rank=0)

        offset_val   = unary_to_const_int(nb, 0)
        k_last_off   = binary_set_offset(k_last, offset_val)
        k_updated    = binary_row_wise_append(k_last_off, K_b)

        offset_val_v = unary_to_const_int(nb, 0)
        v_last_off   = binary_set_offset(v_last, offset_val_v)
        v_updated    = binary_row_wise_append(v_last_off, V_b)

        k_seq_raw = flat_reassemble([k_updated, k_notlast], control=last_sel, reassemble_rank=0)
        v_seq_raw = flat_reassemble([v_updated, v_notlast], control=last_sel, reassemble_rank=0)
        k_seq = flatten(k_seq_raw, min_rank=0, max_rank=1)
        v_seq = flatten(v_seq_raw, min_rank=0, max_rank=1)

        Q_kv_flat = flatten(Q_kv, min_rank=0, max_rank=1)
        Q_per_head = parallelize(Q_kv_flat, NUM_KV_HEADS)

        k_rows    = retile_streamify(k_seq, chunk=1, split_row=True)
        k_rows_3d = reshape_pad_stream(k_rows, chunk_size=NUM_KV_HEADS, reshape_rank=0)
        k_buf     = bufferize(k_rows_3d, rank=1)

        v_rows    = retile_streamify(v_seq, chunk=1, split_row=True)
        v_rows_3d = reshape_pad_stream(v_rows, chunk_size=NUM_KV_HEADS, reshape_rank=0)
        v_buf     = bufferize(v_rows_3d, rank=1)

        group_outputs = []
        for h in range(NUM_KV_HEADS):
            Q_h = Q_per_head[h]

            if h == 0:
                k_h_st  = streamify(k_buf, stride=(1,), out_shape_tiled=(1,))
                k_row_h = accum_add(k_h_st, rank=1)
            else:
                k_h_st    = streamify(k_buf, stride=(1,), out_shape_tiled=(h+1,))
                k_h_acc   = accum_add(k_h_st, rank=1)
                k_hm1_st  = streamify(k_buf, stride=(1,), out_shape_tiled=(h,))
                k_hm1_acc = accum_add(k_hm1_st, rank=1)
                k_row_h   = binary_add(k_h_acc, unary_mul_imm(k_hm1_acc, -1.0, compute_bw=CBW), compute_bw=CBW)

            if h == 0:
                v_h_st  = streamify(v_buf, stride=(1,), out_shape_tiled=(1,))
                v_row_h = accum_add(v_h_st, rank=1)
            else:
                v_h_st    = streamify(v_buf, stride=(1,), out_shape_tiled=(h+1,))
                v_h_acc   = accum_add(v_h_st, rank=1)
                v_hm1_st  = streamify(v_buf, stride=(1,), out_shape_tiled=(h,))
                v_hm1_acc = accum_add(v_hm1_st, rank=1)
                v_row_h   = binary_add(v_h_acc, unary_mul_imm(v_hm1_acc, -1.0, compute_bw=CBW), compute_bw=CBW)

            Q_h_2d  = reshape_stream(Q_h, chunk_size=1, rank=0)
            Q_h_exp = expand_ref(Q_h_2d, k_row_h, expand_rank=1)

            scores = binary_matmul(Q_h_exp, k_row_h, weight_transposed=True, compute_bw=CBW)

            row_max = accum_max(scores, rank=1)
            rm_2d   = reshape_stream(row_max, chunk_size=1, rank=0)
            rm_bc   = expand_ref(rm_2d, scores, expand_rank=1)
            sh_s    = binary_add(scores, unary_mul_imm(rm_bc, -1.0, compute_bw=CBW), compute_bw=CBW)
            exp_s   = unary_exp(sh_s, compute_bw=CBW)

            denom   = accum_add(exp_s, rank=1)

            context_raw = binary_matmul(exp_s, v_row_h, compute_bw=CBW)
            context     = accum_add(context_raw, rank=1)

            d_rep   = repeat_static(denom, HEAD_DIM)
            d_tiled = accum_retile_col(d_rep, rank=1)
            out_h   = binary_div(context, d_tiled, compute_bw=CBW)

            group_outputs.append(out_h)

        batch_out  = static_reassemble(group_outputs)
        batch_out2 = promote_outer(batch_out)
        attn_out_list.append(batch_out2)

    all_attn  = static_reassemble(attn_out_list)
    attn_hd   = accum_retile_row(all_attn, rank=1)

    attn_rows = retile_streamify(attn_hd, chunk=1, split_row=True)
    attn_2d   = reshape_stream(attn_rows, chunk_size=NUM_HEADS, rank=0)
    attn_512  = accum_retile_col(attn_2d, rank=1)

    # ============================================================
    # 7. O-projection + residual (16-way shared parallelism)
    # ============================================================
    ow       = offchip_load_ref(attn_512, o_proj_weight, stride=(1,),
                                out_shape_tiled=(D_CHUNKS,), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8)
    attn_512_par = parallelize(attn_512, PAR_FACTOR)
    ow_par = parallelize(ow, PAR_FACTOR)
    o_parts = []
    for i in range(PAR_FACTOR):
        o_rep = repeat_ref(attn_512_par[i], ow_par[i])
        o_mm = binary_matmul(o_rep, ow_par[i], compute_bw=CBW)
        o_parts.append(o_mm)
    o_mm_full = static_reassemble(o_parts)
    o_out    = accum_retile_col(o_mm_full, rank=1)

    res_add_0 = binary_add(o_out, x, compute_bw=CBW)

    # ============================================================
    # 8. RMSNorm on res_add_0
    # ============================================================
    r_sq   = unary_square(res_add_0, compute_bw=CBW)
    r_sum  = unary_rowwise_sum(r_sq)
    r_mean = unary_mul_imm(r_sum, 1.0 / D, compute_bw=CBW)
    r_eps  = unary_add_imm(r_mean, 1e-6, compute_bw=CBW)
    r_inv  = unary_rsqrt(r_eps, compute_bw=CBW)
    r_irep = repeat_static(r_inv, D)
    r_itil = accum_retile_col(r_irep, rank=1)
    normed_2 = binary_mul(res_add_0, r_itil, compute_bw=CBW)

    # ============================================================
    # 9. Promote to (64,1,1,512)
    # ============================================================
    normed_2_out  = promote(normed_2, rank=0)
    res_add_0_out = promote(res_add_0, rank=0)

    return normed_2_out, res_add_0_out

def moe_block_0(normed_2, res_add_0, expert_onehot, expert_weights, w_gate_list, w_up_list, w_down_list, *, out_shapes):
    batch = 64
    n_experts = 8
    n_active = 2
    D = 512
    F_dim = 1792
    TILE_F = 64
    F_TILES = F_dim // TILE_F  # 28

    normed_flat = flatten(normed_2, min_rank=0, max_rank=1)
    res_flat = flatten(res_add_0, min_rank=0, max_rank=1)

    normed_po = promote_outer(normed_flat)
    normed_rep = repeat_static(normed_po, n_active)

    sel = select_gen(expert_onehot, is_multihot=False, n=n_experts)
    partitioned = flat_partition(normed_rep, sel, n=n_experts, partition_rank=0)

    expert_outputs = []
    for i in range(n_experts):
        xi = partitioned[i]

        gate_w = offchip_load_ref(
            xi, w_gate_list[i],
            stride=(1,), out_shape_tiled=(F_TILES,),
            tile_row=D, tile_col=TILE_F,
            par_dispatch=32,
        )
        up_w = offchip_load_ref(
            xi, w_up_list[i],
            stride=(1,), out_shape_tiled=(F_TILES,),
            tile_row=D, tile_col=TILE_F,
            par_dispatch=32,
        )
        down_w = offchip_load_ref(
            xi, w_down_list[i],
            stride=(1,), out_shape_tiled=(F_TILES,),
            tile_row=TILE_F, tile_col=D,
            par_dispatch=32,
        )

        xi_exp = repeat_ref(xi, gate_w)

        gate_out = binary_matmul(xi_exp, gate_w, compute_bw=32)
        up_out   = binary_matmul(xi_exp, up_w, compute_bw=32)
        hidden   = binary_mul(unary_silu(gate_out, compute_bw=32), up_out, compute_bw=32)
        down_out = binary_map_accum(hidden, down_w, rank=1, compute_bw=32)

        expert_outputs.append(down_out)

    reassembled = flat_reassemble(expert_outputs, sel, reassemble_rank=0)
    per_slot = accum_add(reassembled, rank=1)

    ew_stream = offchip_load(
        expert_weights,
        stride=(n_active, 1),
        out_shape_tiled=(batch, n_active),
        tile_row=1, tile_col=1,
        par_dispatch=32,
    )

    weighted = binary_mul(per_slot, ew_stream, compute_bw=32)
    moe_sum = accum_add(weighted, rank=1)

    moe_flat = flatten(moe_sum, min_rank=0, max_rank=1)
    final = binary_add(moe_flat, res_flat, compute_bw=32)
    result = promote(final, rank=0)
    return result

def tiled_reference(dims, tensors):
    # Use a medium-perf, low-memory attention variant paired with the
    # highest-performance MoE variant. This creates a new Pareto point
    # (~2.35M cycles, ~4.95MB on-chip) distinct from the previously-
    # accepted low-memory/slow-MoE mix and the fast-but-high-memory combo.
    normed_2, res_add_0 = attention_block_1(
        tensors["input_tensor"],
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        tensors["k_cache"],
        tensors["v_cache"],
        tensors["num_token_list"],
        tensors["o_proj_weight"],
        out_shapes=((64, 1, 1, 512), (64, 1, 1, 512)),
    )

    moe_out = moe_block_0(
        normed_2,
        res_add_0,
        tensors["expert_onehot"],
        tensors["expert_weights"],
        tensors["w_gate_list"],
        tensors["w_up_list"],
        tensors["w_down_list"],
        out_shapes=((64, 1, 1, 512),),
    )

    return offchip_store(moe_out)