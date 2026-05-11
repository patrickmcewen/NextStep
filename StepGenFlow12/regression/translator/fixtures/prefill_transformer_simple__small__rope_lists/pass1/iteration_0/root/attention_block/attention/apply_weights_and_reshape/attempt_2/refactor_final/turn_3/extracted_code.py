# This implementation follows the PyTorch reference:
#   attn_weights @ Vh  →  (Hkv, Q_per_KV, S, D)
#   then reshaped to (S, Hkv*Q_per_KV, D)
#
# The DSL does not provide a direct stream↔tile transpose, so we
#   1. compute the weighted sum (still stream = heads, tile‑row = seq_len);
#   2. create a zero‑filled output tensor of shape (seq_len,)×tile(num_heads, head_dim);
#   3. split the weighted result into per‑head streams via `parallelize`;
#   4. for each head, generate a constant offset tensor (stream = seq_len, tile = 1×1)
#      indicating the target row inside the per‑token tile;
#   5. split the head’s data into (seq_len,)×tile(1, head_dim);
#   6. use `binary_set_offset` + `binary_row_wise_append` to write the head’s rows
#      into the output at the proper offsets.
#
# The loop assembles the final tensor whose stream dimension is the sequence
# length and whose tile rows index the heads, exactly matching the required
# vanilla shape (seq_len, num_heads, head_dim).
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Vh across the second KV‑head dimension and compute the weighted sum.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)
    weighted = binary_matmul(attn_weights, Vh_exp)          # stream (Hkv, Q_per_KV), tile (S, D)

    # 2. Number of heads = Hkv * Q_per_KV.
    num_heads = attn_weights.shape[0] * attn_weights.shape[1]

    # 3. Build a zero‑filled tensor of shape (seq_len,) × tile(num_heads, head_dim).
    #    This will serve as the destination for the per‑head rows.
    zeros = binary_sub_imm(binary_is_equal(weighted, weighted), 1.0)          # all zeros, same shape as `weighted`
    zeros_flat = flatten(zeros, min_rank=0, max_rank=1)                       # merge head streams → stream (num_heads,)
    zeros_merged = accum_retile_row(zeros_flat, rank=1)                       # tile rows = num_heads * seq_len
    zeros_promoted = promote(zeros_merged, rank=0)                            # add leading stream dim
    result = retile_streamify(zeros_promoted, chunk=num_heads, split_row=True)  # stream = seq_len, tile (num_heads, head_dim)

    # 4. Split the weighted tensor into per‑head streams.
    head_streams = parallelize(weighted, num_heads)   # list of `num_heads` tensors, each stream (1,) tile (seq_len, head_dim)

    # 5. Scatter each head’s rows into `result` at the correct offset.
    for h_idx, head in enumerate(head_streams):
        # a) Build a constant‑offset tensor of shape (seq_len,) × tile(1,1) with value = h_idx.
        #    Start from a zero‑like tensor derived from the current head.
        zeros_mask = binary_sub_imm(binary_is_equal(head, head), 1.0)          # zeros, shape (1,) tile (seq_len, head_dim)
        offsets_sum = unary_rowwise_sum(zeros_mask)                            # shape (1,) tile (seq_len, 1)
        offsets = retile_streamify(offsets_sum, chunk=1, split_row=True)       # shape (seq_len,) tile (1,1)
        offsets = unary_add_imm(offsets, constant=float(h_idx))                # now all entries == h_idx

        # b) Convert the head data into (seq_len,) × tile(1, head_dim).
        rows = retile_streamify(head, chunk=1, split_row=True)                 # shape (seq_len,) tile (1, head_dim)

        # c) Write the rows into the destination at the computed offsets.
        result = binary_row_wise_append(binary_set_offset(result, offsets), rows)

    # `result` now has stream shape (seq_len,) and tile shape (num_heads, head_dim),
    # i.e. vanilla shape (seq_len, num_heads, head_dim) as required.
    return result