# The node receives on‑chip tensors:
#   attn_weights: shape (4, 4, 64, 64) → stream (4,4) tile (64,64)
#   Vh:          shape (4, 1, 64, 32) → stream (4,1) tile (64,32)
# The reference computes `attn_weights @ Vh` → shape (4,4,64,32) and then
# reshapes it to vanilla shape (64, 16, 32).  This is a stream↔tile transpose:
#   * stream dimensions (4,4) become the tile‑row dimension (16)
#   * the tile‑row dimension (64) becomes the stream dimension (seq_len)
# The implementation builds a zero‑filled destination tensor of shape
# (seq_len,)×tile (num_heads, head_dim) and then writes each head’s rows
# into it at the proper offset using `binary_set_offset` + `binary_row_wise_append`.
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Vh across the second KV‑head dimension and compute the weighted sum.
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)
    weighted = binary_matmul(attn_weights, Vh_exp)          # stream (4,4) tile (64,32)

    # 2. Allocate a zero‑filled output tensor of shape (seq_len,) × tile (num_heads, head_dim).
    #    Re‑use the shape of `weighted` to create the zeros.
    ones = binary_is_equal(weighted, weighted)               # all 1.0, same shape as `weighted`
    zeros = unary_sub_imm(ones, 1.0)                         # all 0.0, same shape
    zeros_flat = flatten(zeros, min_rank=0, max_rank=1)     # stream (16,) tile (64,32)
    zeros_tile = accum_retile_row(zeros_flat, rank=1)       # tile (1024,32) – no stream
    zeros_tile = promote(zeros_tile, rank=0)                # stream (1,) tile (1024,32)
    num_heads = attn_weights.shape[0] * attn_weights.shape[1]   # 4 × 4 = 16
    result = retile_streamify(zeros_tile,
                              chunk=num_heads,
                              split_row=True)        # stream (64,) tile (16,32)

    # 3. Split the weighted tensor into per‑head streams.
    merged = flatten(weighted, min_rank=0, max_rank=1)      # stream (16,) tile (64,32)
    head_streams = parallelize(merged, num_heads)          # list of 16 tensors,
                                                             # each shape (1,) tile (64,32)

    # 4. For each head, turn its tile rows into a stream and write them into `result`
    #    at the row offset equal to the head index.
    for h_idx, head in enumerate(head_streams):
        # a) Convert the head’s tile (64 × 32) into a stream of rows:
        head_rows = retile_streamify(head,
                                     chunk=1,
                                     split_row=True)          # shape (64,) tile (1,32)

        # b) Build a constant‑offset tensor of shape (seq_len,) × tile (1,1) whose
        #    value equals the current head index.
        offset_scalar = torch.tensor([float(h_idx)], dtype=torch.float32)
        offset_tile   = metadata_gen(offset_scalar)           # stream (1,) tile (1,1)
        offsets = expand_ref(offset_tile,
                             result,
                             expand_rank=1)               # stream (64,) tile (1,1)

        # c) Write the rows into the destination at the computed offsets.
        result = binary_row_wise_append(
                    binary_set_offset(result, offsets),
                    head_rows)

    # `result` now has stream shape (64,) and tile shape (16,32),
    # i.e. the required vanilla shape (64, 16, 32).
    return result