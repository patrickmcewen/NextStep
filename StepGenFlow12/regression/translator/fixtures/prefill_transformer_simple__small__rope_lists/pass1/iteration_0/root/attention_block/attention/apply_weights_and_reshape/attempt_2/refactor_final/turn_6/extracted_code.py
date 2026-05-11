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