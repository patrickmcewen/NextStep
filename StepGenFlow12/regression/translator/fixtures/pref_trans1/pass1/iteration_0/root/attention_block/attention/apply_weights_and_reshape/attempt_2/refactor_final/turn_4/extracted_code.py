# The node receives on‑chip tensors:
#   attn_weights: shape (4, 4, 64, 64) → stream (4,4) tile (64,64)
#   Vh:          shape (4, 1, 64, 32) → stream (4,1) tile (64,32)
# Goal: compute the weighted sum `attn_weights @ Vh` and then reshape the
# result to the vanilla shape (seq_len, num_heads, head_dim) = (64, 16, 32).
# This is equivalent to a transpose that swaps the two head‑related stream
# dimensions with the sequence‑length tile dimension.  The implementation
# builds a zero‑filled destination tensor and writes each head’s rows into
# it at the correct offset using `binary_set_offset` / `binary_row_wise_append`.
def apply_weights_and_reshape(attn_weights, Vh, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------
    # 1. Broadcast Vh across the second KV‑head dimension.
    # ------------------------------------------------------------
    Vh_exp = expand_ref(Vh, attn_weights, expand_rank=1)

    # ------------------------------------------------------------
    # 2. Weighted sum over values (batched matmul).
    # ------------------------------------------------------------
    weighted = binary_matmul(attn_weights, Vh_exp)        # stream (4,4) tile (64,32)

    # ------------------------------------------------------------
    # 3. Merge the two head‑related stream dimensions into one.
    # ------------------------------------------------------------
    merged = flatten(weighted, min_rank=0, max_rank=1)    # stream (16,) tile (64,32)

    # ------------------------------------------------------------
    # 4. Prepare a zero‑filled destination tensor of shape
    #    (seq_len,) × tile (num_heads, head_dim) → (64,) × (16,32).
    # ------------------------------------------------------------
    #   a) Create a tensor of zeros with the same shape as `merged`.
    ones = binary_is_equal(weighted, weighted)            # all ones, shape (16,64,32)
    zeros = unary_sub_imm(ones, 1.0)                       # all zeros, same shape
    zeros_flat = flatten(zeros, min_rank=0, max_rank=1)   # stream (16,) tile (64,32)
    zeros_tile = accum_retile_row(zeros_flat, rank=1)     # tile (1024,32) – no stream

    #   b) Split the combined tile‑row dimension (1024 = seq_len × num_heads)
    #      back into a stream (seq_len) and tile‑row (num_heads).
    num_heads = attn_weights.shape[0] * attn_weights.shape[1]   # 4*4 = 16
    result = reshape_stream(zeros_tile,
                            chunk_size=num_heads,
                            add_outer_dim=True)           # stream (64,) tile (16,32)

    # ------------------------------------------------------------
    # 5. Split `merged` into per‑head tensors.
    #    `parallelize` partitions a stream of length 16 into 16 sub‑streams,
    #    each holding a single head.
    # ------------------------------------------------------------
    head_streams = parallelize(merged, num_heads)   # list of 16 tensors,
                                                     # each shape (1,) tile (64,32)

    # ------------------------------------------------------------
    # 6. For each head, turn its tile‑rows into a stream and write them
    #    into `result` at the appropriate row offset.
    # ------------------------------------------------------------
    for h_idx, head in enumerate(head_streams):
        # a) Convert the head’s tile (64 rows) into a stream of rows:
        #    shape → (64,) tile (1, head_dim)
        head_rows = retile_streamify(head,
                                     chunk=1,
                                     split_row=True)       # (64,)×(1,32)

        # b) Build an offset tensor of shape (seq_len, num_heads, 1, 1)
        #    where every entry equals the current head index `h_idx`.
        #    We create a scalar int tensor from the Python int (allowed) and
        #    promote it to the full shape via `expand_ref`.
        offset_scalar = torch.tensor([h_idx])               # int tensor, shape (1,)
        offset_tile   = metadata_gen(offset_scalar)         # (1,1,1,1) → stream (1,) tile (1,1)
        offset_full   = expand_ref(offset_tile,
                                   result,
                                   expand_rank=2)          # (64,16,1,1)

        # c) Write the rows into the destination at the computed offsets.
        result = binary_row_wise_append(
                    binary_set_offset(result, offset_full),
                    head_rows)

    # `result` now has stream shape (64,) and tile shape (16,32),
    # i.e. vanilla shape (64, 16, 32) as required.
    return result