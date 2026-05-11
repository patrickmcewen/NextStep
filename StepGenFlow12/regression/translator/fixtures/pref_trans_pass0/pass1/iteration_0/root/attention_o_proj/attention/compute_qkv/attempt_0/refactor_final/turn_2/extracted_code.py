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