"""STeP implementation: GQA single-token decode with a tiled KV cache and
in-place append of the new key/value before attention.

Mirrors the flashattn graph in
step_tl/end_to_end/attention/flashattn.py (build_flashattn_graph), wrapped
by step_tl/end_to_end/attention/static_parallel.py (build_static_par).
The reference is at seed_kernels/gqa_tiled_decode/reference.py.

Pipeline (per batch element b, for each kv-head h; everything below runs
as one fused dataflow graph):

  -------- Stage 1: load streams (one tile per b) --------
  Load Q          -> (1, B, num_kv_heads) / (query_per_kvhead, head_dim)
  Load K_new      -> (1, B, num_kv_heads) / (1, head_dim)
  Load V_new      -> (1, B, num_kv_heads) / (1, head_dim)
  MetadataGen idx           -> (1, B) / (1, 1)
  MetadataGen seq_len_tiled -> (1, B) / (1, 1)
  MetadataGen offset        -> (1, B) / (1, 1)

  -------- Stage 2: K-cache read + last-tile patch --------
  CacheReadAddrGen(idx, seq_len_tiled, row_offset=cache_row_offset_tiled)
  RandomOffChipLoad k_cache -> (B, num_kv_heads, DynN) / (tile_N, head_dim)
  FilterLastTile(seq_len_tiled)   -> multihot control [B, num_kv_heads, DynN]
  FlatPartition by control        -> (last_tile_stream, rest_stream)
  SetOffset(offset)               -> last_tile slot index for the new K row
  RowWiseAppend(K_new)            -> overwrite slot `offset` with K_new
  FlatReassemble                  -> patched K cache tiles
  Flatten                         -> formatted_k_cache stream

  -------- Stage 3: K write-back --------
  CacheWriteAddrGen(idx, seq_len_tiled)
  RandomOffChipStore k_cache_underlying  (Broadcast off the patched stream)

  -------- Stage 4: V-cache read + last-tile patch (mirrors Stage 2) --------
  -------- Stage 5: V write-back (mirrors Stage 3) --------

  -------- Stage 6: attention compute --------
  RepeatRef(Q, ref=formatted_k_cache)        # broadcast Q over DynN
  qkt = BinaryMap(Matmul, weight_transposed=True)   # (q_per_kv, tile_N)
  exp = UnaryMap(Exp)
  num   = BinaryMapAccum(Matmul, init=Zero)  # accum exp @ V over DynN
  tile_rowsum = Accum(Add, init=Zero)        # accum exp over DynN tiles
  rowsum      = UnaryMap(RowWiseSum)         # collapse tile_N to 1
  softmax = BinaryMap(Div)                   # num / rowsum

  -------- Stage 7: store output --------
  OffChipStore softmax  (no max-subtract, no 1/sqrt(d) — matches the math
                         in the reference verbatim)

Output layout (chosen to match reference.compute_gold):
  output : [B, num_kv_heads, query_per_kvhead, head_dim]
"""

SEED = 42


def build_graph(dims):
    batch_size = dims["batch_size"]
    num_kv_heads = dims["num_kv_heads"]
    query_per_kvhead = dims["query_per_kvhead"]
    head_dim = dims["head_dim"]
    max_seq_len_tiles = dims["max_seq_len_tiles"]
    tile_N = dims["tile_N"]
    PAR_DISPATCH = dims.get("par_dispatch", 4)
    cache_row_offset_tiled = max_seq_len_tiles  # tiles per batch slot

    max_seq_len = max_seq_len_tiles * tile_N

    # RNG order MUST match seed_kernels/gqa_tiled_decode/reference.py:get_inputs
    # and precompute._precompute_gqa_tiled_decode exactly.
    torch.manual_seed(SEED)
    query   = torch.randn(batch_size, num_kv_heads, query_per_kvhead, head_dim)
    key     = torch.randn(batch_size, num_kv_heads, head_dim)
    value   = torch.randn(batch_size, num_kv_heads, head_dim)
    k_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)
    v_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)
    idx           = torch.arange(batch_size, dtype=torch.int64)
    seq_len_tiled = torch.randint(low=1, high=max_seq_len_tiles + 1,
                                  size=(batch_size,), dtype=torch.int64)
    offset        = torch.randint(low=0, high=tile_N,
                                  size=(batch_size,), dtype=torch.int64)

    step_graph = Graph()

    # ----- Stage 1: load Q/K_new/V_new and metadata streams -----
    # TODO: LinearOffChipLoad for query (tile [query_per_kvhead, head_dim]),
    #       key, value (tile [1, head_dim]); MetadataGen for idx, seq_len_tiled,
    #       offset; Broadcast metadata to the consumers that need it
    #       (idx -> 4, seq_len_tiled -> 6, offset -> 2; see flashattn.py).

    # ----- Stage 2: K-cache read + in-place K_new append at last tile -----
    # TODO: CacheReadAddrGen, RandomOffChipLoad(k_cache_flat),
    #       FilterLastTile -> FlatPartition (last vs rest),
    #       SetOffset + RowWiseAppend(key) on last-tile partition,
    #       FlatReassemble + Flatten -> formatted_k_cache.

    # ----- Stage 3: K-cache write-back of patched last tile -----
    # TODO: CacheWriteAddrGen, RandomOffChipStore on the broadcast of the
    #       appended last-tile stream (channel_dict[...] = cache fifo depth).

    # ----- Stage 4: V-cache read + in-place V_new append at last tile -----
    # TODO: mirror Stage 2 for v_cache and value -> formatted_v_cache.

    # ----- Stage 5: V-cache write-back of patched last tile -----
    # TODO: mirror Stage 3 for v_cache.

    # ----- Stage 6: attention compute (no max-subtract, no sm_scale) -----
    # TODO:
    #   expanded_q = RepeatRef(query, ref=formatted_k_cache)
    #   qkt        = BinaryMap(Matmul(weight_transposed=True), expanded_q,
    #                          formatted_k_cache)         # (q_per_kv, tile_N)
    #   exp        = UnaryMap(Exp(), qkt)
    #   num        = BinaryMapAccum(Matmul(), exp, formatted_v_cache,
    #                               init=Zero(q_per_kv, head_dim), rank=1)
    #   tile_sum   = Accum(Add(), exp,
    #                      init=Zero(tile of (q_per_kv, tile_N)), accum_rank=1)
    #   rowsum     = UnaryMap(RowWiseSum(), tile_sum)     # (q_per_kv, 1)
    #   softmax    = BinaryMap(Div(), num, rowsum)        # write_back_mu=True

    # ----- Stage 7: store output -----
    # Output shape mirrors the reference: [B, num_kv_heads, query_per_kvhead, head_dim]
    # TODO: reassemble softmax into the [B, num_kv_heads] outer layout and
    #       OffChipStore. (StaticReassemble across par_factor lanes if a
    #       parallelized version is built later, mirroring static_parallel.py.)

    output = None  # OffChipStore(graph=step_graph, input=softmax,
                   #              par_dispatch=PAR_DISPATCH, store_file_name="output")

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
