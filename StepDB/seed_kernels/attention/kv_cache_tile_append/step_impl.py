"""STeP implementation: tiled KV cache last-tile append.

Mirrors `step_tl/end_to_end/attention/flashattn.py:84-434`
(`build_flashattn_graph`, Stages 1-6) with the attention compute and
output store dropped: load the per-(batch, kv_head) tiled K/V cache,
partition off the last tile, set the slot at `offset` to the new
key/value via SetOffset + RowWiseAppend, FlatReassemble back into a
full cache stream, and RandomOffChipStore the patched last tile.
The two `append_*` streams are also fed into a StaticReassemble +
OffChipStore so a single tensor flows out of the graph for
correctness checking against `reference.py:compute_gold`.

Cache layout note. The simulator's RandomOffChipLoad treats the
underlying as 2-D `[R, head_dim]` and computes tile addresses as
`idx * cache_row_offset_tiled + col`. To give each (batch, kv_head)
slot its own contiguous block we permute the cache from
`[B, max_seq_len, num_kv_heads, head_dim]` to
`[B, num_kv_heads, max_seq_len, head_dim]` and flatten the leading
three dims; per-(b, h) metadata streams (`idx_flat`, `seq_len_flat`,
`offset_flat`) replace the
`RepeatStatic(*, repeat_factor=num_kv_heads)` blocks in flashattn.py.
This makes the tile address `(b * num_kv_heads + h) * max_seq_len_tiles
+ col` map cleanly to row `(b*num_kv_heads + h) * max_seq_len + col *
tile_N` of the flat underlying.

Output stream order. `StaticReassemble([append_k, append_v],
merge_rank=0)` interleaves at the (b, h) granularity: tile 2k =
K_tile[b*H + h], tile 2k+1 = V_tile[b*H + h]. After OffChipStore +
`_untile_store` this matches the reference's row-major
`[2 * B * num_kv_heads * tile_N, head_dim]` layout.

Partition / FilterLastTile indexing. The Rust simulator's
`FilterLastTile` (in `step-perf/src/operator/flatmap.rs:522-593`)
emits multihot `[true, false]` for the last tile and `[false, true]`
for non-last. `FlatPartition` then routes column 0 = last and
column 1 = non-last, so `(partition, 0)` is the last-tile stream
(B*H tiles) and `(partition, 1)` is the rest. The Python functional
simulator inverts this convention; this kernel is built for the
Rust simulator (`evaluate.py`).
"""

SEED = 42


def build_graph(dims):
    batch_size = dims["batch_size"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    max_seq_len_tiles = dims["max_seq_len_tiles"]
    tile_N = dims["tile_N"]
    PAR_DISPATCH = dims.get("par_dispatch", 4)
    cache_row_offset_tiled = max_seq_len_tiles
    max_seq_len = max_seq_len_tiles * tile_N
    BH = batch_size * num_kv_heads

    # RNG order MUST match seed_kernels/kv_cache_tile_append/reference.py:get_inputs
    # and precompute._precompute_kv_cache_tile_append exactly.
    torch.manual_seed(SEED)
    key     = torch.randn(batch_size, num_kv_heads, head_dim)
    value   = torch.randn(batch_size, num_kv_heads, head_dim)
    k_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)
    v_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)
    idx_orig      = torch.arange(batch_size, dtype=torch.int64)
    seq_len_tiled = torch.randint(low=1, high=max_seq_len_tiles + 1,
                                  size=(batch_size,), dtype=torch.int64)
    offset        = torch.randint(low=0, high=tile_N,
                                  size=(batch_size,), dtype=torch.int64)

    # Per-(b, h) flattened tensors. Cache permute makes each (b, h) slot a
    # contiguous max_seq_len-row block in the flat underlying.
    k_cache_flat = k_cache.permute(0, 2, 1, 3).contiguous().flatten(0, 2)
    v_cache_flat = v_cache.permute(0, 2, 1, 3).contiguous().flatten(0, 2)
    key_flat   = key.reshape(BH, head_dim)
    value_flat = value.reshape(BH, head_dim)
    idx_flat     = torch.arange(BH, dtype=torch.int64).to(torch.uint64)
    seq_len_flat = seq_len_tiled.repeat_interleave(num_kv_heads).to(torch.uint64)
    offset_flat  = offset.repeat_interleave(num_kv_heads).to(torch.uint64)

    step_graph = Graph()

    # ---------- Stage 1: load streams + metadata ----------
    load_idx     = MetadataGen(tensor=idx_flat)
    load_seq_len = MetadataGen(tensor=seq_len_flat)
    load_offset  = MetadataGen(tensor=offset_flat)

    # idx feeds 2 CacheReadAddrGen + 2 CacheWriteAddrGen = 4 consumers.
    # seq_len feeds 2 CacheReadAddrGen + 2 FilterLastTile + 2 CacheWriteAddrGen = 6.
    # offset feeds 2 SetOffset BinaryMap = 2.
    bcast_idx     = Broadcast(graph=step_graph, input=load_idx,     num_consumers=4)
    bcast_seq_len = Broadcast(graph=step_graph, input=load_seq_len, num_consumers=6)
    bcast_offset  = Broadcast(graph=step_graph, input=load_offset,  num_consumers=2)

    # key / value: [BH] tile [1, head_dim]
    load_key = LinearOffChipLoad(
        underlying=key_flat, stride=(1,),
        out_shape_tiled=(BH,),
        tile_row=1, tile_col=head_dim, par_dispatch=PAR_DISPATCH,
    )
    load_value = LinearOffChipLoad(
        underlying=value_flat, stride=(1,),
        out_shape_tiled=(BH,),
        tile_row=1, tile_col=head_dim, par_dispatch=PAR_DISPATCH,
    )

    def _build_branch(cache_underlying, load_xv, idx_port, seq_len_ports, offset_port):
        """Stages 1-3 (or 4-6) of flashattn.py: load -> partition off last tile
        -> SetOffset+RowWiseAppend -> FlatReassemble + writeback. Returns
        `append_xv` (the patched-last-tile stream) so the caller can hook the
        downstream writeback and the StaticReassemble that builds our output.
        """
        # ----- load cache tiles -----
        load_addr = CacheReadAddrGen(
            graph=step_graph,
            idx=(bcast_idx, idx_port),
            seq_len=(bcast_seq_len, seq_len_ports[0]),
            row_offset=cache_row_offset_tiled,
        )
        cache_load = RandomOffChipLoad(
            graph=step_graph,
            underlying=cache_underlying,
            raddr=load_addr,
            tile_row=tile_N, tile_col=head_dim,
            base_addr_byte=0, par_dispatch=PAR_DISPATCH,
        )

        # ----- partition off the last tile per (b, h) -----
        control = FilterLastTile(
            graph=step_graph,
            input=(bcast_seq_len, seq_len_ports[1]),
        )
        # Force the dynamic dim of the multihot control to match cache_load's
        # DynN, otherwise FlatPartition sees two distinct symbolic dims and
        # the Rust simulator's channel routing closes prematurely (mirrors
        # flashattn.py:127-130).
        control.stream.shape = (
            control.stream.shape[:-1] + (cache_load.stream.shape[-1],)
        )
        partition = FlatPartition(
            graph=step_graph,
            input=cache_load,
            control=control,
            partition_rank=0,
            switch_cycles=[1, 1],
            write_back_mu=False,
            num_consumers=2,
        )
        # Rust FilterLastTile routes last tiles to col 0 and rest to col 1
        # (see `step-perf/src/operator/flatmap.rs:539-580`). The "last"
        # partition has exactly one tile per (b, h); promote its output
        # stream shape from a symbolic DynDim to the static `BH` count so
        # downstream StaticReassemble + OffChipStore can serialize it.
        partition._stream[0].shape = (BH,)

        # Sink the non-last partition stream (Stages 4/6 in flashattn.py
        # FlatReassemble it into the rebuilt cache; this benchmark stops
        # at the modification, so just drain it).
        ConsumerContext(graph=step_graph, input=(partition, 1))

        # ----- SetOffset on the last-tile slot -----
        # `(partition, 0)` is a 1-D stream of [BH] tiles; metadata is
        # [1, BH, 1, 1] from MetadataGen + Broadcast, so flatten to align
        # outer dims for the BinaryMap.
        offset_flat_stream = Flatten(
            graph=step_graph, input=(bcast_offset, offset_port),
            min_rank=0, max_rank=1,
        )
        set_off = BinaryMap(
            graph=step_graph,
            in1=(partition, 0),
            in2=offset_flat_stream,
            fn=SetOffset(),
            write_back_mu=False, compute_bw=0,
        )
        # ----- write key/value into the offset slot -----
        # load_xv is [1, BH, 1, head_dim]; flatten to [BH, 1, head_dim].
        xv_flat = Flatten(
            graph=step_graph, input=load_xv,
            min_rank=0, max_rank=1,
        )
        appended = BinaryMap(
            graph=step_graph,
            in1=set_off,
            in2=xv_flat,
            fn=RowWiseAppend(),
            write_back_mu=False, compute_bw=0,
        )
        # Broadcast: one path goes to the cache write-back (Stage 3 / 6 in
        # flashattn.py), the other to the StaticReassemble that builds our
        # gold output.
        appended_bcast = Broadcast(
            graph=step_graph, input=appended, num_consumers=2,
        )

        # ----- RandomOffChipStore: write the patched last tile back -----
        # Address generator: for each (b, h), the destination tile index is
        # idx * cache_row_offset_tiled + (seq_len - 1).
        write_addr = BinaryMap(
            graph=step_graph,
            in1=(bcast_idx, idx_port + 2),
            in2=(bcast_seq_len, seq_len_ports[2]),
            fn=CacheWriteAddrGen(row_offset=cache_row_offset_tiled),
            write_back_mu=False, compute_bw=0,
        )
        RandomOffChipStore(
            graph=step_graph,
            underlying=cache_underlying,
            waddr=write_addr,
            wdata=(appended_bcast, 0),
            tile_row=tile_N, tile_col=head_dim,
            base_addr_byte=0,
            buffer_depth=1,
            par_dispatch=PAR_DISPATCH,
            mock_bf16=False,
            has_done_stream=False,
        )

        return (appended_bcast, 1)

    append_k = _build_branch(
        cache_underlying=k_cache_flat, load_xv=load_key,
        idx_port=0, seq_len_ports=[0, 1, 2], offset_port=0,
    )
    append_v = _build_branch(
        cache_underlying=v_cache_flat, load_xv=load_value,
        idx_port=1, seq_len_ports=[3, 4, 5], offset_port=1,
    )

    # ---------- Output: stack K and V appended tiles, store ----------
    # StaticReassemble at merge_rank=0 interleaves the two streams: tile 2k
    # is K_tile[k = b*H + h] and tile 2k+1 is V_tile[k]. _untile_store then
    # reshapes to [2 * BH * tile_N, head_dim].
    combined = StaticReassemble(
        graph=step_graph,
        inputs=[append_k, append_v],
        merge_rank=0,
        switch_cycles=[1, 1],
    )
    # OffChipStore strips the leading stream dim to derive `tensor_shape_tiled`,
    # so a 1-D input would leave it empty and break the Rust simulator. Add a
    # leading singleton via Promote.
    combined_promoted = Promote(
        graph=step_graph,
        input=combined,
        promote_rank=len(combined.stream.shape),
    )

    output = OffChipStore(
        graph=step_graph,
        input=combined_promoted,
        par_dispatch=PAR_DISPATCH,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
