# The black‑box `attention_compute` expects its inputs in the GQA layout:
#   Qh:  (num_kv_heads, query_per_kvhead, seq_len, head_dim)
#   Kh/Vh: (num_kv_heads, 1, seq_len, head_dim)
# The parent supplies Q/K/V as streams with shape
#   Q: (1, seq_len, num_heads, tile_c)   – tile rows hold the 16 heads
#   K/V: (1, seq_len, num_kv_heads, tile_c) – tile rows hold the 4 KV heads
#
# We must rearrange the on‑chip streams without any raw tensor ops.
# The transformation is performed with three DSL primitives:
#   * `retile_streamify`  – makes tile rows part of the stream (splits them out)
#   * `parallelize`       – splits a leading stream dimension into N separate streams
#   * `static_reassemble` – interleaves the N streams back into a single tensor with a
#                           user‑specified stream shape.
#
# This combination lets us transpose the (seq, head) layout needed for GQA.
#
# After the child produces its result we perform the inverse transformation to
# restore the original (seq, head) ordering expected by the parent.
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Q → Qh  (num_kv_heads=4, query_per_kvhead=4, seq_len=64)
    # ------------------------------------------------------------------
    # 1) Pull the 16 heads out of the tile dimension into the stream.
    Q_flat = retile_streamify(Q, chunk=1, split_row=True)          # (1,1024,1,32)

    # 2) Split the (head, seq) stream into 16 separate streams, each
    #    representing one head across the full sequence.
    #    After `flatten` the leading batch dim (size 1) and the head dim (size 16)
    #    are merged, leaving a stream shape (16, 64).
    Q_merge = flatten(Q_flat, min_rank=2, max_rank=3)             # (16,64,1,32)

    # 3) `parallelize` works on the leading stream dimension, which is now 16.
    Q_heads = parallelize(Q_merge, n=16)                         # 16 × (1,64,1,32)

    # 4) Re‑assemble the 16 head streams into the GQA layout.
    #    The target stream shape (4, 4, 64) corresponds to
    #    (num_kv_heads, query_per_kvhead, seq_len).
    Qh = static_reassemble(Q_heads, target_stream_shape=(4, 4, 64))  # (4,4,64,1,32)

    # ------------------------------------------------------------------
    # K → Kh   (num_kv_heads=4)
    # ------------------------------------------------------------------
    K_flat = retile_streamify(K, chunk=1, split_row=True)          # (1,256,1,32)
    K_merge = flatten(K_flat, min_rank=2, max_rank=3)             # (4,64,1,32)
    K_heads = parallelize(K_merge, n=4)                           # 4 × (1,64,1,32)
    Kh = static_reassemble(K_heads, target_stream_shape=(4, 1, 64))  # (4,1,64,1,32)

    # ------------------------------------------------------------------
    # V → Vh   (num_kv_heads=4)
    # ------------------------------------------------------------------
    V_flat = retile_streamify(V, chunk=1, split_row=True)          # (1,256,1,32)
    V_merge = flatten(V_flat, min_rank=2, max_rank=3)             # (4,64,1,32)
    V_heads = parallelize(V_merge, n=4)                           # 4 × (1,64,1,32)
    Vh = static_reassemble(V_heads, target_stream_shape=(4, 1, 64))  # (4,1,64,1,32)

    # ------------------------------------------------------------------
    # Core attention (black‑box)
    # ------------------------------------------------------------------
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )  # vanilla shape (4,4,64,32) → stream (4,4,64,1,32)

    # ------------------------------------------------------------------
    # Inverse transform: back to (1, seq_len=64, num_heads=16, tile_c=32)
    # ------------------------------------------------------------------
    # 1) Merge the KV‑head and query‑per‑KV dimensions (the first two stream
    #    axes) back into a single head axis.
    attn_merge = flatten(attn, min_rank=1, max_rank=2)            # (16,64,1,32)

    # 2) Split the leading head axis (size 16) into 16 separate streams,
    #    each carrying one head across the full sequence.
    attn_parts = parallelize(attn_merge, n=16)                    # 16 × (1,64,1,32)

    # 3) Re‑assemble these 16 streams into the original (seq, head) layout.
    #    The target stream shape (1, 64, 16) restores the parent‑expected ordering.
    out_stream = static_reassemble(
        attn_parts, target_stream_shape=(1, 64, 16)
    )                                                            # (1,64,16,1,32)

    # 4) Move the head dimension from the stream back into the tile rows
    #    (the parent’s original tiling had tile rows = 16).
    out = accum_retile_row(out_stream, rank=1)                    # (1,64,16,32)

    # ------------------------------------------------------------------
    # Verify contract compliance
    # ------------------------------------------------------------------
    assert out.shape == out_shapes[0], (
        f"attention_core: output shape {out.shape} does not match expected "
        f"{out_shapes[0]}"
    )
    return out