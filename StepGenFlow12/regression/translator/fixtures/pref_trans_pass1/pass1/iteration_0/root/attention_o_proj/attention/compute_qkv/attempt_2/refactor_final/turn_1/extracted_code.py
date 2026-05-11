def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    """
    Produce the three tiled Q‑, K‑ and V‑streams required by the attention
    kernel.

    The raw inputs already arrive as on‑chip streams:
        Q : stream‑shape (64,)   tile (16, 32)   ← heads = 16
        K,V : stream‑shape (64,) tile (4, 32)   ← kv‑heads = 4

    Desired outputs (stream‑shape + tile):
        Qh : (4, 4, 64)  tile (64, 32)   ← (kv, q‑per‑kv, seq)
        Kh,Vh : (4, 1, 64) tile (64, 32)  ← (kv, 1, seq)

    The transformation consists of three independent pipelines that only
    use DSL primitives (no raw tensor arithmetic).

    Q‑pipeline
    ----------
    1. promote – add a singleton innermost stream dimension so that the
       subsequent retile can split the head‑dimension (tile rows) onto a
       **separate** stream axis.
    2. retile_streamify – move the 16 head rows into a new stream axis,
       leaving tile‑rows = 1.
    3. reshape_stream – split the new “heads” stream axis (size 16) into
       (kv = 4, q_per_kv = 4).
    4. bufferize – absorb the three remaining stream axes (seq, kv, q) into
       a buffer grid.
    5. streamify – read the buffer with out‑shape (kv, q, seq) and stride
       that re‑orders the axes to (kv, q, seq).  The stride mirrors the
       row‑major layout of the buffer: (kv‑stride = q, q‑stride = 1,
       seq‑stride = kv × q).
    6. accum_retile_row – pull the innermost stream axis (seq) into the
       tile rows, yielding the final tile‑shape (64, 32) and stream shape
       (4, 4).

    K/V‑pipeline (identical for both tensors)
    -----------------------------------------
    1. promote – create a singleton stream axis.
    2. retile_streamify – move the 4 kv‑head rows onto that axis,
       obtaining stream shape (seq, kv) and tile‑rows = 1.
    3. bufferize – absorb both stream axes into a buffer grid.
    4. streamify – emit stream shape (kv, seq) with stride (1, kv)
       (kv becomes the outer stream dimension).
    5. promote – insert a singleton stream axis **between** kv and seq
       (rank = 1) giving (kv, 1, seq).
    6. accum_retile_row – merge the innermost axis (seq) into the tile
       rows, producing the required (kv, 1) stream shape and tile‑rows = 64.

    All intermediate tensors retain the original element type
    (Float32) and the appropriate dynamic‑mask information.
    """
    # ----------------------- Q -------------------------------------------------
    q = promote(Q, rank=0)                              # (64,1)  tile(16,32)
    q = retile_streamify(q, chunk=1, split_row=True)    # (64,16) tile(1,32)
    q = reshape_stream(q, chunk_size=4, rank=0)         # (64,4,4) tile(1,32)
    q = bufferize(q, rank=3)                            # buffer shape (64,4,4)
    q = streamify(
        q,
        stride=[4, 1, 16],                               # kv stride, q stride, seq stride
        out_shape_tiled=(4, 4, 64),
    )                                                    # (4,4,64) tile(1,32)
    Qh = accum_retile_row(q, rank=1)                    # (4,4) tile(64,32)

    # ----------------------- K -------------------------------------------------
    k = promote(K, rank=0)                              # (64,1)  tile(4,32)
    k = retile_streamify(k, chunk=1, split_row=True)    # (64,4) tile(1,32)
    k = bufferize(k, rank=2)                            # buffer shape (64,4)
    k = streamify(
        k,
        stride=[1, 4],                                   # kv stride, seq stride
        out_shape_tiled=(4, 64),
    )                                                    # (4,64) tile(1,32)
    k = promote(k, rank=1)                              # (4,1,64) tile(1,32)
    Kh = accum_retile_row(k, rank=1)                    # (4,1) tile(64,32)

    # ----------------------- V -------------------------------------------------
    v = promote(V, rank=0)                              # (64,1)  tile(4,32)
    v = retile_streamify(v, chunk=1, split_row=True)    # (64,4) tile(1,32)
    v = bufferize(v, rank=2)                            # buffer shape (64,4)
    v = streamify(
        v,
        stride=[1, 4],
        out_shape_tiled=(4, 64),
    )                                                    # (4,64) tile(1,32)
    v = promote(v, rank=1)                              # (4,1,64) tile(1,32)
    Vh = accum_retile_row(v, rank=1)                    # (4,1) tile(64,32)

    return Qh, Kh, Vh