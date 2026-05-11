# The attention core can be expressed entirely with the `attention_compute`
# child.  The only work required is to reshape the on‑chip inputs Q, K,
# V into the GQA layout expected by the child, invoke it, and then undo
# the reshapes to produce a stream of shape (1, 64, 16, 32) as required
# by the contract.  All reshapes use the DSL primitives; no raw tensor
# methods appear.
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # ---------- Q → Qh ----------
    #  Q: (1, 64, 16, 32)  (stream: 1,64)  tile: (16,32)
    #  1) pull the tile‑row dimension into the stream
    Qr = retile_streamify(Q, chunk=1, split_row=True)           # (1,1024,1,32)
    #  2) split the stream dimension 1024 into (16,64)
    Qs = reshape_stream(Qr, chunk_size=64, rank=0)              # (1,16,64,1,32)
    #  3) split the 16 into (4,4) → (Hkv, qpkv)
    Qt = reshape_stream(Qs, chunk_size=4, rank=1)               # (1,4,4,64,1,32)
    #  4) drop the leading dummy stream dim (size 1) so we have
    #     (Hkv, qpkv, S) as a stream
    Qh = flatten(Qt, min_rank=2, max_rank=3)                    # (4,4,64,1,32)

    # ---------- K → Kh ----------
    #  K: (1, 64, 4, 32)
    Kr = retile_streamify(K, chunk=1, split_row=True)           # (1,256,1,32)
    #  split 256 → (4,64)  (Hkv, S)
    Ks = reshape_stream(Kr, chunk_size=64, rank=0)              # (1,4,64,1,32)
    #  further split S (64) into (1,64) to create the required singleton
    #  KV‑head dimension
    Kt = reshape_stream(Ks, chunk_size=64, rank=0)              # (1,4,1,64,1,32)
    #  drop the leading dummy, yielding (Hkv, 1, S)
    Kh = flatten(Kt, min_rank=2, max_rank=3)                    # (4,1,64,1,32)

    # ---------- V → Vh ----------
    Vr = retile_streamify(V, chunk=1, split_row=True)           # (1,256,1,32)
    Vs = reshape_stream(Vr, chunk_size=64, rank=0)              # (1,4,64,1,32)
    Vt = reshape_stream(Vs, chunk_size=64, rank=0)              # (1,4,1,64,1,32)
    Vh = flatten(Vt, min_rank=2, max_rank=3)                    # (4,1,64,1,32)

    # ---------- Core attention ----------
    #  The child expects streams whose tile shape is (1,32); we request a
    #  matching stream layout for its vanilla output (4,4,64,32).
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 1, 32),),   # stream shape of the vanilla output
        out_perms=None,
    )

    # ---------- Inverse transforms to (1, 64, 16, 32) ----------
    #  1) merge the two head dimensions back into one (16, S)
    a1 = flatten(attn, min_rank=1, max_rank=2)                # (16,64,1,32)
    #  2) add a leading dummy stream dimension
    a2 = promote_outer(a1)                                    # (1,16,64,1,32)
    #  3) merge the (16, 64) stream dims into a single flat dimension
    a3 = flatten(a2, min_rank=0, max_rank=1)                  # (1,1024,1,32)
    #  4) split that flat dimension into (64, 16) → (seq_len, tile_rows)
    a4 = reshape_stream(a3, chunk_size=16, rank=0)            # (1,64,16,1,32)
    #  5) fold the tile‑row size (16) into the tile dimensions
    out = accum_retile_row(a4, rank=1)                        # (1,64,16,32)

    # Verify that the produced shape matches the contract
    assert out.shape == out_shapes[0], (
        f"attention_core: output shape {out.shape} does not match expected {out_shapes[0]}"
    )
    return out