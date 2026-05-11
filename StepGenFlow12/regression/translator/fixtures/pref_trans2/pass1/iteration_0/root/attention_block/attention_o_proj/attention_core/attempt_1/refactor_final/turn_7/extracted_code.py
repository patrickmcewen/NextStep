# ----------------------------------------------------------------------
# Implementation notes
# ----------------------------------------------------------------------
# Q, K, V are on‑chip streams:
#   Q : (1, 64) × tile (16, 32)   # seq_len=64, 16 heads, head_dim=32
#   K : (1, 64) × tile ( 4, 32)   # seq_len=64, 4 KV‑heads, head_dim=32
#   V : (1, 64) × tile ( 4, 32)
#
# The child `attention_compute` expects its inputs in vanilla shape:
#   Qh : (4, 4, 64, 32)   # (kv_heads, heads_per_kv, seq_len, head_dim)
#   Kh : (4, 1, 64, 32)
#   Vh : (4, 1, 64, 32)
#
# We build those tensors with retile/reshape/retile operations that exactly
# mimic the PyTorch view‑permute sequence.  After the core attention
# computation we must undo the permutation and reshape back to the
# required output layout (1, 64, 16, 32).  This is done with a
# retile → flatten → reshape → accum_retile_row → promote chain that swaps
# the sequence dimension back into the stream and makes the head dimension
# a tile dimension.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # -------------------- Q -> (kv, heads_per_kv, seq) --------------------
    Qh = retile_streamify(Q, chunk=1, split_row=True)                # (1,1024)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=256, rank=0)                  # (1,4,256)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=64, rank=0)                   # (1,4,4,64)×tile(1,32)
    Qh = accum_retile_row(Qh, rank=1)                                # (1,4,4)×tile(64,32)

    # -------------------- K -> (kv, 1, seq) -------------------------------
    Kh = retile_streamify(K, chunk=1, split_row=True)                # (1,256)×tile(1,32)
    Kh = reshape_stream(Kh, chunk_size=64, rank=0)                   # (1,4,64)×tile(1,32)
    Kh = accum_retile_row(Kh, rank=1)                                # (1,4)×tile(64,32)

    # -------------------- V -> (kv, 1, seq) -------------------------------
    Vh = retile_streamify(V, chunk=1, split_row=True)                # (1,256)×tile(1,32)
    Vh = reshape_stream(Vh, chunk_size=64, rank=0)                   # (1,4,64)×tile(1,32)
    Vh = accum_retile_row(Vh, rank=1)                                # (1,4)×tile(64,32)

    # -------------------- Core attention (blackbox) ----------------------
    # The child expects its vanilla output shape (4,4,64,32).
    attn = attention_compute(
        Qh, Kh, Vh,
        out_shapes=((4, 4, 64, 32),),
        out_perms=(None,),
    )

    # -------------------- Convert back to (1, 64, 16, 32) ----------------
    # Move the sequence dimension (currently a tile row) into the stream.
    attn = retile_streamify(attn, chunk=1, split_row=True)           # (1,4,256)×tile(1,32)

    # Collapse all stream dimensions into one.
    attn = flatten(attn, min_rank=0, max_rank=2)                     # (1024,1,32)

    # Split that single stream dimension into (seq=64, heads_total=16).
    attn = reshape_stream(attn, chunk_size=16, rank=0)               # (64,16,1,32)

    # Fold the heads_total stream dimension into the tile rows.
    attn = accum_retile_row(attn, rank=1)                            # (64,16,32)

    # Add a leading singleton stream dimension to match the contract.
    attn = promote(attn, rank=1)                                     # (1,64,16,32)

    return attn