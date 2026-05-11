# ----------------------------------------------------------------------
# Implementation notes
# ----------------------------------------------------------------------
# Q, K, V are on‑chip streams:
#   Q : (1, 64) × tile (16, 32)   # seq_len=64, 16 heads, head_dim=32
#   K : (1, 64) × tile ( 4, 32)   # seq_len=64, 4 KV‑heads, head_dim=32
#   V : (1, 64) × tile ( 4, 32)
#
# The child blackbox `attention_compute` expects *vanilla* inputs:
#   Qh : (4, 4, 64, 32)   # (kv_heads, heads_per_kv, seq_len, head_dim)
#   Kh : (4, 1, 64, 32)
#   Vh : (4, 1, 64, 32)
#
# We first reshape the on‑chip streams into those vanilla layouts using only
# DSL primitives (`retile_streamify`, `reshape_stream`, `accum_retile_row`).
# After the core attention, we invert the transformation to obtain the
# required parent output shape (1, 64, 16, 32).
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # -------------------- Q -> (kv, heads_per_kv, seq) --------------------
    Qh = retile_streamify(Q, chunk=1, split_row=True)                # (1,1024)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=256, rank=0)                  # (1,4,256)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=64, rank=0)                   # (1,4,4,64)×tile(1,32)
    Qh = accum_retile_row(Qh, rank=1)                               # (1,4,4)×tile(64,32)

    # -------------------- K -> (kv, 1, seq) -----------------------------
    Kh = retile_streamify(K, chunk=1, split_row=True)                # (1,256)×tile(1,32)
    Kh = reshape_stream(Kh, chunk_size=64, rank=0)                   # (1,4,64)×tile(1,32)
    Kh = accum_retile_row(Kh, rank=1)                               # (1,4)×tile(64,32)

    # -------------------- V -> (kv, 1, seq) -----------------------------
    Vh = retile_streamify(V, chunk=1, split_row=True)                # (1,256)×tile(1,32)
    Vh = reshape_stream(Vh, chunk_size=64, rank=0)                   # (1,4,64)×tile(1,32)
    Vh = accum_retile_row(Vh, rank=1)                               # (1,4)×tile(64,32)

    # -------------------- Core attention (blackbox) --------------------
    # The child expects its vanilla output shape.
    attn = attention_compute(
        Qh, Kh, Vh,
        out_shapes=((4, 4, 64, 32),),   # vanilla shape expected by the stub
        out_perms=(None,),
    )

    # -------------------- Convert back to parent layout (1,64,16,32) --------------------
    attn = retile_streamify(attn, chunk=1, split_row=True)          # (1,4,4,64)×tile(1,32)
    attn = flatten(attn, min_rank=0, max_rank=1)                   # (1,4,256)×tile(1,32)
    attn = flatten(attn, min_rank=0, max_rank=1)                   # (1,1024)×tile(1,32)
    attn = reshape_stream(attn, chunk_size=16, rank=0)             # (1,64,16)×tile(1,32)
    attn = accum_retile_row(attn, rank=1)                          # (1,64)×tile(16,32)

    return attn