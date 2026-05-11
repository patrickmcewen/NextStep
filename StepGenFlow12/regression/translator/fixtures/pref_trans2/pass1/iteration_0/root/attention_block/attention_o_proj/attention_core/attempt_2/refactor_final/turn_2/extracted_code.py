# The inputs Q, K, V are already on‑chip streams:
#   Q : (1, 64)×tile(16, 32)   – (batch, seq_len) stream, heads as tile‑rows
#   K, V : (1, 64)×tile(4, 32) – (batch, seq_len) stream, kv‑heads as tile‑rows
# We must reinterpret them as the GQA layout expected by `attention_compute`:
#   Qh : (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32) → stream(4,4)×tile(64,32)
#   Kh, Vh : (kv_head=4, 1, seq_len=64, head_dim=32) → stream(4,1)×tile(64,32)
# This is done with `restream`, whose `stride` encodes the mapping from the
# original buffer grid (batch, seq_len, heads, head_dim) to the target
# stream dimensions.
# After the core attention we invert the transformation: merge the two head‑
# related stream dimensions, promote the leading batch dim, then split the
# tile‑row dimension (seq_len) back into a stream dimension, restoring the
# original layout (1,64)×tile(16,32).

def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Q → (kv_head=4, q_per_kv=4, seq_len=64, head_dim=32)
    Qh = restream(
        Q,
        stride=(128, 32, 512, 1),          # kv_head stride, q_per_kv stride, seq_len stride, head_dim stride
        out_shape_tiled=(4, 4, 64, 32),
    )

    # K → (kv_head=4, 1, seq_len=64, head_dim=32)
    Kh = restream(
        K,
        stride=(32, 0, 128, 1),             # kv_head stride, broadcast (0), seq_len stride, head_dim stride
        out_shape_tiled=(4, 1, 64, 32),
    )

    # V → (kv_head=4, 1, seq_len=64, head_dim=32)
    Vh = restream(
        V,
        stride=(32, 0, 128, 1),
        out_shape_tiled=(4, 1, 64, 32),
    )

    # Core attention (blackbox)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=((4, 4, 64, 32),),
        out_perms=(None,),
    )

    # Inverse transformation back to (batch=1, seq_len=64)×tile(16,32)
    attn_flat = flatten(attn, min_rank=0, max_rank=1)   # merge kv_head and q_per_kv → stream(16)
    attn_outer = promote_outer(attn_flat)                # add leading batch dim → stream(1,16)
    out = retile_streamify(
        attn_outer,
        chunk=16,            # heads per token
        split_row=True,
    )                         # stream(1,64)×tile(16,32)

    return out