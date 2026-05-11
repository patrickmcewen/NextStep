# The attention node simply chains the two child blackboxes that implement the
# model's QKV projection and the attention computation.  The inputs Q, K, V are
# already on‑chip (they are StepTensors produced by a sibling DSL op), so they
# can be fed directly to `compute_qkv`.  We provide the exact tile‑stream shapes
# that the child expects – these are the vanilla shapes of its three outputs,
# each expressed as a stream shape with the last two dimensions being the tile
# size.  The resulting Qh, Kh, Vh tensors are then passed to
# `attention_compute`, propagating the caller‑provided `out_shapes` and
# `out_perms` so the final output conforms to the contract.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # Qh, Kh, Vh = compute_qkv(Q, K, V)
    Qh, Kh, Vh = compute_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),   # Qh: (num_kv_heads, heads_per_kv, seq_len, head_dim)
            (4, 1, 64, 32),   # Kh: (num_kv_heads, 1, seq_len, head_dim)
            (4, 1, 64, 32),   # Vh: (num_kv_heads, 1, seq_len, head_dim)
        ),
        out_perms=None,
    )
    # attn = attention_compute(Qh, Kh, Vh)
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn