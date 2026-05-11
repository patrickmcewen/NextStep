# Implements the attention node by sequencing the three provided child
# blackboxes.  Q, K, V are already on‑chip streams with shape
# (seq_len, num_heads, head_dim) = (64, 16, 32).  We first project them to
# grouped heads via `prepare_qkv`, compute attention weights with
# `attention_weights`, and finally apply the weights to V and reshape back
# to the original layout using `apply_weights_and_reshape`.  The contract
# requires the node to return a single stream of shape (64, 16, 32); the
# child calls are given explicit tile‑stream shapes that match their
# expected vanilla shapes.  The parent may request an output permutation
# via `out_perms`, but this node only supports the identity permutation,
# which is the case for the current contract.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # The node currently only supports the identity permutation.
    assert out_perms is None or out_perms[0] is None, "non‑identity output permutations not supported"

    # Project Q, K, V into the grouped‑head representation.
    # Expected stream shapes:
    #   Qh: (4, 4, 64, 32)   – 4 query‑head groups, 4 heads per group
    #   Kh: (4, 1, 64, 32)   – 4 key groups, single head per group
    #   Vh: (4, 1, 64, 32)   – 4 value groups, single head per group
    Qh, Kh, Vh = prepare_qkv(
        Q, K, V,
        out_shapes=(
            (4, 4, 64, 32),
            (4, 1, 64, 32),
            (4, 1, 64, 32),
        ),
    )

    # Compute raw attention scores.
    # Shape: (4, 4, 64, 64) – per‑group query‑key dot‑product.
    attn_weights = attention_weights(
        Qh, Kh,
        out_shapes=((4, 4, 64, 64),),
    )

    # Apply the attention weights to the values and reshape back to the
    # original (seq_len, num_heads, head_dim) layout.
    # Final shape: (64, 16, 32).
    out = apply_weights_and_reshape(
        attn_weights, Vh,
        out_shapes=((64, 16, 32),),
    )

    return out