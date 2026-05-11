# The attention node simply wires the two blackbox sub‑modules.
# The inputs Q, K, V are already on‑chip stream tensors, so they can
# be passed directly to the child `compute_qkv`.  That child must be
# instructed to emit its three outputs as streams with the expected
# shapes:
#   Qh: stream (4,4)  × tile (64,32)
#   Kh: stream (4,1)  × tile (64,32)
#   Vh: stream (4,1)  × tile (64,32)
# Those shapes satisfy the STeP stream requirement (rank ≥ 3).
# Afterwards the resulting Qh, Kh, Vh are fed to `attention_compute`,
# which produces the final attention tensor.  Its output shape is
# dictated by this node’s contract (`out_shapes` and `out_perms`), so
# we forward those arguments unchanged.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # Shapes for the three tensors produced by compute_qkv.
    qkv_out_shapes = ((4, 4, 64, 32), (4, 1, 64, 32), (4, 1, 64, 32))

    # Compute Qh, Kh, Vh using the blackbox child.
    Qh, Kh, Vh = compute_qkv(
        Q, K, V,
        out_shapes=qkv_out_shapes,
        out_perms=None,
    )

    # Compute the final attention result, respecting the caller‑provided
    # output shape/permutation contract.
    return attention_compute(
        Qh, Kh, Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )