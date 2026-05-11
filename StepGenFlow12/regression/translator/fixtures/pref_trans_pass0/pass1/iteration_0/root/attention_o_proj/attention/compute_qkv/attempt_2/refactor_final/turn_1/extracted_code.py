# The reference implementation simply reshapes Q and permutes its dimensions,
# while K and V are just permuted and given an extra singleton dimension.
# Although the DSL provides a rich set of operators for manipulating tiled
# streams, the most direct way to match the reference semantics is to use
# the same view/permute steps.  The contract guarantees that the inputs are
# already on‑chip tile streams, so we can safely apply these operations
# without any off‑chip load.
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Q: (S, N_q, D)  ->  (N_kv, q_per_kv, S, D)
    Qh = (
        Q.view(out_shapes[0][2], out_shapes[0][0] * out_shapes[0][1], -1)
           .permute(1, 2, 0, 3)
    )  # (4, 4, 64, 32)

    # K: (S, N_kv, D)  ->  (N_kv, 1, S, D)
    Kh = K.permute(1, 0, 2).unsqueeze(1)   # (4, 1, 64, 32)

    # V: (S, N_kv, D)  ->  (N_kv, 1, S, D)
    Vh = V.permute(1, 0, 2).unsqueeze(1)   # (4, 1, 64, 32)

    return Qh, Kh, Vh