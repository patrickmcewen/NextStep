# The core attention operation can be delegated to the pre‑registered
# `attention_compute` blackbox.  The blackbox internally performs the
# required view‑and‑permute to obtain the (Hkv, qpkv, S, D) layout and
# carries out the attention calculation.  Since all inputs are already
# on‑chip stream tensors, we can pass them straight through without any
# additional DSL manipulation.  The desired output shape and permutation
# are expressed via the `out_shapes` and `out_perms` arguments.
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    return attention_compute(Q, K, V, out_shapes=out_shapes, out_perms=out_perms)