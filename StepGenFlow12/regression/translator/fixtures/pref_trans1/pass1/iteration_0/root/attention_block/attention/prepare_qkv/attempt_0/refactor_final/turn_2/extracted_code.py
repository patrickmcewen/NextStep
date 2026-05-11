# The Q, K, V tensors are on‑chip streams:
#   Q: (seq_len, num_heads, head_dim)
#   K, V: (seq_len, num_kv_heads, head_dim)
#
# Desired layout:
#   Qh → (num_kv_heads, query_per_kvhead, seq_len, head_dim)
#   Kh, Vh → (num_kv_heads, 1, seq_len, head_dim)
#
# The transformation can be expressed as a single off‑chip load that performs the
# logical permutation via its stride arguments, followed by a flatten that drops
# the leading singleton stream dimension and finally a row‑wise retile that moves
# the sequence length from the stream into the tile rows.
#
# For Q we load tiles of size (1, head_dim) so each tile corresponds to a single
# head row.  The stride (qpkv, 1, num_heads) together with out_shape_tiled
# (num_kv_heads, qpkv, seq_len) yields a linear index   idx = s*num_heads + hk*qpkv + qid,
# which exactly matches the PyTorch view‑permute order.
#
# For K/V we load the same 1‑row tiles but with stride (1, 0, num_kv_heads) and a
# dummy stream dimension of size 1, then flatten away the leading 1 and finally
# retile the sequence dimension into the tile rows.
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # scalar dimensions
    seq_len = Q.shape[0]                 # S
    num_heads = Q.shape[1]               # H
    num_kv_heads = K.shape[1]            # Hkv
    query_per_kvhead = num_heads // num_kv_heads  # qpkv
    head_dim = Q.shape[2]                # D (same for K, V)

    # -------------------------------------------------
    # Q → (Hkv, qpkv, S, D)
    # -------------------------------------------------
    Q_load = offchip_load(
        Q,
        stride=(query_per_kvhead, 1, num_heads),
        out_shape_tiled=(num_kv_heads, query_per_kvhead, seq_len),
        tile_row=1,
        tile_col=head_dim,
    )
    # drop the leading singleton stream dim (merge 1 × Hkv → Hkv)
    Q_flat = flatten(Q_load, min_rank=2, max_rank=3)
    # move the sequence‑length stream dim into the tile rows
    Qh = accum_retile_row(Q_flat, rank=1)

    # -------------------------------------------------
    # K → (Hkv, 1, S, D)
    # -------------------------------------------------
    K_load = offchip_load(
        K,
        stride=(1, 0, num_kv_heads),
        out_shape_tiled=(num_kv_heads, 1, seq_len),
        tile_row=1,
        tile_col=head_dim,
    )
    K_flat = flatten(K_load, min_rank=2, max_rank=3)
    Kh = accum_retile_row(K_flat, rank=1)

    # -------------------------------------------------
    # V → (Hkv, 1, S, D)
    # -------------------------------------------------
    V_load = offchip_load(
        V,
        stride=(1, 0, num_kv_heads),
        out_shape_tiled=(num_kv_heads, 1, seq_len),
        tile_row=1,
        tile_col=head_dim,
    )
    V_flat = flatten(V_load, min_rank=2, max_rank=3)
    Vh = accum_retile_row(V_flat, rank=1)

    return Qh, Kh, Vh