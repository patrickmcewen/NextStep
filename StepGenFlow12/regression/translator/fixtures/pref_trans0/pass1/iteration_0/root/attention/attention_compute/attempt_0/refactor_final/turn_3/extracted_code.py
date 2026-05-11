# The function takes three on‑chip streams:
#   Q : stream(seq_len) × tile(num_heads, head_dim)
#   K : stream(seq_len) × tile(num_kv_heads, head_dim)
#   V : stream(seq_len) × tile(num_kv_heads, head_dim)
#
# The heavy attention kernel (`attention_compute__root_attention_attention_compute`)
# expects its inputs in GQA layout:
#   Qh : stream(num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
#   Kh : stream(num_kv_heads, 1)                × tile(seq_len, head_dim)
#   Vh : stream(num_kv_heads, 1)                × tile(seq_len, head_dim)
#
# We reshape Q, K, V using only DSL primitives to match that layout, call the
# child, then transpose the result back to the contract‑specified shape
# (seq_len, num_heads, head_dim).  No raw tensor methods are used.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Dimension bookkeeping (pure Python scalars)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                  # 16
    head_dim = Q.shape[2]                   # 32
    num_kv_heads = K.shape[1]               # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Transform Q → Qh  (stream: kv, q_per_kv; tile rows: seq_len)
    # ------------------------------------------------------------------
    Qh = retile_streamify(Q, chunk=1, split_row=True)                     # → stream(S*H) × tile(1, D)
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)                  # → stream(num_heads, seq_len) × tile(1, D)
    Qh = reshape_stream(Qh, chunk_size=query_per_kvhead, rank=1)         # → stream(kv, q_per_kv, seq_len) × tile(1, D)
    Qh = accum_retile_row(Qh, rank=1)                                    # → stream(kv, q_per_kv) × tile(seq_len, D)

    # ------------------------------------------------------------------
    # Transform K → Kh  (stream: kv, 1; tile rows: seq_len)
    # ------------------------------------------------------------------
    Kh = retile_streamify(K, chunk=1, split_row=True)                   # → stream(S*kv) × tile(1, D)
    Kh = reshape_stream(Kh, chunk_size=seq_len, rank=0)                  # → stream(kv, seq_len) × tile(1, D)
    Kh = accum_retile_row(Kh, rank=1)                                    # → stream(kv) × tile(seq_len, D)
    Kh = promote(Kh, rank=1)                                             # → stream(kv, 1) × tile(seq_len, D)

    # ------------------------------------------------------------------
    # Transform V → Vh  (same rule as K)
    # ------------------------------------------------------------------
    Vh = retile_streamify(V, chunk=1, split_row=True)                   # → stream(S*kv) × tile(1, D)
    Vh = reshape_stream(Vh, chunk_size=seq_len, rank=0)                  # → stream(kv, seq_len) × tile(1, D)
    Vh = accum_retile_row(Vh, rank=1)                                    # → stream(kv) × tile(seq_len, D)
    Vh = promote(Vh, rank=1)                                             # → stream(kv, 1) × tile(seq_len, D)

    # ------------------------------------------------------------------
    # Heavy attention compute (child blackbox)
    # ------------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),
        ),
        out_perms=(None,),
    )
    # `attn` now has shape stream(kv, q_per_kv) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # Convert back to contract shape: (seq_len,) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    out = retile_streamify(attn, chunk=1, split_row=True)                # → stream(kv, q_per_kv*seq_len) × tile(1, D)
    out = reshape_stream(out, chunk_size=seq_len, rank=0)                # → stream(kv, q_per_kv, seq_len) × tile(1, D)
    out = flatten(out, min_rank=1, max_rank=2)                           # merge kv & q_per_kv → stream(num_heads, seq_len) × tile(1, D)
    out = flatten(out, min_rank=0, max_rank=1)                           # merge remaining stream dims → stream(num_heads*seq_len) × tile(1, D)
    out = reshape_stream(out, chunk_size=num_heads, rank=0)              # split back → stream(seq_len, num_heads) × tile(1, D)
    out = accum_retile_row(out, rank=1)                                  # move heads into tile rows → stream(seq_len) × tile(num_heads, D)

    # The produced tensor matches the declared `out_shapes[0]`.
    return out