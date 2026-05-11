# Reshape Q, K, V into the GQA layout expected by the heavy‐attention
# blackbox, invoke it, and finally reshape the result back to the contract
# shape (64, 16, 32).  All shape manipulations use only DSL primitives.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # -------------------- Q → Qh : (Hkv, Q_per_KV, S, D) --------------------
    # 1. Move the head dimension (16) into the stream, splitting it into
    #    groups of size 4 → creates a stream that interleaves token and KV‑head.
    q_tmp = retile_streamify(Q, chunk=4, split_row=True)      # stream(256,)×tile(4,32)

    # 2. Split the combined stream into the 4 KV‑head groups.
    q_hk_list = parallelize(q_tmp, n=4)                       # list of 4 tensors,
                                                             # each stream(64,)×tile(4,32)

    # 3. For each KV‑head, reshape the tensor so that the remaining head
    #    dimension (query‑per‑KV) becomes a stream dimension.
    def _sub_transform(x):
        x = reshape_stream(x, chunk_size=4, rank=0)          # stream(16,4)×tile(4,32)
        x = accum_retile_row(x, rank=1)                      # stream(16,)×tile(16,32)
        x = reshape_stream(x, chunk_size=4, rank=0)          # stream(4,4)×tile(16,32)
        x = accum_retile_row(x, rank=1)                      # stream(4,)×tile(64,32)
        x = reshape_stream(x, chunk_size=1, rank=0)          # stream(4,1)×tile(64,32)
        return x

    q_parts = [_sub_transform(t) for t in q_hk_list]        # each stream(4,1)×tile(64,32)

    # 4. Drop the singleton stream dimension and merge the remaining two
    #    stream axes (4,1 → 4) so each part is stream(4,)×tile(64,32).
    q_parts = [flatten(p, min_rank=0, max_rank=1) for p in q_parts]

    # 5. Concatenate the four parts (KV‑head dimension) and reshape to
    #    stream(Hkv, Q_per_KV) = (4,4).
    q_data, _ = eager_merge(q_parts)                         # stream(16,)×tile(64,32)
    Qh = reshape_stream(q_data, chunk_size=4, rank=0)        # stream(4,4)×tile(64,32)

    # -------------------- K → Kh : (Hkv, 1, S, D) --------------------
    Kh = reshape_stream(K, chunk_size=4, rank=0)              # stream(16,4)×tile(4,32)
    Kh = accum_retile_row(Kh, rank=1)                        # stream(16,)×tile(16,32)
    Kh = reshape_stream(Kh, chunk_size=4, rank=0)            # stream(4,4)×tile(16,32)
    Kh = accum_retile_row(Kh, rank=1)                        # stream(4,)×tile(64,32)
    Kh = reshape_stream(Kh, chunk_size=1, rank=0)            # stream(4,1)×tile(64,32)

    # -------------------- V → Vh : (Hkv, 1, S, D) --------------------
    Vh = reshape_stream(V, chunk_size=4, rank=0)              # stream(16,4)×tile(4,32)
    Vh = accum_retile_row(Vh, rank=1)                        # stream(16,)×tile(16,32)
    Vh = reshape_stream(Vh, chunk_size=4, rank=0)            # stream(4,4)×tile(16,32)
    Vh = accum_retile_row(Vh, rank=1)                        # stream(4,)×tile(64,32)
    Vh = reshape_stream(Vh, chunk_size=1, rank=0)            # stream(4,1)×tile(64,32)

    # -------------------- Heavy‑attention computation --------------------
    # Request the child's native layout; it will return a tensor with
    # vanilla shape (4, 4, 64, 32).
    child_out_shapes = ((4, 4, 64, 32),)
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh,
        out_shapes=child_out_shapes,
        out_perms=(None,),
    )

    # -------------------- Convert back to contract shape (S, H, D) --------------------
    # Merge the two GQA stream dimensions, then move the token dimension back
    # to the tile (producing stream(64,)×tile(16,32) → vanilla (64,16,32)).
    attn = flatten(attn, min_rank=0, max_rank=1)           # stream(16,)×tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)  # stream(64,)×tile(16,32)

    return attn