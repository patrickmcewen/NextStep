# The attention child produces a (64, 1, 512) tiled stream (tile = 1×512).
# We broadcast the O‑projection weight (512×512) across that stream,
# flatten the extra singleton stream dimensions introduced by offchip_load_ref,
# and perform a batched matmul.  The residual tensor is loaded as a 1×512 tile
# for each sequence position, flattened to the same stream shape, and then added.
def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1️⃣  Compute the attention output in the exact tiled shape needed for the
    #     projection: stream‑dim 64, tile = (1, 512).
    attn = attention(
        Q, K, V,
        out_shapes=((64, 1, 512),),   # (seq_len, 1, hidden)
        out_perms=(None,),
    )

    # 2️⃣  Load the projection matrix (512×512) as a single tile and broadcast it
    #     to the attention stream.  offchip_load_ref adds two singleton stream
    #     dimensions (the reference stream plus the `out_shape_tiled` dim);
    #     we collapse them with `flatten`.
    weight_ref = offchip_load_ref(
        attn,
        o_proj_weight,
        stride=(0,),               # broadcast same tile everywhere
        out_shape_tiled=(1,),      # one tile in the extra dim
        tile_row=512,
        tile_col=512,
    )
    weight = flatten(weight_ref, min_rank=0, max_rank=1)   # merge the two 1‑dims → stream (64,)

    # 3️⃣  Perform the O‑projection: (64,1,512) @ (512,512) → (64,1,512)
    proj = binary_matmul(attn, weight)

    # 4️⃣  Load the residual input tensor.  Tile shape (1,512) matches the
    #     projection output tile.  offchip_load yields a leading singleton
    #     stream dimension; flatten merges it into the 64‑dim stream.
    resid_load = offchip_load(
        input_tensor,
        stride=(1,),               # advance one row per stream element
        out_shape_tiled=(64,),     # one tile per sequence position
        tile_row=1,
        tile_col=512,
    )
    resid = flatten(resid_load, min_rank=0, max_rank=1)   # shape → (64,1,512)

    # 5️⃣  Add the residual connection.
    out = binary_add(proj, resid)

    return out