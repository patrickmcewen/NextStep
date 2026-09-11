import torch
from src.contract import Contract


def test_contract_records_arg_names_and_shapes():
    q = torch.randn(2, 4, 8, 16)   # tiled view
    c = Contract(
        arg_names=("Q",),
        vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 8, 16),),
        tiled_values=(q,),
        out_shapes=((2, 4, 8, 16),),
    )
    assert c.arg_names == ("Q",)
    assert c.tiled_shapes == ((2, 4, 8, 16),)
    assert torch.equal(c.tiled_values[0], q)


def test_contract_with_tiling():
    q = torch.randn(2, 4, 2, 4, 4, 4)   # [B, H, S0, D0, S1, D1]
    c = Contract(
        arg_names=("Q",),
        vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 2, 4, 4, 4),),
        tiled_values=(q,),
        out_shapes=((2, 4, 2, 4, 4, 4),),
    )
    assert c.tiled_shapes[0] == (2, 4, 2, 4, 4, 4)


def test_contract_with_multiple_outputs():
    """Multi-output children (e.g. preprocess_heads -> Qh, Kh, Vh) record
    one ``out_shapes`` entry per output."""
    c = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 64),),
        tiled_shapes=((4, 64),),
        tiled_values=(torch.randn(4, 64),),
        out_shapes=((4, 8, 8), (4, 4, 8), (4, 4, 8)),
    )
    assert len(c.out_shapes) == 3
    assert c.out_shapes[0] == (4, 8, 8)


def test_contract_is_frozen():
    import dataclasses
    c = Contract(arg_names=(), vanilla_shapes=(), tiled_shapes=(),
                 tiled_values=(), out_shapes=((1, 1, 1),))
    try:
        c.out_shapes = ((1, 2, 1),)   # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


def test_contract_rejects_sub_stream_rank_out_shapes():
    """STeP invariant: every out_shape must be at least rank 3
    (>= 1 stream dim + 2 tile dims)."""
    raised = False
    try:
        Contract(arg_names=(), vanilla_shapes=(), tiled_shapes=(),
                 tiled_values=(), out_shapes=((4, 8),))
    except AssertionError as e:
        raised = True
        assert "rank >= 3" in str(e), str(e)
    assert raised, "Contract should reject 2D out_shapes"


def test_contract_rejects_sub_stream_rank_in_multi_output():
    """The invariant is per-entry: one bad shape in a multi-output contract fails."""
    raised = False
    try:
        Contract(arg_names=(), vanilla_shapes=(), tiled_shapes=(),
                 tiled_values=(),
                 out_shapes=((4, 8, 2), (4, 8)))   # second is 2D
    except AssertionError as e:
        raised = True
        assert "out_shapes[1]" in str(e), str(e)
    assert raised


def test_contract_max_tile_none_skips_check():
    """``max_tile=None`` (default) leaves the existing behavior untouched."""
    Contract(
        arg_names=("Q",), vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 8, 16),), tiled_values=(torch.randn(2, 4, 8, 16),),
        out_shapes=((2, 4, 8, 16),), max_tile=None,
    )


def test_contract_max_tile_rejects_oversize_out_shape():
    """Last two dims of an out_shape must be <= max_tile."""
    raised = False
    try:
        Contract(
            arg_names=("Q",), vanilla_shapes=((2, 4, 8, 16),),
            tiled_shapes=((2, 4, 8, 16),),
            tiled_values=(torch.randn(2, 4, 8, 16),),
            out_shapes=((2, 4, 8, 16),),   # tile (8, 16) > max_tile=4
            max_tile=4,
        )
    except AssertionError as e:
        raised = True
        assert "out_shapes[0]" in str(e) and "max_tile=4" in str(e), str(e)
    assert raised


def test_contract_max_tile_accepts_in_bound_out_shape():
    """Tile dims at the bound are allowed (``<=`` not ``<``)."""
    Contract(
        arg_names=("Q",), vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 2, 4, 4, 4),),
        tiled_values=(torch.randn(2, 4, 2, 4, 4, 4),),
        out_shapes=((2, 4, 2, 4, 4, 4),),   # tile (4, 4) == max_tile
        max_tile=4,
        arg_is_raw=(False,),
    )


def test_contract_max_tile_skips_tiled_shapes_when_arg_is_raw_unstamped():
    """Un-stamped contracts (``arg_is_raw=()``) defer the tiled_shapes check;
    only out_shapes is enforced at first construction. The orchestrator
    re-validates after stamping rawness via ``dataclasses.replace``."""
    # tiled_shapes last dims are (8, 16) — would exceed max_tile=4 if checked,
    # but ``arg_is_raw=()`` means we don't yet know if Q is on-chip or raw.
    Contract(
        arg_names=("Q",), vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 8, 16),),
        tiled_values=(torch.randn(2, 4, 8, 16),),
        out_shapes=((2, 4, 1, 1),),   # in-bound out_shape
        max_tile=4,
    )


def test_contract_max_tile_skips_raw_args_in_tiled_shapes():
    """Raw forwards (``arg_is_raw[i] = True``) store the vanilla shape in
    tiled_shapes — the child's ``offchip_load`` is what tiles them, so the
    parent's call-site shape legitimately exceeds the cap."""
    Contract(
        arg_names=("W",), vanilla_shapes=((128, 128),),
        tiled_shapes=((128, 128),),   # vanilla — would fail if max_tile checked
        tiled_values=(torch.randn(128, 128),),
        out_shapes=((1, 4, 4),),
        max_tile=4,
        arg_is_raw=(True,),
    )


def test_contract_max_tile_rejects_oversize_on_chip_tiled_shape():
    """On-chip tensor args (``arg_is_raw[i] = False``) must have their last
    two dims within ``max_tile``."""
    raised = False
    try:
        Contract(
            arg_names=("Q",), vanilla_shapes=((2, 4, 8, 16),),
            tiled_shapes=((2, 4, 8, 16),),   # tile (8, 16) > max_tile=4
            tiled_values=(torch.randn(2, 4, 8, 16),),
            out_shapes=((2, 4, 4, 4),),
            max_tile=4,
            arg_is_raw=(False,),
        )
    except AssertionError as e:
        raised = True
        assert "tiled_shapes[0]" in str(e) and "max_tile=4" in str(e), str(e)
    assert raised


def test_contract_max_tile_rechecks_on_replace_with_arg_is_raw():
    """``dataclasses.replace`` re-runs ``__post_init__``; once arg_is_raw is
    stamped, an on-chip tiled_shape that exceeds the bound fails."""
    import dataclasses
    # First construction passes (un-stamped, tiled_shapes deferred).
    c = Contract(
        arg_names=("Q",), vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 8, 16),),
        tiled_values=(torch.randn(2, 4, 8, 16),),
        out_shapes=((2, 4, 1, 1),),
        max_tile=4,
    )
    # Stamping as raw is fine — tiled_shape skipped for raw args.
    dataclasses.replace(c, arg_is_raw=(True,))
    # Stamping as on-chip now fires the deferred check.
    raised = False
    try:
        dataclasses.replace(c, arg_is_raw=(False,))
    except AssertionError as e:
        raised = True
        assert "tiled_shapes[0]" in str(e), str(e)
    assert raised


def test_contract_max_tile_rejects_non_positive():
    """``max_tile`` must be a positive int (or None)."""
    raised = False
    try:
        Contract(
            arg_names=(), vanilla_shapes=(), tiled_shapes=(),
            tiled_values=(), out_shapes=((1, 1, 1),), max_tile=0,
        )
    except AssertionError as e:
        raised = True
        assert "max_tile" in str(e), str(e)
    assert raised
