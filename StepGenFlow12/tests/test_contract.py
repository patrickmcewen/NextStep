import torch
from src.contract import Contract


def test_contract_records_arg_names_and_shapes():
    q = torch.randn(2, 4, 8, 16)   # tiled view
    c = Contract(
        arg_names=("Q",),
        vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 8, 16),),
        tiled_values=(q,),
        out_shape=(2, 4, 8, 16),
        out_perm=None,
    )
    assert c.arg_names == ("Q",)
    assert c.tiled_shapes == ((2, 4, 8, 16),)
    assert torch.equal(c.tiled_values[0], q)
    assert c.out_perm is None


def test_contract_with_perm_and_tiling():
    q = torch.randn(2, 4, 2, 4, 4, 4)   # [B, H, S0, D0, S1, D1]
    c = Contract(
        arg_names=("Q",),
        vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 2, 4, 4, 4),),
        tiled_values=(q,),
        out_shape=(2, 4, 2, 4, 4, 4),
        out_perm=(0, 1, 2, 4, 3, 5),
    )
    assert c.out_perm == (0, 1, 2, 4, 3, 5)
    assert c.tiled_shapes[0] == (2, 4, 2, 4, 4, 4)


def test_contract_is_frozen():
    import dataclasses
    c = Contract(arg_names=(), vanilla_shapes=(), tiled_shapes=(),
                 tiled_values=(), out_shape=(1,), out_perm=None)
    try:
        c.out_shape = (2,)   # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised
