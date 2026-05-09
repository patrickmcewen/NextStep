from src.prompts import build_pass1_user_prompt
from src.contract import Contract


def test_root_prompt_skips_contract_block():
    prompt = build_pass1_user_prompt(
        node_name="root",
        is_root=True,
        reference_code="def tiled_reference(dims, tensors): pass",
        dims={"B": 2},
        tensors={"x": __import__("torch").randn(2, 4)},
        contract=None,
        children_signatures=[("attention", ("Q", "K", "V"), ((2, 4, 8, 16),)*3)],
        function_signature="def tiled_reference(dims, tensors):",
    )
    assert "tiled_reference" in prompt
    assert "attention" in prompt           # child blackbox listed
    assert "Contract" not in prompt        # root has no parent contract


def test_nonroot_prompt_includes_contract():
    import torch
    contract = Contract(
        arg_names=("Q",),
        vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 2, 4, 4, 4),),
        tiled_values=(torch.randn(2, 4, 2, 4, 4, 4),),
        out_shape=(2, 4, 2, 4, 4, 4),
        out_perm=None,
    )
    prompt = build_pass1_user_prompt(
        node_name="attention",
        is_root=False,
        reference_code="class Model: ...",
        dims={"B": 2},
        tensors={},
        contract=contract,
        children_signatures=[],
        function_signature="def attention(Q, *, out_shape, out_perm=None):",
    )
    assert "(2, 4, 2, 4, 4, 4)" in prompt   # tiled shape rendered
    assert "(2, 4, 8, 16)" in prompt         # vanilla shape rendered
