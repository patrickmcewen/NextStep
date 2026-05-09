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
        children_signatures=[
            ("attention", ("Q", "K", "V"), ((2, 4, 8, 16),) * 3,
             ((2, 4, 8, 16),), False),
        ],
        function_signature="def tiled_reference(dims, tensors):",
    )
    assert "tiled_reference" in prompt
    assert "attention" in prompt           # child blackbox listed
    assert "Contract" not in prompt        # root has no parent contract
    assert "out_shapes" in prompt          # plural convention
    assert "out_shape," not in prompt and "out_shape=" not in prompt
    # Has children → reference header announces it's the planner's
    # decomposition and the prompt teaches the self.<child>(...) → blackbox
    # mapping so the LLM doesn't re-derive the split.
    assert "planner-decomposed parent" in prompt
    assert "self.<child_name>" in prompt


def test_leaf_prompt_keeps_plain_reference_header():
    """Leaf node (no children_signatures): use the plain header, no
    decomposition note — there is nothing to decompose against."""
    prompt = build_pass1_user_prompt(
        node_name="root",
        is_root=True,
        reference_code="class Model: ...",
        dims={"B": 2},
        tensors={"x": __import__("torch").randn(2, 4)},
        contract=None,
        children_signatures=[],
        function_signature="def tiled_reference(dims, tensors):",
    )
    assert "### PyTorch Reference" in prompt
    assert "planner-decomposed parent" not in prompt
    assert "self.<child_name>" not in prompt


def test_nonroot_prompt_includes_contract():
    import torch
    contract = Contract(
        arg_names=("Q",),
        vanilla_shapes=((2, 4, 8, 16),),
        tiled_shapes=((2, 4, 2, 4, 4, 4),),
        tiled_values=(torch.randn(2, 4, 2, 4, 4, 4),),
        out_shapes=((2, 4, 2, 4, 4, 4),),
        out_perms=(None,),
    )
    prompt = build_pass1_user_prompt(
        node_name="attention",
        is_root=False,
        reference_code="class Model: ...",
        dims={"B": 2},
        tensors={},
        contract=contract,
        children_signatures=[],
        function_signature="def attention(Q, *, out_shapes, out_perms=None):",
    )
    assert "(2, 4, 2, 4, 4, 4)" in prompt   # tiled shape rendered
    assert "(2, 4, 8, 16)" in prompt         # vanilla shape rendered
    assert "single tensor" in prompt         # one-output indicator


def test_nonroot_prompt_renders_multi_output_contract():
    """Children declared by parent with N>1 outputs render plural shapes
    and instruct the LLM to destructure at the call site."""
    import torch
    contract = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 96),),
        tiled_shapes=((4, 96),),
        tiled_values=(torch.randn(4, 96),),
        out_shapes=((4, 4, 4, 8), (4, 4, 1, 8), (4, 4, 1, 8)),
        out_perms=(None, (1, 0, 2, 3), None),
    )
    prompt = build_pass1_user_prompt(
        node_name="preprocess_heads",
        is_root=False,
        reference_code="class Model: ...",
        dims={"S": 4},
        tensors={},
        contract=contract,
        children_signatures=[],
        function_signature="def preprocess_heads(x, *, out_shapes, out_perms=None):",
    )
    # All three output shapes rendered
    assert "(4, 4, 4, 8)" in prompt
    assert "(4, 4, 1, 8)" in prompt
    # Second output perm rendered
    assert "(1, 0, 2, 3)" in prompt
    # Plural convention everywhere
    assert "out_shapes" in prompt
    assert "out_perms" in prompt
    # Number-of-outputs indicator
    assert "Number of outputs: 3" in prompt
