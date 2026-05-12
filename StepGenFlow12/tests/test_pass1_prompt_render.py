from src.prompts import build_pass1_user_prompt
from src.contract import Contract
from src.node_signature import TensorArg


def test_root_prompt_skips_contract_block():
    prompt = build_pass1_user_prompt(
        node_name="root",
        is_root=True,
        reference_code="def tiled_reference(dims, tensors): pass",
        dims={"B": 2},
        tensors={"x": __import__("torch").randn(2, 4)},
        contract=None,
        children_signatures=[
            ("attention", ("Q", "K", "V"),
             tuple(TensorArg(shape=(2, 4, 8, 16)) for _ in range(3)),
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
        arg_is_raw=(False,),
    )
    prompt = build_pass1_user_prompt(
        node_name="attention",
        is_root=False,
        reference_code="class Model: ...",
        dims={"B": 2},
        tensors={},
        contract=contract,
        children_signatures=[],
        function_signature="def attention(Q, *, out_shapes):",
    )
    assert "(2, 4, 2, 4, 4, 4)" in prompt   # tiled shape rendered
    assert "(2, 4, 8, 16)" in prompt         # vanilla shape rendered
    assert "single tensor" in prompt         # one-output indicator
    # The single arg's row should carry the on-chip tag.
    arg_row = next(l for l in prompt.splitlines() if "`Q`" in l)
    assert "**on-chip**" in arg_row
    assert "**RAW**" not in arg_row


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
        arg_is_raw=(True,),
    )
    prompt = build_pass1_user_prompt(
        node_name="preprocess_heads",
        is_root=False,
        reference_code="class Model: ...",
        dims={"S": 4},
        tensors={},
        contract=contract,
        children_signatures=[],
        function_signature="def preprocess_heads(x, *, out_shapes):",
    )
    # All three output shapes rendered
    assert "(4, 4, 4, 8)" in prompt
    assert "(4, 4, 1, 8)" in prompt
    # Plural convention
    assert "out_shapes" in prompt
    # Number-of-outputs indicator
    assert "Number of outputs: 3" in prompt
    # The single arg's row should carry the RAW tag.
    arg_row = next(l for l in prompt.splitlines() if "`x`:" in l)
    assert "**RAW**" in arg_row


def test_nonroot_prompt_mixes_raw_and_onchip_args():
    """Mixed arg list: one on-chip (sibling DSL output), one raw (forwarded
    weight). Both tags should appear, paired with the right arg name."""
    import torch
    contract = Contract(
        arg_names=("x_stream", "weight"),
        vanilla_shapes=((4, 16), (16, 16)),
        tiled_shapes=((4, 1, 16), (16, 16)),
        tiled_values=(torch.randn(4, 1, 16), torch.randn(16, 16)),
        out_shapes=((4, 1, 16),),
        arg_is_raw=(False, True),
    )
    prompt = build_pass1_user_prompt(
        node_name="proj",
        is_root=False,
        reference_code="class Model: ...",
        dims={"S": 4},
        tensors={},
        contract=contract,
        children_signatures=[],
        function_signature="def proj(x_stream, weight, *, out_shapes):",
    )
    # Both tags rendered.
    assert "**on-chip**" in prompt
    assert "**RAW**" in prompt
    # Tag pairs with the correct arg name on the same line.
    onchip_lines = [l for l in prompt.splitlines() if "**on-chip**" in l]
    raw_lines = [l for l in prompt.splitlines() if "**RAW**" in l and "loaded" not in l]
    # The single arg-row each.
    assert any("`x_stream`" in l for l in onchip_lines)
    assert any("`weight`" in l for l in raw_lines)
