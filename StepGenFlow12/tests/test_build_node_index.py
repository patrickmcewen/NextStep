"""Unit tests for _build_node_index (Task 8).

Uses a minimal two-level tree fixture: a root that calls a ChildModel, and a
leaf child.  The ROOT_REF inlines ChildModel so the two exec namespaces don't
need to share class definitions.
"""

import torch

from src.planner import PlanNode, Tree

# Root reference inlines ChildModel so exec works without cross-namespace sharing.
ROOT_REF = '''
import torch
import torch.nn as nn


class ChildModel(nn.Module):
    def forward(self, x):
        return x + 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = ChildModel()

    def forward(self, x):
        y = self.child(x)
        return y * 2


def get_inputs(dims):
    return [torch.randn(4, 8)]


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
'''

# Child's standalone reference (its own class Model, same semantics as ChildModel)
CHILD_REF = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return x + 1


def get_inputs(dims):
    return [torch.randn(4, 8)]


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
'''


def _make_tree():
    leaf = PlanNode(
        name="child",
        path="root/child",
        reference_code=CHILD_REF,
        refactored_code=None,
        is_leaf=True,
        children=(),
    )
    root = PlanNode(
        name="root",
        path="root",
        reference_code=ROOT_REF,
        refactored_code=ROOT_REF,  # refactored_code used to capture child inputs
        is_leaf=False,
        children=(leaf,),
    )
    return Tree(root=root)


def test_build_node_index_root_signature():
    from src.orchestrator import _build_node_index

    tree = _make_tree()
    tensors = {"x": torch.randn(4, 8)}
    signatures, ref_modules = _build_node_index(tree, tensors)

    assert signatures["root"].arg_names == ("x",)
    assert signatures["root"].arg_shapes == ((4, 8),)
    assert "root" in ref_modules


def test_build_node_index_child_signature():
    from src.orchestrator import _build_node_index

    tree = _make_tree()
    tensors = {"x": torch.randn(4, 8)}
    signatures, ref_modules = _build_node_index(tree, tensors)

    # The child is called with x directly from the root's forward
    assert signatures["root/child"].arg_names == ("x",)
    assert signatures["root/child"].arg_shapes == ((4, 8),)
    assert "root/child" in ref_modules


# Function-based StepDB references define only `compute_gold(dims, tensors)` —
# no `class Model`. The planner produces children with `forward(self, dims, tensors)`,
# which the v1 contract design (one tensor per arg) cannot represent. `refactor_tree`
# must reject this combination at entry rather than crashing later.
FUNCTION_BASED_ROOT_REF = '''
import torch


def compute_gold(dims, tensors):
    return tensors["x"] + 1
'''


def test_refactor_tree_rejects_function_based_root_with_children():
    import asyncio

    import pytest

    from src.orchestrator import refactor_tree

    leaf = PlanNode(
        name="child", path="root/child", reference_code=CHILD_REF,
        refactored_code=None, is_leaf=True, children=(),
    )
    root = PlanNode(
        name="root", path="root", reference_code=FUNCTION_BASED_ROOT_REF,
        refactored_code="<unused for this guard>", is_leaf=False, children=(leaf,),
    )
    tree = Tree(root=root)

    async def _run():
        await refactor_tree(
            tree=tree, dims={}, root_kernel="dummy",
            ckpt_root=None, agent_factory=None, max_turns=1,
            log=lambda *_a, **_k: None, tensors={"x": torch.randn(4, 8)},
        )

    with pytest.raises(AssertionError, match="function-based"):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# List-typed root inputs (per-expert weight stacks, per-batch seq lengths)
# ---------------------------------------------------------------------------

LEAF_LIST_INPUT_REF = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, w_gate_list, num_token_list):
        out = torch.zeros_like(x)
        for w in w_gate_list:
            out = out + x @ w
        for i in range(x.shape[0]):
            out[i, :num_token_list[i]] = out[i, :num_token_list[i]] * 2.0
        return out


def get_inputs(dims):
    return [
        torch.randn(4, 8),
        [torch.randn(8, 8) for _ in range(3)],
        [2, 5, 3, 7],
    ]


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
'''


def test_build_node_index_root_with_list_inputs():
    """The end_to_end crash repro: root forward takes list[Tensor] + list[int].

    Pre-fix, _build_node_index → extract_signature → tuple(t.shape) crashed
    on the first list. Post-fix, signature classifies each input as TensorArg,
    ListOfTensorArg, or ListOfIntArg.
    """
    from src.node_signature import ListOfIntArg, ListOfTensorArg, TensorArg
    from src.orchestrator import _build_node_index

    leaf = PlanNode(
        name="root", path="root",
        reference_code=LEAF_LIST_INPUT_REF,
        refactored_code=None, is_leaf=True, children=(),
    )
    tree = Tree(root=leaf)
    tensors = {
        "x": torch.randn(4, 8),
        "w_gate_list": [torch.randn(8, 8) for _ in range(3)],
        "num_token_list": [2, 5, 3, 7],
    }
    signatures, ref_modules = _build_node_index(tree, tensors)

    sig = signatures["root"]
    assert sig.arg_names == ("x", "w_gate_list", "num_token_list")
    assert sig.arg_specs == (
        TensorArg(shape=(4, 8)),
        ListOfTensorArg(length=3, elem_shape=(8, 8)),
        ListOfIntArg(length=4),
    )
    assert sig.out_shapes == ((4, 8),)
