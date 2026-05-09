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
