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
