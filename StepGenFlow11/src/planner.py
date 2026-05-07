"""Decomposition planner for StepGenFlow.

This module owns the tree dataclasses (``PlanNode``, ``Tree``). The LLM
response parser, mechanical guards, and recursive ``plan()`` driver land
in subsequent tasks. Phase 0 of the orchestrator calls ``plan(root_reference)``
to produce a ``Tree`` that Phase 1's per-node refactor walker consumes.
"""

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class PlanNode:
    """One node in the decomposition tree.

    ``reference_code`` is the original PyTorch reference for this node (the
    file the planner was invoked on). ``refactored_code`` is the post-split
    parent that uses children Models (None if this node is a leaf).
    """
    name: str
    path: str
    reference_code: str
    refactored_code: str | None
    is_leaf: bool
    children: tuple["PlanNode", ...]


@dataclass(frozen=True)
class Tree:
    root: PlanNode

    def iter_leaves(self) -> Iterator[PlanNode]:
        yield from _iter_leaves(self.root)

    def iter_topological(self) -> Iterator[PlanNode]:
        """Post-order: every node yielded after all its descendants."""
        yield from _iter_topological(self.root)

    def find(self, path: str) -> PlanNode:
        node = _find(self.root, path)
        assert node is not None, f"path not found in tree: {path!r}"
        return node

    def find_owner(self, path: str) -> PlanNode | None:
        """Return the immediate parent of the node at ``path`` (None for root)."""
        if path == self.root.path:
            return None
        return _find_owner(self.root, path)


def _iter_leaves(node: PlanNode) -> Iterator[PlanNode]:
    if node.is_leaf:
        yield node
        return
    for c in node.children:
        yield from _iter_leaves(c)


def _iter_topological(node: PlanNode) -> Iterator[PlanNode]:
    for c in node.children:
        yield from _iter_topological(c)
    yield node


def _find(node: PlanNode, path: str) -> PlanNode | None:
    if node.path == path:
        return node
    for c in node.children:
        found = _find(c, path)
        if found is not None:
            return found
    return None


def _find_owner(node: PlanNode, path: str) -> PlanNode | None:
    for c in node.children:
        if c.path == path:
            return node
        found = _find_owner(c, path)
        if found is not None:
            return found
    return None
