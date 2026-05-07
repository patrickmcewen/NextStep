"""Decomposition planner for StepGenFlow.

This module owns the tree dataclasses (``PlanNode``, ``Tree``). The LLM
response parser, mechanical guards, and recursive ``plan()`` driver land
in subsequent tasks. Phase 0 of the orchestrator calls ``plan(root_reference)``
to produce a ``Tree`` that Phase 1's per-node refactor walker consumes.
"""

import re
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


class NotADecision(Exception):
    """Raised when LLM output contains no ``DECISION:`` marker."""


class MalformedSplit(AssertionError):
    """Raised when a split response is structurally invalid."""


@dataclass(frozen=True)
class ParsedChild:
    name: str
    reference_code: str


@dataclass(frozen=True)
class ParsedSplit:
    children: tuple[ParsedChild, ...]
    refactored_parent_code: str


_DECISION_RE = re.compile(r"^\s*DECISION:\s*(leaf|split)\s*$", re.MULTILINE)
_CHILD_HEADER_RE = re.compile(r"^#\s*child:\s*(\w+)\s*$", re.MULTILINE)
_PARENT_HEADER_RE = re.compile(r"^#\s*refactored parent\s*$", re.MULTILINE)


def parse_planner_response(text: str):
    """Parse a planner LLM response.

    Returns ``"leaf"`` for a leaf decision, or ``ParsedSplit`` for a split.
    Raises ``NotADecision`` if no DECISION marker is found, or
    ``MalformedSplit`` if the split structure is invalid.
    """
    m = _DECISION_RE.search(text)
    if m is None:
        raise NotADecision("response contains no DECISION: marker")
    if m.group(1) == "leaf":
        return "leaf"

    after_decision = text[m.end():]

    markers = []
    for cm in _CHILD_HEADER_RE.finditer(after_decision):
        markers.append(("child", cm.group(1), cm.start(), cm.end()))
    pm = _PARENT_HEADER_RE.search(after_decision)
    if pm is None:
        raise MalformedSplit(
            "split response is missing the '# refactored parent' section"
        )
    markers.append(("parent", None, pm.start(), pm.end()))
    markers.sort(key=lambda m: m[2])

    children: list[ParsedChild] = []
    refactored_parent_code: str | None = None
    seen_names: set[str] = set()
    for i, (kind, name, _, header_end) in enumerate(markers):
        next_start = markers[i + 1][2] if i + 1 < len(markers) else len(after_decision)
        body = after_decision[header_end:next_start].strip()
        if kind == "child":
            assert name is not None
            if name in seen_names:
                raise MalformedSplit(f"duplicate child name: {name!r}")
            seen_names.add(name)
            assert "class Model" in body, (
                f"child {name!r}: missing 'class Model' in body"
            )
            assert "def get_inputs" in body, (
                f"child {name!r}: missing 'def get_inputs' in body"
            )
            children.append(ParsedChild(name=name, reference_code=body))
        else:
            refactored_parent_code = body

    if len(children) < 2:
        raise MalformedSplit(
            f"split response must contain at least 2 children, got {len(children)}"
        )
    assert refactored_parent_code is not None

    return ParsedSplit(
        children=tuple(children),
        refactored_parent_code=refactored_parent_code,
    )


def synthesize_reference_module(body: str) -> str:
    """Take a body containing ``class Model`` + ``def get_inputs`` and return
    a full, runnable reference.py text.

    Adds ``import torch`` / ``import torch.nn as nn`` at the top if missing,
    plus ``get_init_inputs(dims) -> []`` and a default ``compute_gold(dims)``
    that runs ``Model()(*get_inputs(dims))``. Idempotent.
    """
    out = body.lstrip("\n")

    if "import torch" not in out:
        out = "import torch\nimport torch.nn as nn\n\n" + out
    elif "import torch.nn as nn" not in out:
        out = "import torch.nn as nn\n" + out

    if "def get_init_inputs(dims):" not in out:
        out = out.rstrip() + "\n\n\ndef get_init_inputs(dims):\n    return []\n"

    if "def compute_gold(dims):" not in out:
        out = out.rstrip() + (
            "\n\n\ndef compute_gold(dims):\n"
            "    inputs = get_inputs(dims)\n"
            "    return Model()(*inputs)\n"
        )

    return out
