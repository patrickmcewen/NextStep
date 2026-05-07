"""Decomposition planner for StepGenFlow.

This module owns the tree dataclasses (``PlanNode``, ``Tree``). The LLM
response parser, mechanical guards, and recursive ``plan()`` driver land
in subsequent tasks. Phase 0 of the orchestrator calls ``plan(root_reference)``
to produce a ``Tree`` that Phase 1's per-node refactor walker consumes.
"""

import ast
import asyncio
import re
from dataclasses import dataclass
from typing import Iterator

import torch


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
_CODE_FENCE_RE = re.compile(r"^```(?:python|py)?\s*$", re.MULTILINE)


def _strip_code_fences(body: str) -> str:
    """Remove markdown ``` fences that LLMs sometimes wrap around block bodies."""
    return _CODE_FENCE_RE.sub("", body).strip()


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
        body = _strip_code_fences(after_decision[header_end:next_start].strip())
        if kind == "child":
            assert name is not None
            if name in seen_names:
                raise MalformedSplit(f"duplicate child name: {name!r}")
            seen_names.add(name)
            if "class Model" not in body:
                raise MalformedSplit(
                    f"child {name!r}: missing 'class Model' in body"
                )
            if "def get_inputs" not in body:
                raise MalformedSplit(
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


class GuardFailure(AssertionError):
    """Raised when a mechanical guard rejects a parsed split.

    The driver catches this and feeds ``str(exc)`` back to the LLM as guard
    feedback in the next retry within the same planner call.
    """


def _extract_model_forward(code: str) -> ast.FunctionDef:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return item
    raise AssertionError("could not find class Model with forward() method in code")


def _forward_body_op_count(forward_node: ast.FunctionDef) -> int:
    count = 0
    for node in ast.walk(forward_node):
        if isinstance(node, (ast.BinOp, ast.Call)):
            count += 1
    return count


def check_anti_passthrough(children: list) -> None:
    """Each child's forward() must contain >=1 torch operation."""
    for child in children:
        forward = _extract_model_forward(child.reference_code)
        ops = _forward_body_op_count(forward)
        if ops == 0:
            raise GuardFailure(
                f"child {child.name!r}: forward() is a passthrough "
                f"(no torch operations in body). A child must do real work."
            )


def _ast_node_count(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def check_anti_monolith(original_parent_code: str, refactored_parent_code: str) -> None:
    """Refactored parent's forward() must be strictly smaller than original's."""
    orig_forward = _extract_model_forward(original_parent_code)
    new_forward = _extract_model_forward(refactored_parent_code)
    orig_size = _ast_node_count(orig_forward)
    new_size = _ast_node_count(new_forward)
    if new_size >= orig_size:
        raise GuardFailure(
            f"refactored parent's forward() is not smaller than the original's "
            f"(refactored AST nodes: {new_size}, original: {orig_size}). The split "
            f"didn't actually move work into children."
        )


def _camel_case(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def check_compose(original_reference_code: str,
                  refactored_parent_code: str,
                  children: list,
                  dims: dict) -> None:
    """Numerical compose check.

    Run the refactored parent (which uses children's Model classes via
    ``<ChildName>Model``) against the original parent on the original parent's
    inputs. Outputs must agree within ``rel_err < 1e-5``.
    """
    orig_ns: dict = {}
    exec(original_reference_code, orig_ns)
    assert "Model" in orig_ns and "get_inputs" in orig_ns, (
        "original reference must define Model and get_inputs"
    )

    inputs = orig_ns["get_inputs"](dims)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    with torch.no_grad():
        original_out = orig_ns["Model"]()(*inputs)

    new_ns: dict = {}
    for child in children:
        child_ns: dict = {}
        exec(child.reference_code, child_ns)
        assert "Model" in child_ns, (
            f"child {child.name!r} reference is missing class Model"
        )
        new_ns[f"{_camel_case(child.name)}Model"] = child_ns["Model"]

    exec(refactored_parent_code, new_ns)
    assert "Model" in new_ns, "refactored parent code must define class Model"

    with torch.no_grad():
        refactored_out = new_ns["Model"]()(*inputs)

    pairs = (
        list(zip(original_out, refactored_out))
        if isinstance(original_out, tuple)
        else [(original_out, refactored_out)]
    )
    for i, (a, b) in enumerate(pairs):
        assert a.shape == b.shape, (
            f"compose check failed: shape mismatch at output {i}: "
            f"{tuple(a.shape)} vs {tuple(b.shape)}"
        )
        max_abs = (a - b).abs().max().item()
        rel = max_abs / (a.abs().max().item() + 1e-12)
        if rel >= 1e-5:
            raise GuardFailure(
                f"compose check failed: output {i} has rel_err={rel:.3e} "
                f"(max_abs_err={max_abs:.3e}). The refactored parent does not "
                f"reproduce the original's output."
            )


def build_node_tensors(reference_code: str, dims: dict) -> dict:
    """Run ``get_inputs(dims)`` and zip the resulting tuple with the names of
    ``Model.forward``'s positional arguments (excluding ``self``).

    Returns a ``{arg_name: Tensor, ...}`` dict.
    """
    forward = _extract_model_forward(reference_code)
    arg_names = [a.arg for a in forward.args.args if a.arg != "self"]

    ns: dict = {}
    exec(reference_code, ns)
    assert "get_inputs" in ns, "reference must define get_inputs(dims)"
    inputs = ns["get_inputs"](dims)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    assert len(arg_names) == len(inputs), (
        f"forward() declares {len(arg_names)} args ({arg_names}) but "
        f"get_inputs returned {len(inputs)} tensor(s)"
    )
    return dict(zip(arg_names, inputs))


class PlannerExhausted(RuntimeError):
    """Raised when a planner call exhausts its retry budget without a valid response."""

    def __init__(self, node_path: str, last_message: str):
        self.node_path = node_path
        self.last_message = last_message
        super().__init__(
            f"planner exhausted retry budget at node {node_path!r}: {last_message}"
        )


def _path_tail(path: str) -> str:
    return path.rsplit("/", 1)[-1]


async def plan(*, reference_code: str, dims: dict, agent, path: str,
               runner_fn, retry_budget: int = 3,
               replan_context: dict | None = None) -> "PlanNode":
    """Recursively decompose a node.

    ``runner_fn(agent, conversation)`` is called per LLM turn (the default real
    implementation passes ``Runner.run`` from agents-SDK; tests pass a fake).
    ``replan_context`` (when set) renders a re-plan user prompt instead of the
    initial one.
    """
    from src.prompts import build_planner_user_prompt, build_replan_user_prompt

    if replan_context is None:
        user = build_planner_user_prompt(reference_code=reference_code, dims=dims)
    else:
        user = build_replan_user_prompt(
            reference_code=reference_code, dims=dims,
            replan_iteration=replan_context["replan_iteration"],
            node_path=replan_context["node_path"],
            failing_node=replan_context["failing_node"],
            last_turn_messages=replan_context["last_turn_messages"],
            sibling_results=replan_context["sibling_results"],
        )

    conversation = [{"role": "user", "content": user}]
    last_message = "no LLM call made"

    for attempt in range(retry_budget):
        result = await runner_fn(agent, conversation)
        text = result.final_output
        last_message = text

        try:
            parsed = parse_planner_response(text)
        except NotADecision as exc:
            conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content": (
                f"Your previous response had no DECISION: marker ({exc}). "
                f"Respond with exactly DECISION: leaf or DECISION: split + bodies."
            )})
            continue
        except MalformedSplit as exc:
            conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content": (
                f"Your split response was malformed: {exc}. Fix and retry."
            )})
            continue

        if parsed == "leaf":
            return PlanNode(name=_path_tail(path), path=path,
                            reference_code=reference_code,
                            refactored_code=None,
                            is_leaf=True, children=())

        children_full = tuple(
            ParsedChild(name=child.name,
                        reference_code=synthesize_reference_module(child.reference_code))
            for child in parsed.children
        )
        refactored_full = synthesize_reference_module(parsed.refactored_parent_code)

        try:
            check_anti_passthrough(list(children_full))
            check_anti_monolith(reference_code, refactored_full)
            check_compose(reference_code, refactored_full,
                          list(children_full), dims)
        except GuardFailure as exc:
            conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content": (
                f"Mechanical guard rejected your split: {exc}. Fix and retry."
            )})
            continue

        child_tasks = [
            plan(reference_code=child.reference_code, dims=dims, agent=agent,
                 path=f"{path}/{child.name}", runner_fn=runner_fn,
                 retry_budget=retry_budget)
            for child in children_full
        ]
        child_subtrees = await asyncio.gather(*child_tasks)

        return PlanNode(name=_path_tail(path), path=path,
                        reference_code=reference_code,
                        refactored_code=refactored_full,
                        is_leaf=False,
                        children=tuple(child_subtrees))

    raise PlannerExhausted(path, last_message)
