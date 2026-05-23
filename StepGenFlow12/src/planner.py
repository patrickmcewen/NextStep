"""Decomposition planner for StepGenFlow.

This module owns the tree dataclasses (``PlanNode``, ``Tree``). The LLM
response parser, mechanical guards, and recursive ``plan()`` driver land
in subsequent tasks. Phase 0 of the orchestrator calls ``plan(root_reference)``
to produce a ``Tree`` that Phase 1's per-node refactor walker consumes.
"""

import ast
import asyncio
import inspect
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch

from src.token_accounting import write_turn_tokens


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
_CHILD_HEADER_RE = re.compile(r"^[ \t]*#\s*child:\s*(\w+)\s*$", re.MULTILINE)
_PARENT_HEADER_RE = re.compile(r"^[ \t]*#\s*refactored parent\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```(?:python|py)?\s*$", re.MULTILINE)
_FENCED_BLOCK_RE = re.compile(
    r"^```(?:python|py)?\s*\n(.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)


def _strip_code_fences(body: str) -> str:
    """Extract code from a body that may be wrapped in ``` fences.

    If at least one fenced block ``` ```python ... ``` ``` is present, returns
    only the concatenated contents of those blocks — discarding any prose
    before, between, or after them. (LLMs sometimes append a trailing
    "explanation" paragraph after the closing fence; that prose then ends up
    in ast.parse and crashes downstream guards.)

    If no fenced block is present, returns the body unchanged (modulo strip).
    """
    blocks = _FENCED_BLOCK_RE.findall(body)
    if blocks:
        return "\n\n".join(b.rstrip() for b in blocks).strip()
    # No fences: the LLM may have indented the whole block (e.g. as a sub-bullet
    # of the DECISION line). Dedent before stripping so common leading
    # whitespace is removed *per line* rather than only off the first line.
    return textwrap.dedent(body.strip("\n")).strip()


def _snake_to_camel(snake: str) -> str:
    return "".join(p.capitalize() for p in snake.split("_"))


def _normalize_child_model_class(body: str, child_name: str) -> str:
    """Allow children to be declared as ``class <CamelCase>Model`` instead of the
    literal ``class Model``. The system prompt's parent block uses
    ``<ChildName>Model`` for the references, which makes the per-child
    ``class Model`` rule easy to misread; LLMs often pick the more readable
    convention. When detected, rename the class to ``Model`` so downstream
    synth + compose keep a uniform contract.
    """
    if re.search(r"^class\s+Model\b", body, flags=re.MULTILINE):
        return body
    expected = _snake_to_camel(child_name) + "Model"
    pattern = re.compile(rf"^class\s+{re.escape(expected)}\b", flags=re.MULTILINE)
    if pattern.search(body):
        return pattern.sub("class Model", body, count=1)
    # Fallback: snake→Camel doesn't capture acronym preservation (rms_norm →
    # RMSNormModel, not RmsNormModel). If exactly one top-level
    # ``class <Identifier>Model(nn.Module)`` exists in the body, treat it as
    # the child Model and rename it. Multiple matches are ambiguous and left
    # alone so the missing-Model guard fires loudly.
    fallback = re.compile(
        r"^class\s+(\w+)Model\s*\(\s*nn\.Module\s*\)", flags=re.MULTILINE
    )
    matches = fallback.findall(body)
    if len(matches) == 1:
        return fallback.sub("class Model(nn.Module)", body, count=1)
    return body


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
        # Pass the raw inter-marker slice (no outer .strip): _strip_code_fences
        # uses textwrap.dedent on the no-fence path, which only works when the
        # common leading indent is preserved across all lines.
        body = _strip_code_fences(after_decision[header_end:next_start])
        if kind == "child":
            assert name is not None
            if name in seen_names:
                raise MalformedSplit(f"duplicate child name: {name!r}")
            seen_names.add(name)
            body = _normalize_child_model_class(body, name)
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

    if len(children) < 1:
        raise MalformedSplit(
            f"split response must contain at least 1 child, got {len(children)}"
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

    has_torch = re.search(r"^import\s+torch\s*$", out, flags=re.MULTILINE) is not None
    has_torch_nn = re.search(r"^import\s+torch\.nn\s+as\s+nn\s*$", out, flags=re.MULTILINE) is not None
    prepend = ""
    if not has_torch:
        prepend += "import torch\n"
    if not has_torch_nn:
        prepend += "import torch.nn as nn\n"
    if prepend:
        out = prepend + "\n" + out

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


def has_class_model(code: str) -> bool:
    """True iff ``code`` defines a top-level ``class Model`` with a ``forward`` method.

    A handful of StepDB references (e.g., transformer_layer/prefill_transformer_simple,
    mlp/moe_routed) ship as function-based modules with only ``compute_gold`` —
    no Model class. The planner needs a different code path for those at the root.
    """
    try:
        _extract_model_forward(code)
    except AssertionError:
        return False
    return True


def _extract_compute_gold_body(code: str) -> ast.FunctionDef:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_gold":
            return node
    raise AssertionError("could not find def compute_gold in code")


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
    """Refactored parent's forward() must be strictly smaller than original's.

    For function-based originals (no class Model — e.g. prefill_transformer_simple),
    we use ``compute_gold``'s body as the size baseline since that's where the
    actual work lives.
    """
    if has_class_model(original_parent_code):
        orig_baseline = _extract_model_forward(original_parent_code)
    else:
        orig_baseline = _extract_compute_gold_body(original_parent_code)
    new_forward = _extract_model_forward(refactored_parent_code)
    orig_size = _ast_node_count(orig_baseline)
    new_size = _ast_node_count(new_forward)
    if new_size >= orig_size:
        raise GuardFailure(
            f"refactored parent's forward() is not smaller than the original's "
            f"(refactored AST nodes: {new_size}, original: {orig_size}). The split "
            f"didn't actually move work into children. You can also consider declaring as a LEAF if you don't want to split up the problem."
        )


def _camel_case(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


# Names reserved across the tree: ``tiled_reference`` is the hardcoded entry
# point for the root DSL function; if a descendant claimed that name, its
# ``def {name}(...)`` would collide with the root's def in the pass-2
# composed namespace. Add other globally-reserved names here if they emerge.
_RESERVED_NODE_NAMES: frozenset[str] = frozenset({"tiled_reference"})


def disambiguate_tree(tree: "Tree") -> tuple["Tree", dict[str, tuple[str, str]]]:
    """Ensure every node's ``name`` is globally unique across the tree.

    Two distinct nodes sharing a ``name`` corrupts the pipeline silently:

    - **Pass-1 stub injection** keys by ``child.name``: two siblings with
      the same name would inject one stub on top of the other, recording
      a contract for only one of them.
    - **Pass-2 composition** keys by ``node.name`` (``descendants[node.name]
      = pass1_dsls[node.path]`` in ``_pass2_compose``): any cross-tree
      duplicate silently clobbers, so the composed graph executes the
      wrong DSL body.
    - **Parent/child name collision** (e.g. parent function defined as
      ``def moe_dispatch(...)`` with a child also named ``moe_dispatch``)
      shadows the child's stub in the parent's namespace, forcing the LLM
      into ugly aliasing workarounds that the textual rawness extractor
      can't follow.

    A pre-order walk fixes the first occurrence's name (or root) and
    renames any later collider to a path-derived disambiguation. Each
    rename also rewrites the offending node's *parent* refactored_code
    so its ``self.{old}`` and ``{OldCamel}Model`` references point at
    the new name. The pass-1 LLM is generated *after* this disambiguation
    runs, so its prompts are built from the new ``node.name`` and need
    no further rewrite.

    Returns ``(new_tree, renames)`` where ``renames`` maps the *original*
    path of each renamed node to ``(old_name, new_name)`` for logging.
    An empty ``renames`` dict means the tree was already unique
    (in which case ``new_tree is tree``).
    """
    used: set[str] = set(_RESERVED_NODE_NAMES)
    renames_by_path: dict[str, str] = {}  # original node.path -> new node.name
    log_renames: dict[str, tuple[str, str]] = {}  # original path -> (old, new)

    def claim(node: PlanNode, parent_path: str) -> None:
        if node.name not in used:
            used.add(node.name)
            return
        # Disambiguate using the node's parent path (with `/` swapped for
        # `_` so the result is still a valid Python identifier). Suffix
        # with a counter only if the path-derived candidate also collides.
        suffix = parent_path.replace("/", "_") if parent_path else "root"
        candidate = f"{node.name}__{suffix}"
        i = 2
        while candidate in used:
            candidate = f"{node.name}__{suffix}_{i}"
            i += 1
        renames_by_path[node.path] = candidate
        log_renames[node.path] = (node.name, candidate)
        used.add(candidate)

    # Pre-order walk: parent's name is fixed before its children are
    # considered, so on a parent/child collision the child gets renamed
    # (matches the existing user expectation that the parent path keeps
    # its function name).
    def walk(node: PlanNode, parent_path: str) -> None:
        claim(node, parent_path)
        for c in node.children:
            walk(c, node.path)

    walk(tree.root, "")
    if not renames_by_path:
        return tree, {}

    return _rebuild_with_renames(tree, renames_by_path), log_renames


def _rebuild_with_renames(tree: "Tree", renames: dict[str, str]) -> "Tree":
    """Reconstruct ``tree`` so that any node whose path is in ``renames``
    carries the new name, its path is rebuilt from ancestors' new paths,
    and its parent's ``refactored_code`` references the new name."""

    def rebuild(node: PlanNode, new_parent_path: str | None) -> PlanNode:
        new_name = renames.get(node.path, node.name)
        new_path = (node.path if new_parent_path is None
                    else f"{new_parent_path}/{new_name}")
        # If any of *this* node's direct children are being renamed, the
        # textual references in this node's own refactored_code need to
        # be patched (``self.<old>`` and ``<OldCamel>Model``).
        child_renames: dict[str, str] = {}
        for c in node.children:
            new_child_name = renames.get(c.path, c.name)
            if new_child_name != c.name:
                child_renames[c.name] = new_child_name
        new_refactored = node.refactored_code
        if new_refactored is not None and child_renames:
            new_refactored = _rewrite_parent_refactored_code(
                new_refactored, child_renames)
        new_children = tuple(rebuild(c, new_path) for c in node.children)
        return PlanNode(
            name=new_name,
            path=new_path,
            reference_code=node.reference_code,
            refactored_code=new_refactored,
            is_leaf=node.is_leaf,
            children=new_children,
        )

    return Tree(root=rebuild(tree.root, None))


def _rewrite_parent_refactored_code(code: str,
                                     renames: dict[str, str]) -> str:
    """Apply child renames (old_snake -> new_snake) to a parent's
    refactored Module code. Two textual surfaces are referenced:

    - ``self.<old>``           -> ``self.<new>``           (init + forward)
    - ``<OldCamel>Model``      -> ``<NewCamel>Model``      (class injection)

    Both replacements are word-boundary-anchored so a child whose name is
    a prefix of another (``q``, ``q_proj``) doesn't bleed into the longer
    one's references.
    """
    for old, new in renames.items():
        old_camel = _camel_case(old)
        new_camel = _camel_case(new)
        code = re.sub(rf"\bself\.{re.escape(old)}\b", f"self.{new}", code)
        code = re.sub(rf"\b{re.escape(old_camel)}Model\b",
                      f"{new_camel}Model", code)
    return code


def check_no_dead_children(refactored_parent_code: str, children: list) -> None:
    """Each child must be invoked at least once from the refactored parent's
    ``forward``. A child that is bound in ``__init__`` but never called is a
    dead child — usually a sign the LLM is padding the response to satisfy a
    quota rather than actually decomposing.
    """
    forward = _extract_model_forward(refactored_parent_code)
    called_attrs: set[str] = set()
    for node in ast.walk(forward):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name) and value.id == "self":
                called_attrs.add(node.func.attr)
    for child in children:
        if child.name not in called_attrs:
            raise GuardFailure(
                f"child {child.name!r} is dead: not invoked anywhere in the "
                f"refactored parent's forward(). Either remove the child or "
                f"wire it into the parent's computation."
            )


def check_children_runnable(children: list, dims: dict) -> None:
    """Ensure each child's reference module is actually runnable on ``dims``.

    The compose check only execs child *modules* (defining their classes); it
    never calls the children's ``get_inputs`` or ``Model()``. When a child block
    has a typo in its ``_model_config`` dispatch (or omits a helper entirely),
    the bug only surfaces at the *next* recursion level — where ``plan()`` uses
    the buggy child as the new "original" and ``check_compose`` calls the
    broken ``get_inputs(dims)``. That AssertionError/NameError is not a
    GuardFailure and crashes the whole outer iteration.

    Catching it here converts the failure into a retryable GuardFailure at the
    level where the LLM actually emitted the broken child, so the retry has the
    relevant context.
    """
    for child in children:
        ns: dict = {}
        try:
            exec(child.reference_code, ns)
        except Exception as exc:
            raise GuardFailure(
                f"child {child.name!r}: module-level exec failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if "get_inputs" not in ns:
            raise GuardFailure(
                f"child {child.name!r} reference is missing get_inputs(dims). "
                f"Every child block must define get_inputs(dims) at module scope."
            )
        if "Model" not in ns:
            raise GuardFailure(
                f"child {child.name!r} reference is missing class Model. "
                f"Every child block must define a class Model(nn.Module) at module scope."
            )
        try:
            inputs = ns["get_inputs"](dims)
        except Exception as exc:
            raise GuardFailure(
                f"child {child.name!r}: get_inputs(dims) crashed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        elif not isinstance(inputs, tuple):
            inputs = (inputs,)
        child_init_inputs = ns.get("get_init_inputs", lambda d: [])(dims)
        try:
            with torch.no_grad():
                ns["Model"](*child_init_inputs)(*inputs)
        except Exception as exc:
            raise GuardFailure(
                f"child {child.name!r}: Model()(*get_inputs(dims)) crashed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def check_compose(original_reference_code: str,
                  refactored_parent_code: str,
                  children: list,
                  dims: dict,
                  tensors: dict | None = None) -> None:
    """Numerical compose check.

    Run the refactored parent (which uses children's Model classes via
    ``<ChildName>Model``) against the original parent on the original parent's
    inputs. Outputs must agree within ``rel_err < 1e-5``.

    For function-based originals (``compute_gold(dims, tensors)`` with no
    ``get_inputs`` / ``class Model``), the caller supplies the precomputed
    ``tensors`` dict; we synthesize ``inputs = (dims, tensors)`` to match the
    LLM-emitted ``forward(self, dims, tensors)`` convention for refactored
    parents and children of such originals.
    """
    orig_ns: dict = {}
    exec(original_reference_code, orig_ns)

    if "get_inputs" in orig_ns:
        inputs = orig_ns["get_inputs"](dims)
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        elif not isinstance(inputs, tuple):
            inputs = (inputs,)
    else:
        assert "compute_gold" in orig_ns, (
            "original reference must define either get_inputs+Model or "
            "compute_gold(dims, tensors)"
        )
        assert tensors is not None, (
            "function-based original (compute_gold) requires the caller to "
            "supply precomputed tensors via plan(tensors=...)"
        )
        inputs = (dims, tensors)

    # Per-Model init args (e.g. compile-time integers like n_head). Defaults to
    # [] so kernels without get_init_inputs keep the old Model() behavior.
    orig_init_inputs = orig_ns.get("get_init_inputs", lambda d: [])(dims)
    with torch.no_grad():
        if "Model" in orig_ns:
            original_out = orig_ns["Model"](*orig_init_inputs)(*inputs)
        else:
            assert "compute_gold" in orig_ns
            if tensors is not None:
                original_out = orig_ns["compute_gold"](dims, tensors)
            else:
                original_out = orig_ns["compute_gold"](dims)

    new_ns: dict = {}
    for child in children:
        child_ns: dict = {}
        exec(child.reference_code, child_ns)
        assert "Model" in child_ns, (
            f"child {child.name!r} reference is missing class Model"
        )
        new_ns[f"{_camel_case(child.name)}Model"] = child_ns["Model"]

    try:
        exec(refactored_parent_code, new_ns)
    except Exception as exc:
        raise GuardFailure(
            f"compose check failed: refactored parent code did not exec cleanly: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    assert "Model" in new_ns, "refactored parent code must define class Model"

    # Refactored parent inherits the original's get_init_inputs (planner copies
    # it verbatim); fall back to the original's init_inputs if absent.
    new_init_inputs = new_ns.get("get_init_inputs", lambda d: orig_init_inputs)(dims)
    try:
        with torch.no_grad():
            refactored_out = new_ns["Model"](*new_init_inputs)(*inputs)
    except Exception as exc:
        raise GuardFailure(
            f"compose check failed: refactored parent's forward() crashed when "
            f"run on the original inputs: {type(exc).__name__}: {exc}. The "
            f"children's input/output contracts do not compose with the parent."
        ) from exc

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


def check_children_called_with_tensors_only(refactored_parent_code: str,
                                              children: list,
                                              dims: dict,
                                              original_reference_code: str) -> None:
    """Every parent→child call site must pass a supported ArgSpec kind.

    Pass-1 records each child call's args as a per-position contract and
    replays them through a blackbox stub. Anything outside the supported
    ArgSpec set (``TensorArg``, ``IntArg``, ``ListOfTensorArg``,
    ``ListOfIntArg``) cannot be expressed in that contract or in the DSL
    the child gets lowered to, so it has to be rejected here rather than
    crashing later in ``_build_node_index``. Python ``int`` scalars are
    accepted (they classify as ``IntArg`` and the wrapper forwards them
    unchanged); ``bool`` is rejected because there is no ``BoolArg``.
    Function-based parents (``forward(self, dims, tensors)``) are skipped
    — they're already rejected at ``refactor_tree`` entry.
    """
    if not has_class_model(original_reference_code):
        return

    orig_ns: dict = {}
    exec(original_reference_code, orig_ns)
    inputs = orig_ns["get_inputs"](dims)
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    elif not isinstance(inputs, tuple):
        inputs = (inputs,)

    parent_init_inputs = orig_ns.get("get_init_inputs", lambda d: [])(dims)
    new_ns: dict = {}
    for child in children:
        child_ns: dict = {}
        exec(child.reference_code, child_ns)
        new_ns[f"{_camel_case(child.name)}Model"] = child_ns["Model"]
    exec(refactored_parent_code, new_ns)
    parent_model = new_ns["Model"](
        *(new_ns.get("get_init_inputs", lambda d: parent_init_inputs)(dims))
    )

    child_attr_names = {c.name for c in children}
    captured: dict = {}
    hooks = []
    for attr_name, submodule in parent_model.named_children():
        if attr_name in child_attr_names:
            def make_hook(cname):
                def hook(_module, args):
                    captured.setdefault(cname, args)
                return hook
            hooks.append(submodule.register_forward_pre_hook(make_hook(attr_name)))

    with torch.no_grad():
        parent_model(*inputs)
    for h in hooks:
        h.remove()

    for child in children:
        if child.name not in captured:
            continue  # check_no_dead_children already covers unreached children
        for i, arg in enumerate(captured[child.name]):
            if isinstance(arg, torch.Tensor):
                # Reject 0-D scalar tensors — they have no tile-stream
                # representation. Direct the LLM to pass a Python int instead
                # (which classifies as IntArg and forwards unchanged).
                if arg.dim() == 0:
                    raise GuardFailure(
                        f"child {child.name!r} is called with a 0-D "
                        f"torch.Tensor at position {i}; the DSL has no "
                        f"scalar-tensor stream type. Pass the value as a "
                        f"Python int instead (e.g. `dims[\"H\"]`, not "
                        f"`torch.tensor(dims[\"H\"])`) so it classifies as "
                        f"IntArg and the child receives it as a host-side "
                        f"scalar."
                    )
                continue
            # Python int scalar (IntArg). ``bool`` is an ``int`` subclass —
            # reject it explicitly so a planner bug doesn't silently coerce
            # ``True`` into ``IntArg(1)``.
            if isinstance(arg, int) and not isinstance(arg, bool):
                continue
            if isinstance(arg, list) and len(arg) > 0:
                if all(isinstance(x, torch.Tensor) for x in arg):
                    elem_shape = tuple(arg[0].shape)
                    bad = [
                        j for j, x in enumerate(arg)
                        if tuple(x.shape) != elem_shape
                    ]
                    if bad:
                        raise GuardFailure(
                            f"child {child.name!r} is called with a "
                            f"list[Tensor] at position {i} whose elements "
                            f"have mismatched shapes (index 0: {elem_shape}, "
                            f"indices {bad[:5]}{'...' if len(bad) > 5 else ''} "
                            f"differ). list[Tensor] inputs must be "
                            f"homogeneous — every element must share the "
                            f"same shape so the child can iterate them with "
                            f"identical per-element DSL ops."
                        )
                    continue
                if all(isinstance(x, int) and not isinstance(x, bool)
                       for x in arg):
                    continue
                types = sorted({type(x).__name__ for x in arg})
                raise GuardFailure(
                    f"child {child.name!r} is called with a list at position "
                    f"{i} with mixed/unsupported element types ({types}). "
                    f"Only homogeneous list[Tensor] and list[int] are "
                    f"supported as child call-site args."
                )
            raise GuardFailure(
                f"child {child.name!r} is called with an unsupported arg at "
                f"position {i} (type={type(arg).__name__}). Allowed kinds at "
                f"a child call site: torch.Tensor (rank >= 1), Python int "
                f"(host-side scalar), list[Tensor] (homogeneous shapes), "
                f"list[int]. Bools, floats, 0-D tensors, and other scalars "
                f"are rejected — restructure so this position is one of the "
                f"supported kinds (e.g. pass `dims[\"H\"]` as a Python int, "
                f"not `torch.tensor(dims[\"H\"])` or `float(dims[\"H\"])`)."
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
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    elif not isinstance(inputs, tuple):
        inputs = (inputs,)

    assert len(arg_names) == len(inputs), (
        f"forward() declares {len(arg_names)} args ({arg_names}) but "
        f"get_inputs returned {len(inputs)} tensor(s)"
    )
    return dict(zip(arg_names, inputs))


def _path_tail(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _node_turn_dir(turn_root: Path | None, path: str, attempt: int | None,
                    *, pass_name: str | None = None) -> Path | None:
    if turn_root is None:
        return None
    safe = path.replace("/", "_")
    base = Path(turn_root) / safe
    if pass_name is not None:
        base = base / pass_name
    if attempt is not None:
        base = base / f"turn_{attempt}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _extract_reasoning(run_result) -> str:
    """Concatenate reasoning summaries from a RunResult, or "" if absent.

    Mirrors orchestrator._reasoning_text but is duplicated here to keep the
    planner module free of orchestrator imports. Tolerant of fake results
    used in tests (which have no ``new_items`` attribute).
    """
    items = getattr(run_result, "new_items", None)
    if not items:
        return ""
    from agents import ReasoningItem
    chunks: list[str] = []
    for item in items:
        if isinstance(item, ReasoningItem):
            for summary in item.raw_item.summary:
                chunks.append(summary.text)
    return "\n\n".join(chunks)


def _extract_hf_module_source(reference_code: str, dims: dict) -> str | None:
    """Return ``inspect.getsource`` for the HF class the wrapper's ``self.model``
    is an instance of, or ``None`` for any non-HF / non-introspectable reference.

    This is the planner's only HF-aware step: when ``reference_code`` defines a
    ``Model`` whose constructor builds ``self.model = <some HF module>``, the
    LLM is given the full PyTorch source of that wrapped class so it can
    decompose against the actual graph instead of inventing it.

    Best-effort: any failure (no Model, no self.model, dynamically-created
    class, missing get_init_inputs args, etc.) returns ``None`` and the prompt
    falls back to its non-HF shape. ``inspect.getsource`` itself raises
    ``OSError`` on classes whose defining file can't be located.
    """
    ns: dict = {}
    try:
        exec(reference_code, ns)
        if "Model" not in ns:
            return None
        init_inputs = ns.get("get_init_inputs", lambda d: [])(dims)
        m = ns["Model"](*init_inputs)
        inner = getattr(m, "model", None)
        if inner is None:
            return None
        cls = type(inner)
        src = inspect.getsource(cls)
        return f"# {cls.__module__}.{cls.__qualname__}\n{src}"
    except Exception:
        return None


async def plan(*, reference_code: str, dims: dict, agent, path: str,
               runner_fn, retry_budget: int = 8,
               replan_context: dict | None = None,
               log: Callable[[str], None] = print,
               turn_root: Path | None = None,
               max_depth: int | None = None,
               precompute_source: str | None = None,
               tensors: dict | None = None) -> "PlanNode":
    """Recursively decompose a node.

    ``runner_fn(agent, conversation)`` is called per LLM turn (the default real
    implementation passes ``Runner.run`` from agents-SDK; tests pass a fake).
    ``replan_context`` (when set) renders a re-plan user prompt instead of the
    initial one. ``log`` receives one line per decision point for visibility.
    ``turn_root`` (when set) is the directory under which per-node turn artifacts
    are written: ``turn_root/<node_path>/turn_<N>/{user_prompt,response,status}.txt``.
    ``max_depth`` (when set) caps tree depth: any node at depth >= max_depth is
    forced to LEAF without consulting the LLM. Depth is derived from ``path``
    (root=0, root/foo=1, root/foo/bar=2). ``None`` means no limit.
    """
    from src.prompts import build_planner_user_prompt, build_replan_user_prompt

    depth = path.count("/")
    if max_depth is not None and depth >= max_depth:
        log(f"[planner] node={path!r} at depth={depth} (max_depth={max_depth}) — forcing LEAF without LLM call")
        if turn_root is not None:
            forced_dir = turn_root / path.replace("/", "_")
            forced_dir.mkdir(parents=True, exist_ok=True)
            (forced_dir / "MAX_DEPTH_FORCED_LEAF.txt").write_text(
                f"depth={depth} reached max_depth={max_depth}; declaring LEAF without LLM call.\n"
            )
        return PlanNode(name=_path_tail(path), path=path,
                        reference_code=reference_code,
                        refactored_code=None,
                        is_leaf=True, children=())

    mode = "replan" if replan_context else "initial"
    log(f"[planner] node={path!r} ({mode}) — calling LLM, retry_budget={retry_budget}")

    hf_module_source = _extract_hf_module_source(reference_code, dims)
    if hf_module_source is not None:
        log(f"[planner] node={path!r} — including wrapped HF module source in prompt "
            f"({len(hf_module_source)} chars)")

    if replan_context is None:
        user = build_planner_user_prompt(
            reference_code=reference_code, dims=dims,
            precompute_source=precompute_source,
            hf_module_source=hf_module_source,
        )
    else:
        user = build_replan_user_prompt(
            reference_code=reference_code, dims=dims,
            replan_iteration=replan_context["replan_iteration"],
            node_path=replan_context["node_path"],
            failing_node=replan_context["failing_node"],
            last_turn_messages=replan_context["last_turn_messages"],
            sibling_results=replan_context["sibling_results"],
            precompute_source=precompute_source,
            hf_module_source=hf_module_source,
        )

    conversation = [{"role": "user", "content": user}]
    last_message = "no LLM call made"

    for attempt in range(retry_budget):
        log(f"[planner] node={path!r} attempt {attempt + 1}/{retry_budget}: awaiting LLM response")
        turn_dir = _node_turn_dir(turn_root, path, attempt)
        if turn_dir is not None:
            last_user_msg = conversation[-1]["content"] if conversation[-1]["role"] == "user" else ""
            (turn_dir / "user_prompt.txt").write_text(last_user_msg)
            instructions = getattr(agent, "instructions", None)
            if isinstance(instructions, str):
                (turn_dir / "system_prompt.txt").write_text(instructions)

        from openai import BadRequestError as _BadRequestError
        try:
            result = await runner_fn(agent, conversation)
        except _BadRequestError as exc:
            log(f"[planner] node={path!r} attempt {attempt + 1}: LLM rejected request ({exc}) — falling through to leaf fallback")
            if turn_dir is not None:
                (turn_dir / "status.txt").write_text(f"LLM_BAD_REQUEST: {exc}")
            last_message = f"<LLM rejected: {exc}>"
            break
        text = result.final_output
        last_message = text
        if turn_dir is not None:
            (turn_dir / "response.txt").write_text(text or "")
            reasoning = _extract_reasoning(result)
            if reasoning:
                (turn_dir / "reasoning.txt").write_text(reasoning)
            usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
            write_turn_tokens(turn_dir, usage, kind="main")

        def _set_status(status: str) -> None:
            if turn_dir is not None:
                (turn_dir / "status.txt").write_text(status)

        try:
            parsed = parse_planner_response(text)
        except NotADecision as exc:
            log(f"[planner] node={path!r} attempt {attempt + 1}: NO DECISION marker — retrying")
            _set_status(f"NO_DECISION: {exc}")
            conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content": (
                f"Your previous response had no DECISION: marker ({exc}). "
                f"Respond with exactly DECISION: leaf or DECISION: split + bodies."
            )})
            continue
        except MalformedSplit as exc:
            log(f"[planner] node={path!r} attempt {attempt + 1}: MALFORMED split ({exc}) — retrying")
            _set_status(f"MALFORMED_SPLIT: {exc}")
            conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content": (
                f"Your split response was malformed: {exc}. Fix and retry."
            )})
            continue

        if parsed == "leaf":
            log(f"[planner] node={path!r} attempt {attempt + 1}: decided LEAF — returning")
            _set_status("LEAF")
            return PlanNode(name=_path_tail(path), path=path,
                            reference_code=reference_code,
                            refactored_code=None,
                            is_leaf=True, children=())

        child_names = [c.name for c in parsed.children]
        log(f"[planner] node={path!r} attempt {attempt + 1}: decided SPLIT into {child_names} — running guards")

        children_full = tuple(
            ParsedChild(name=child.name,
                        reference_code=synthesize_reference_module(child.reference_code))
            for child in parsed.children
        )
        refactored_full = synthesize_reference_module(parsed.refactored_parent_code)

        try:
            check_anti_passthrough(list(children_full))
            #check_anti_monolith(reference_code, refactored_full)
            check_no_dead_children(refactored_full, list(children_full))
            check_children_runnable(list(children_full), dims)
            check_compose(reference_code, refactored_full,
                          list(children_full), dims,
                          tensors=tensors if not has_class_model(reference_code) else None)
            check_children_called_with_tensors_only(
                refactored_full, list(children_full), dims, reference_code)
        except GuardFailure as exc:
            log(f"[planner] node={path!r} attempt {attempt + 1}: GUARD FAILED ({exc}) — retrying")
            _set_status(f"GUARD_FAILED: {exc}")
            conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content": (
                f"Mechanical guard rejected your split: {exc}. Fix and retry."
            )})
            continue

        _set_status(f"SPLIT_OK: children={child_names}")
        log(f"[planner] node={path!r} guards PASSED — recursing into {len(children_full)} children")
        child_tasks = [
            plan(reference_code=child.reference_code, dims=dims, agent=agent,
                 path=f"{path}/{child.name}", runner_fn=runner_fn,
                 retry_budget=retry_budget, log=log, turn_root=turn_root,
                 max_depth=max_depth, precompute_source=precompute_source)
            for child in children_full
        ]
        child_subtrees = await asyncio.gather(*child_tasks)
        log(f"[planner] node={path!r} subtree complete (children={child_names})")

        return PlanNode(name=_path_tail(path), path=path,
                        reference_code=reference_code,
                        refactored_code=refactored_full,
                        is_leaf=False,
                        children=tuple(child_subtrees))

    # Exhaustion fallback: declare this node a leaf instead of failing the
    # whole subtree. Every retryable failure mode (anti-monolith, compose,
    # children-runnable, malformed-split, no-decision, etc.) is safely
    # recoverable as a leaf — the leaf path bypasses children entirely and
    # just runs the node's own reference_code through the refactor pass.
    # If the node has a real bug it will resurface there with better context.
    log(
        f"[planner] node={path!r} EXHAUSTED retry budget after {retry_budget} "
        f"attempts — falling back to LEAF"
    )
    if turn_root is not None:
        exhausted_dir = turn_root / path.replace("/", "_")
        exhausted_dir.mkdir(parents=True, exist_ok=True)
        (exhausted_dir / "EXHAUSTED_FALLBACK_TO_LEAF.txt").write_text(
            f"Retry budget {retry_budget} exhausted; declaring this node a leaf.\n"
            f"Last LLM message:\n\n{last_message}"
        )
    return PlanNode(name=_path_tail(path), path=path,
                    reference_code=reference_code,
                    refactored_code=None,
                    is_leaf=True, children=())
