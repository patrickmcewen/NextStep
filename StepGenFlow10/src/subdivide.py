"""Subdivide directive parsing.

The refactor agent may emit a ``SUB_TASKS`` directive instead of a
``tiled_reference`` candidate. This module provides the parser and the
shared types (``VerifiedSubTask``, ``SubdivideOptions``, ``SubdivideCounter``)
that subsequent tasks build on.

The dispatch layer (parallel sub-task execution against fresh refactor passes)
and the registry mutation logic are added in later tasks.

The verified sub-DSL is text-only — never callable from the parent's
tiled_reference. The parent agent reads it as worked code and adapts it.
"""
import torch
from dataclasses import dataclass
from typing import Callable


class NotADirective(Exception):
    """Raised when candidate code defines no top-level SUB_TASKS attribute.

    Used as a control-flow signal so callers fall through to treating the code
    as a normal tiled_reference candidate.
    """


@dataclass(frozen=True)
class VerifiedSubTask:
    name: str
    sub_reference_source: str
    preamble_source: str
    verified_sub_dsl_source: str


@dataclass(frozen=True)
class SubdivideOptions:
    max_subdivide_turns: int  # used by dispatch (T9), not parser
    max_subdivides_per_outer: int
    max_subdivide_depth: int


@dataclass
class SubdivideCounter:
    """Per-outer-attempt counter. Mutated as sub-tasks are dispatched."""
    used: int = 0


_REQUIRED_KEYS = {"name", "preamble", "sub_reference"}


def _func_source(fn: Callable, full_source: str) -> str:
    """Extract the source of a function defined inside ``full_source``.

    The function was just exec'd from ``full_source``, so inspect.getsource
    cannot locate a file for it. We walk the AST of ``full_source`` and match
    by both name and starting line number — name alone is not unique if the
    directive code redefines a function.
    """
    import ast
    tree = ast.parse(full_source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == fn.__name__
                and node.lineno == fn.__code__.co_firstlineno):
            segment = ast.get_source_segment(full_source, node)
            assert segment is not None, (
                f"AST source extraction failed for {fn.__name__!r}; "
                f"this should not happen for code defined in full_source"
            )
            return segment
    assert False, (
        f"function {fn.__name__!r} (lineno={fn.__code__.co_firstlineno}) "
        f"not found in source"
    )


def parse_directive(
    code: str,
    *,
    registry: list[VerifiedSubTask],
    depth: int,
    counter: SubdivideCounter,
    options: SubdivideOptions,
) -> list[dict]:
    namespace = {}
    exec(compile(code, "<subdivide_directive>", "exec"), namespace)

    if "SUB_TASKS" not in namespace:
        raise NotADirective()

    sub_tasks = namespace["SUB_TASKS"]
    assert isinstance(sub_tasks, list), "SUB_TASKS must be a list"
    assert len(sub_tasks) > 0, "SUB_TASKS must be a non-empty list"

    assert depth < options.max_subdivide_depth, (
        f"depth {depth} exceeds max_subdivide_depth {options.max_subdivide_depth}; "
        f"cannot subdivide further"
    )

    assert counter.used + len(sub_tasks) <= options.max_subdivides_per_outer, (
        f"adding {len(sub_tasks)} sub-task(s) (used={counter.used}) exceeds "
        f"the per-outer cap of {options.max_subdivides_per_outer}"
    )

    registry_names = {vst.name for vst in registry}
    seen_names: set[str] = set()
    parsed = []

    for st in sub_tasks:
        assert isinstance(st, dict), f"each SUB_TASKS entry must be a dict, got {type(st)}"

        missing = _REQUIRED_KEYS - st.keys()
        assert not missing, (
            f"sub-task dict missing required keys: {sorted(missing)}"
        )

        extra = st.keys() - _REQUIRED_KEYS
        assert not extra, f"sub-task dict has unexpected keys: {sorted(extra)}"

        name = st["name"]
        assert isinstance(name, str) and name, (
            f"sub-task 'name' must be a non-empty string, got {name!r}"
        )

        assert name not in seen_names, (
            f"duplicate sub-task name {name!r} within directive"
        )
        seen_names.add(name)

        assert name not in registry_names, (
            f"sub-task name {name!r} already verified in registry"
        )

        preamble = st["preamble"]
        sub_reference = st["sub_reference"]
        assert callable(preamble), (
            f"sub-task {name!r}: 'preamble' must be callable, got {type(preamble).__name__}"
        )
        assert callable(sub_reference), (
            f"sub-task {name!r}: 'sub_reference' must be callable, got {type(sub_reference).__name__}"
        )

        parsed.append({
            "name": name,
            "preamble": preamble,
            "sub_reference": sub_reference,
            "preamble_source": _func_source(preamble, code),
            "sub_reference_source": _func_source(sub_reference, code),
        })

    return parsed


@dataclass(frozen=True)
class PreparedSubTask:
    name: str
    sub_tensors: dict
    sub_gold: torch.Tensor | tuple[torch.Tensor, ...]
    preamble_source: str
    sub_reference_source: str
    sub_reference: Callable  # retained for re-executing gold lookup without re-parsing


def prepare_sub_task(parsed: dict, *, dims: dict,
                     parent_tensors: dict) -> PreparedSubTask:
    """Run preamble and sub_reference to produce sub_tensors and sub_gold.

    Wraps the agent-supplied callables in narrow validation: each invocation
    must succeed and return the expected shape (dict, Tensor, or tuple of
    Tensors). Agent-code exceptions are funneled into AssertionError so the
    orchestrator can surface them as directive feedback (the orchestrator's
    directive-failure channel is AssertionError per spec).
    """
    name = parsed["name"]
    preamble = parsed["preamble"]
    sub_reference = parsed["sub_reference"]

    try:
        sub_tensors = preamble(dims, parent_tensors)
    except Exception as exc:
        raise AssertionError(
            f"sub-task {name!r}: preamble raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    assert isinstance(sub_tensors, dict), (
        f"sub-task {name!r}: preamble must return a dict of tensors, "
        f"got {type(sub_tensors).__name__}"
    )

    try:
        sub_gold = sub_reference(dims, sub_tensors)
    except Exception as exc:
        raise AssertionError(
            f"sub-task {name!r}: sub_reference raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    # Mirrors the tuple/tensor validation in tools.py::_exec_dsl_ref. Keep in sync.
    if isinstance(sub_gold, tuple):
        assert len(sub_gold) > 0, (
            f"sub-task {name!r}: sub_reference returned an empty tuple"
        )
        for i, elem in enumerate(sub_gold):
            assert isinstance(elem, torch.Tensor), (
                f"sub-task {name!r}: sub_reference must return a "
                f"torch.Tensor or tuple of torch.Tensor, got tuple element "
                f"[{i}] of type {type(elem).__name__}"
            )
    else:
        assert isinstance(sub_gold, torch.Tensor), (
            f"sub-task {name!r}: sub_reference must return a torch.Tensor "
            f"or tuple of torch.Tensor, got {type(sub_gold).__name__}"
        )

    return PreparedSubTask(
        name=name,
        sub_tensors=sub_tensors,
        sub_gold=sub_gold,
        preamble_source=parsed["preamble_source"],
        sub_reference_source=parsed["sub_reference_source"],
        sub_reference=sub_reference,
    )
