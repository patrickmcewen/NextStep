"""Subdivide directive: parse, dispatch, registry.

The refactor agent may emit a ``SUB_TASKS`` directive instead of a
``tiled_reference`` candidate. The orchestrator parses the directive here,
dispatches each sub-task as a fresh recursive ``refactor_final`` pass on a
smaller mini-kernel (preamble + sub_reference), collects verified sub-DSLs,
and adds them to the parent's per-outer registry as reference material.

The verified sub-DSL is text-only — never callable from the parent's
tiled_reference. The parent agent reads it as worked code and adapts it.
"""
from dataclasses import dataclass
from typing import Callable


class NotADirective(Exception):
    """Raised when candidate code defines no top-level SUB_TASKS attribute.

    Used as a control-flow signal so callers fall through to treating the code
    as a normal tiled_reference candidate.
    """


@dataclass
class VerifiedSubTask:
    name: str
    sub_reference_source: str
    preamble_source: str
    verified_sub_dsl_source: str


@dataclass
class SubdivideOptions:
    max_subdivide_turns: int
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
    cannot locate a file for it. We walk the AST of full_source to find the
    matching FunctionDef.
    """
    import ast
    tree = ast.parse(full_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            segment = ast.get_source_segment(full_source, node)
            assert segment is not None, (
                f"AST source extraction failed for {fn.__name__!r}; "
                f"this should not happen for code defined in full_source"
            )
            return segment
    assert False, f"function {fn.__name__!r} not found in source"


def parse_directive(
    code: str,
    *,
    registry: list,
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
        assert callable(preamble) and callable(sub_reference), (
            f"sub-task {name!r}: 'preamble' and 'sub_reference' must be callable"
        )

        parsed.append({
            "name": name,
            "preamble": preamble,
            "sub_reference": sub_reference,
            "preamble_source": _func_source(preamble, code),
            "sub_reference_source": _func_source(sub_reference, code),
        })

    return parsed
