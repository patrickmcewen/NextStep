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
import asyncio
import json
import uuid
from pathlib import Path

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
class SubTaskOutcome:
    name: str
    success: bool
    failure_reason: str | None = None
    sub_reference_source: str | None = None
    preamble_source: str | None = None
    verified_sub_dsl_source: str | None = None


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

    all_passthrough = (
        len(parent_tensors) > 0
        and set(sub_tensors.keys()) == set(parent_tensors.keys())
        and all(
            k in parent_tensors and sub_tensors[k] is parent_tensors[k]
            for k in sub_tensors
        )
    )
    assert not all_passthrough, (
        f"sub-task {name!r}: preamble is a pure passthrough of all parent "
        f"tensors. This is not a valid decomposition — the sub-task must "
        f"operate on a smaller or transformed piece of the parent's "
        f"computation. Either slice parent tensors, compute intermediates, "
        f"or use only a subset of the parent's tensors. If you can't "
        f"meaningfully decompose the kernel, write tiled_reference directly."
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


# ---------------------------------------------------------------------------
# Dispatch layer (T9)
# ---------------------------------------------------------------------------
# These two are seams overridden by the orchestrator (real impl) or tests
# (fakes). Keeping them as module-level callables means we don't have to
# import _run_pass_loop into this module (avoids circular imports) and tests
# can swap them with monkeypatch.

async def _run_pass_loop_for_sub_task(*, agent, name, kernel_name, dims,
                                      sub_tensors, sub_dir, options,
                                      sub_reference_source,
                                      preamble_source,
                                      registry, depth, counter,
                                      llm_config, log):
    """Default impl: orchestrator wires this in via subdivide.set_runners(...)."""
    raise NotImplementedError(
        "subdivide._run_pass_loop_for_sub_task must be wired by the "
        "orchestrator before dispatch_directive is called"
    )


def _make_subdivide_pass_agent(llm_config: dict, options: SubdivideOptions):
    """Default impl: orchestrator wires this in via set_runners(...)."""
    raise NotImplementedError(
        "subdivide._make_subdivide_pass_agent must be wired by the orchestrator"
    )


def set_runners(*, pass_loop_runner: Callable, pass_agent_factory: Callable):
    """Called once by the orchestrator at startup to wire the seams."""
    global _run_pass_loop_for_sub_task, _make_subdivide_pass_agent
    _run_pass_loop_for_sub_task = pass_loop_runner
    _make_subdivide_pass_agent = pass_agent_factory


async def dispatch_directive(parsed_sub_tasks: list, *,
                              dims: dict, parent_tensors: dict,
                              registry: list, depth: int,
                              counter: SubdivideCounter,
                              options: SubdivideOptions,
                              ckpt_dir, llm_config: dict, log) -> dict:
    """Dispatch each sub-task in parallel; collect verified results.

    On all-success, every VerifiedSubTask is appended to ``registry`` (in
    declared order). On any failure, NO sub-task is added — partial successes
    are discarded so the parent doesn't carry half-state.

    Returns ``{"success": bool, "feedback": str | None}``. Counter is debited
    len(parsed_sub_tasks) regardless of outcome.
    """
    # Debit counter up front so a failed dispatch can't be retried into the same
    # slot — the counter is not a refundable resource.
    counter.used += len(parsed_sub_tasks)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    coros = [
        _run_one_sub_task(
            parsed=parsed, dims=dims, parent_tensors=parent_tensors,
            registry=registry, depth=depth, counter=counter,
            options=options, ckpt_dir=ckpt_dir,
            llm_config=llm_config, log=log,
        )
        for parsed in parsed_sub_tasks
    ]
    outcomes = await asyncio.gather(*coros, return_exceptions=False)

    summary = {
        "sub_tasks": [
            {
                "name": o.name,
                "status": "ok" if o.success else "failed",
                "failure_reason": o.failure_reason,
            }
            for o in outcomes
        ]
    }
    (ckpt_dir / "subdivide_result.json").write_text(
        json.dumps(summary, indent=2)
    )

    if all(o.success for o in outcomes):
        for o in outcomes:
            registry.append(VerifiedSubTask(
                name=o.name,
                sub_reference_source=o.sub_reference_source,
                preamble_source=o.preamble_source,
                verified_sub_dsl_source=o.verified_sub_dsl_source,
            ))
        return {"success": True, "feedback": None}

    failure_lines = ["## Subdivide directive: one or more sub-tasks failed"]
    for o in outcomes:
        if o.success:
            failure_lines.append(
                f"- sub-task {o.name!r}: succeeded (discarded due to "
                f"sibling failure)"
            )
        else:
            failure_lines.append(
                f"- sub-task {o.name!r}: {o.failure_reason}"
            )
    failure_lines.append(
        "\nReconsider the decomposition or implement the work directly. "
        "Successful sub-tasks above were NOT added to the registry — you "
        "must re-emit them with a fresh directive (or different names) if "
        "you want to retry."
    )
    return {"success": False, "feedback": "\n".join(failure_lines)}


async def _run_one_sub_task(*, parsed: dict, dims: dict, parent_tensors: dict,
                             registry: list, depth: int,
                             counter: SubdivideCounter,
                             options: SubdivideOptions, ckpt_dir: Path,
                             llm_config: dict, log) -> SubTaskOutcome:
    name = parsed["name"]
    sub_dir = ckpt_dir / f"sub_{name}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    try:
        prepared = prepare_sub_task(parsed, dims=dims, parent_tensors=parent_tensors)
    except AssertionError as exc:
        return SubTaskOutcome(
            name=name, success=False,
            failure_reason=f"preparation failed: {exc}",
        )

    synth_kernel = f"__sub_{name}_{uuid.uuid4().hex[:8]}__"
    from src.orchestrator import _inject_gold  # lazy: avoids circular import
    _inject_gold(synth_kernel, dims, prepared.sub_gold)

    agent = _make_subdivide_pass_agent(llm_config, options)
    result = await _run_pass_loop_for_sub_task(
        agent=agent, name=name, kernel_name=synth_kernel, dims=dims,
        sub_tensors=prepared.sub_tensors, sub_dir=sub_dir, options=options,
        sub_reference_source=prepared.sub_reference_source,
        preamble_source=prepared.preamble_source,
        registry=registry, depth=depth + 1, counter=counter,
        llm_config=llm_config, log=log,
    )

    if result["success"]:
        (sub_dir / "verified_sub_dsl.py").write_text(result["code"])
        return SubTaskOutcome(
            name=name, success=True,
            sub_reference_source=prepared.sub_reference_source,
            preamble_source=prepared.preamble_source,
            verified_sub_dsl_source=result["code"],
        )
    return SubTaskOutcome(
        name=name, success=False,
        failure_reason=(
            f"refactor pass exhausted {options.max_subdivide_turns} turns "
            f"without producing a verified DSL"
        ),
    )
