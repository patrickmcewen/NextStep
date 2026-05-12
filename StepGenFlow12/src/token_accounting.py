"""Token accounting for StepGenFlow runs.

Two pieces:

1. ``write_turn_tokens(turn_dir, usage, kind)`` — called at every LLM call site
   to persist that call's token usage as a small JSON file (e.g.
   ``turn_0/tokens.json`` for the main turn, ``turn_0/judge_tokens.json`` for
   the inline judge). Each file records prompt / completion / reasoning / total
   counts plus the request count, so per-call costs survive crashes.

2. ``summarize(ckpt_root)`` — walks the checkpoint directory at end-of-pipeline
   and rolls those per-turn files up into a nested dict that mirrors the
   filesystem hierarchy. The top-level grand total + per-outer/per-phase
   breakdowns land in ``result.json["tokens"]``.

The aggregator is purely directory-driven: any new LLM call site only needs to
write its own ``*_tokens.json`` to be counted. There is no in-memory accumulator
to keep in sync — disk is the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Filenames we recognize as per-call token records.
_TOKEN_FILENAMES = ("tokens.json", "judge_tokens.json")


def _zero() -> dict[str, int]:
    return {"requests": 0, "prompt": 0, "completion": 0, "reasoning": 0, "total": 0}


def _accumulate(into: dict[str, int], add: dict[str, int]) -> None:
    for k in ("requests", "prompt", "completion", "reasoning", "total"):
        into[k] += add.get(k, 0)


def write_turn_tokens(turn_dir: Path, usage, kind: str = "main") -> None:
    """Persist one LLM call's token usage into ``turn_dir``.

    ``usage`` is the ``agents.usage.Usage`` object from
    ``RunResult.context_wrapper.usage`` (may be ``None`` when the provider
    didn't return usage info — we still write a zero record so the aggregator
    knows a call happened).

    ``kind`` is ``"main"`` for the primary Runner.run of a turn, ``"judge"``
    for an inline judge call sharing the same ``turn_dir``. The file written is
    ``tokens.json`` or ``judge_tokens.json`` respectively.
    """
    assert kind in ("main", "judge"), f"unknown token kind {kind!r}"
    filename = "tokens.json" if kind == "main" else "judge_tokens.json"

    if usage is None:
        record = _zero()
        record["requests"] = 1  # a call happened, we just have no token detail
    else:
        reasoning = 0
        details = getattr(usage, "output_tokens_details", None)
        if details is not None and getattr(details, "reasoning_tokens", None):
            reasoning = details.reasoning_tokens
        record = {
            "requests": getattr(usage, "requests", 0) or 0,
            "prompt": getattr(usage, "input_tokens", 0) or 0,
            "completion": getattr(usage, "output_tokens", 0) or 0,
            "reasoning": reasoning,
            "total": getattr(usage, "total_tokens", 0) or 0,
        }

    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / filename).write_text(json.dumps(record, indent=2))


def _read_record(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text())
    return {
        "requests": int(raw.get("requests", 0)),
        "prompt": int(raw.get("prompt", 0)),
        "completion": int(raw.get("completion", 0)),
        "reasoning": int(raw.get("reasoning", 0)),
        "total": int(raw.get("total", 0)),
    }


def _summarize_dir(d: Path) -> dict[str, Any]:
    """Recursively summarize a directory.

    Returns a node dict ``{"total": {...}, "<child>": {...}, ...}``. A directory
    with no token files anywhere underneath returns ``None`` so the caller can
    skip wiring it into the tree (avoids polluting the summary with every
    bookkeeping subdir like ``pass2/`` or ``translate/turn_0/`` when those
    contain no LLM calls).
    """
    total = _zero()
    children: dict[str, Any] = {}

    # Per-turn directories contain {tokens.json, judge_tokens.json} but no
    # further hierarchy we care about. Aggregate them flat.
    own_records: list[dict[str, int]] = []
    for name in _TOKEN_FILENAMES:
        p = d / name
        if p.is_file():
            own_records.append(_read_record(p))

    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        child_summary = _summarize_dir(sub)
        if child_summary is None:
            continue
        children[sub.name] = child_summary
        _accumulate(total, child_summary["total"])

    for rec in own_records:
        _accumulate(total, rec)

    if total["total"] == 0 and total["requests"] == 0 and not children:
        return None  # nothing here — let the caller prune

    node: dict[str, Any] = {"total": total}
    node.update(children)
    return node


def summarize(ckpt_root: Path) -> dict[str, Any]:
    """Roll up every ``*_tokens.json`` under ``ckpt_root`` into a nested dict.

    The result mirrors the filesystem: outer dirs contain plan / pass1 /
    refactor_final / translator subtrees; pass1 contains iteration_N → node →
    [attempt_K →] refactor_final → turn_N; plan contains iteration_N → turns →
    node → turn_N. Every level has its own ``"total"`` aggregate; leaf turn
    dirs are absent from the tree (their tokens are folded into the parent),
    keeping the summary compact while still preserving per-attempt and
    per-iteration breakdowns.

    Returns ``{"total": {...zeros}}`` for a tree with no token records (e.g.
    a run that crashed before any LLM call).
    """
    summary = _summarize_dir(Path(ckpt_root))
    if summary is None:
        return {"total": _zero()}
    return summary


def _format_int(n: int) -> str:
    return f"{n:,}"


def _print_tree(node: dict, name: str, depth: int, max_depth: int) -> None:
    indent = "  " * depth
    t = node["total"]
    print(f"{indent}{name}: {_format_int(t['total'])} "
          f"(prompt={_format_int(t['prompt'])}, "
          f"completion={_format_int(t['completion'])}, "
          f"reasoning={_format_int(t['reasoning'])}, "
          f"requests={t['requests']})")
    if max_depth is not None and depth >= max_depth:
        return
    for child_name, child in node.items():
        if child_name == "total":
            continue
        _print_tree(child, child_name, depth + 1, max_depth)


def main() -> None:
    """CLI: ``python -m src.token_accounting <ckpt_dir> [--depth N] [--json]``.

    Walks any checkpoint directory (in-progress or finished) and prints the
    rolled-up token hierarchy. ``--depth`` limits nesting (default: unlimited);
    ``--json`` dumps the raw dict instead of a tree.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ckpt_dir", help="Path to a checkpoint root (or any subdir)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Max nesting depth to print (default: unlimited)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of an indented tree")
    args = parser.parse_args()

    root = Path(args.ckpt_dir)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    summary = summarize(root)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_tree(summary, root.name or str(root), 0, args.depth)


if __name__ == "__main__":
    main()
