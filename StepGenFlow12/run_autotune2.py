#!/usr/bin/env python3
"""CLI entry point for autotune2.

Runs the bottom-up tree-DP autotuner against a kernel whose Pass-1 +
Pass-2 results already exist on disk (mirrors ``run.py``'s
``--resume-after-pass1`` pattern). Loads:

  - the plan tree (``<outer_dir>/plan/iteration_<MAX>/tree.json``) via
    ``orchestrator._load_tree_from_dir``
  - each node's verified Pass-1 DSL via
    ``orchestrator._load_verified_dsls``
  - each non-root node's recorded ``Contract`` (pickle dumped by
    pass-1 at ``<outer_dir>/pass1/iteration_<MAX>/<node_path>/contract.pkl``)
  - kernel-level ``dims`` / ``llm_config`` from
    ``<outer_dir>/../config.json`` (the same config the original run
    used)
  - ``hw_config`` from a separate ``autotune_config.json`` (path passed
    via ``--autotune-config``)
  - ``tensors`` via ``precompute_tensors(kernel_name, dims)``

then constructs the analytical scorer + the per-node LLM agent + the
correctness verifier and calls ``src.autotune2.search.autotune``.
Finally promotes the top-K analytical Pareto entries through the rust
simulator and writes ``autotune2_summary.json``.

This script is the **integration seam** between the autotune2 library
(unit-tested under ``tests/test_autotune2_*.py``) and the rest of the
StepGenFlow12 codebase. It contains plumbing only — all algorithmic
work is in ``src/autotune2/``.

Checkpoint isolation
--------------------
Before any work begins, the source checkpoint (``<ts>/config.json`` +
the chosen ``<ts>/<kernel>/outer_N/``) is copied into a fresh
``<checkpoint_base>/<new_ts>/`` directory and all subsequent reads /
writes happen against the copy. The original ``outer_*`` is never
modified. This mirrors ``run.py --resume-after-pass1``'s pattern.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _snapshot_checkpoint_for_autotune2(
    *, src_outer_dir: Path, checkpoint_base: Path | None,
) -> Path:
    """Copy the source checkpoint into a fresh timestamped dir and return
    the new outer-dir path inside it.

    Mirrors ``run.py --resume-after-pass1`` semantics: the autotune2 run
    is self-contained under a new timestamp dir, leaving the original
    ``outer_*`` checkpoint untouched. The copy preserves the original
    layout (``<ts>/<kernel_grouping>/<outer_N>/...``) so the existing
    ``outer_dir.parent.parent / "config.json"`` lookup in
    ``_load_pass1_state`` continues to work.

    Sibling ``outer_*`` directories of the chosen outer are excluded
    from the copy — autotune2 only needs the one outer_N's pass1
    artifacts.

    ``checkpoint_base`` is the parent dir under which the new timestamp
    dir is created. When None, defaults to the source timestamp dir's
    parent (typically the shared ``checkpoints/`` directory).
    """
    src_kernel_dir = src_outer_dir.parent
    src_ts_dir = src_kernel_dir.parent
    assert src_outer_dir.is_dir(), f"outer_dir not found: {src_outer_dir}"
    assert (src_ts_dir / "config.json").exists(), (
        f"expected config.json at {src_ts_dir / 'config.json'}; the "
        "checkpoint layout is "
        "<ts>/<kernel_grouping>/<outer_N>/ — adjust the snapshot helper "
        "if your layout differs."
    )

    base = checkpoint_base if checkpoint_base is not None else src_ts_dir.parent
    new_ts_dir = base / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    assert not new_ts_dir.exists(), (
        f"new autotune2 checkpoint dir already exists: {new_ts_dir}. "
        "Wait a second for a unique timestamp, or pass --checkpoint-dir "
        "to choose a different base."
    )

    def _ignore(path: str, names: list[str]) -> list[str]:
        # Skip outer_* siblings of the chosen outer; autotune2 doesn't
        # need them and they can be large.
        if Path(path).resolve() == src_kernel_dir:
            return [
                n for n in names
                if n.startswith("outer_") and n != src_outer_dir.name
            ]
        return []

    shutil.copytree(src_ts_dir, new_ts_dir, ignore=_ignore)
    print(
        f"autotune2: snapshot {src_ts_dir} -> {new_ts_dir} "
        f"(excluding sibling outer_* dirs)"
    )
    return new_ts_dir / src_kernel_dir.name / src_outer_dir.name


def _function_signature_for(node, parent_contract) -> str:
    """Mirror pass-1's signature construction (orchestrator.py:2243-2249)."""
    if parent_contract is None:
        return "def tiled_reference(dims, tensors):"
    sig_args = ", ".join(parent_contract.arg_names)
    return f"def {node.name}({sig_args}, *, out_shapes):"


def _format_dims_block(dims: dict) -> str:
    return "```json\n" + json.dumps(dims, indent=2) + "\n```"


def _load_pass1_state(
    outer_dir: Path,
    *,
    kernel: str,
    autotune_config_path: Path,
    llm_config: dict,
) -> dict:
    """Read the plan tree + per-node DSLs + per-node contracts from a
    saved outer_dir.

    Returns ``{tree, pass1_dsls, pass1_contracts, dims, tensors,
    hw_config, llm_config, prompt_inputs, system_prompts}``.

    ``llm_config`` is passed in already-resolved (via
    ``src.config_loader.load_llm_config``) so that ``api_key`` and any
    other profile-only fields not embedded in the checkpoint's run
    config are present. The checkpoint's ``config.json`` is only
    consulted for ``dims``.

    The contract loader expects ``contract.pkl`` to exist at
    ``<outer_dir>/pass1/iteration_<MAX>/<child.path>/contract.pkl``,
    written by the post-Pass-1 contract dump in
    ``orchestrator._pass1_walk``. Outer-dirs created before that dump
    was added will fail loudly here; re-running Pass-1 populates the
    pickles.
    """
    from src.autotune2.prompts import build_autotune2_system_prompt
    from src.autotune2.search import NodePromptInputs
    from src.orchestrator import _load_tree_from_dir, _load_verified_dsls
    from src.prompts import _STEP_DSL_PY, _format_tensors_description

    assert outer_dir.is_dir(), f"--outer-dir not found: {outer_dir}"

    # Surrounding config.json (one level up — the per-kernel grouping dir
    # holds the original run.py invocation's config). Used only for
    # ``dims``; ``llm_config`` is passed in by the caller via the shared
    # ``load_llm_config`` profile loader (so api_key etc. is resolved).
    run_config_path = outer_dir.parent.parent / "config.json"
    assert run_config_path.exists(), (
        f"expected run config at {run_config_path} (the original run.py "
        "config json — used for dims). Adjust the loader if your layout "
        "differs."
    )
    run_config = json.loads(run_config_path.read_text())
    dims = run_config["dims"]

    assert autotune_config_path.exists(), (
        f"--autotune-config not found: {autotune_config_path}"
    )
    autotune_config = json.loads(autotune_config_path.read_text())
    assert "hw_config" in autotune_config, (
        f"{autotune_config_path}: missing required 'hw_config' key"
    )
    hw_config = autotune_config["hw_config"]

    # Tree + DSLs + tensors via shared loaders.
    from precompute import precompute_tensors  # StepDB
    tree = _load_tree_from_dir(outer_dir)
    pass1_dsls = _load_verified_dsls(outer_dir)
    tensors = precompute_tensors(kernel, dims)

    # Locate the latest pass1 iteration dir (contract pickles live under it).
    pass1_root = outer_dir / "pass1"
    iter_dirs = sorted(
        (d for d in pass1_root.iterdir()
         if d.is_dir() and d.name.startswith("iteration_")),
        key=lambda d: int(d.name.split("_", 1)[1]),
    )
    assert iter_dirs, f"no pass1 iterations found under {pass1_root}"
    latest_pass1 = iter_dirs[-1]

    pass1_contracts: dict = {}
    for node in tree.iter_topological():
        if node.path == tree.root.path:
            continue
        pkl = latest_pass1 / node.path / "contract.pkl"
        assert pkl.exists(), (
            f"missing contract pickle for node {node.path!r} at {pkl}. "
            "Pass-1 contract dump was added recently; re-run pass1 to "
            "populate it, or implement a contract-rederivation path."
        )
        with open(pkl, "rb") as f:
            pass1_contracts[node.path] = pickle.load(f)

    # Per-node prompt inputs (user-prompt building blocks) and
    # per-node autotune2 system prompts (system prompts are
    # ``is_leaf``-dependent only; we still index by path for symmetry).
    dsl_code = _STEP_DSL_PY.read_text()
    prompt_inputs: dict = {}
    system_prompts: dict = {}
    dims_block = _format_dims_block(dims)
    for node in tree.iter_topological():
        parent_contract = (
            None if node.path == tree.root.path
            else pass1_contracts[node.path]
        )
        prompt_inputs[node.path] = NodePromptInputs(
            function_signature=_function_signature_for(node, parent_contract),
            pytorch_reference=node.reference_code,
            dims_block=dims_block,
            tensors_block=(
                _format_tensors_description(tensors) if tensors else ""
            ),
        )
        system_prompts[node.path] = build_autotune2_system_prompt(
            is_leaf=node.is_leaf, dsl_code=dsl_code,
        )

    return {
        "tree": tree,
        "pass1_dsls": pass1_dsls,
        "pass1_contracts": pass1_contracts,
        "dims": dims,
        "tensors": tensors,
        "hw_config": hw_config,
        "llm_config": llm_config,
        "prompt_inputs": prompt_inputs,
        "system_prompts": system_prompts,
    }


async def _run_autotune2(args: argparse.Namespace) -> int:
    from src.autotune2.compose import make_analytical_scorer
    from src.autotune2.runtime import (
        build_real_agent_fn,
        build_real_verifier_fn,
        build_rust_evaluate_fn,
        promote_top_k,
        write_autotune2_summary,
    )
    from src.autotune2.search import SearchConfig, autotune
    from src.config_loader import load_llm_config

    src_outer_dir = Path(args.outer_dir).resolve()
    checkpoint_base = (
        Path(args.checkpoint_dir).resolve()
        if args.checkpoint_dir else None
    )

    # Derive --kernel / --preset from the source checkpoint's config.json
    # (the same one _load_pass1_state reads for `dims`) when not provided
    # on the CLI. The original run.py invocation stamps both fields, so
    # they are authoritative for this checkpoint.
    if args.kernel is None or args.preset is None:
        src_config_path = src_outer_dir.parent.parent / "config.json"
        assert src_config_path.exists(), (
            f"cannot default --kernel/--preset: expected config.json at "
            f"{src_config_path}"
        )
        src_config = json.loads(src_config_path.read_text())
        if args.kernel is None:
            assert "kernel" in src_config, (
                f"{src_config_path}: missing 'kernel' — pass --kernel"
            )
            args.kernel = src_config["kernel"]
        if args.preset is None:
            assert "preset" in src_config, (
                f"{src_config_path}: missing 'preset' — pass --preset"
            )
            args.preset = src_config["preset"]

    # Snapshot the source checkpoint into a fresh timestamped dir; all
    # autotune2 artifacts (per-turn logs, variants.py, rust _work_dir,
    # summary JSON) are written under this copy so the original
    # outer_* dir is left untouched.
    outer_dir = _snapshot_checkpoint_for_autotune2(
        src_outer_dir=src_outer_dir, checkpoint_base=checkpoint_base,
    )
    ckpt_dir = outer_dir

    # Resolve the llm config the same way run.py does so api_key and any
    # other profile-only fields are filled in (the checkpoint's embedded
    # llm_config blob lacks api_key).
    llm_config = load_llm_config(args.config, args.model)

    state = _load_pass1_state(
        outer_dir,
        kernel=args.kernel,
        autotune_config_path=Path(args.autotune_config).resolve(),
        llm_config=llm_config,
    )

    score_fn = make_analytical_scorer(
        dims=state["dims"], tensors=state["tensors"],
        hw_config=state["hw_config"],
        max_total_compute_bw=args.compute_bw,
    )

    agent_factory = build_real_agent_fn(llm_config=state["llm_config"])
    verifier = build_real_verifier_fn(
        kernel_name=args.kernel,
        dims=state["dims"],
        tensors=state["tensors"],
        check_order=args.check_order,
    )

    result = await autotune(
        plan_tree=state["tree"],
        pass1_dsls=state["pass1_dsls"],
        pass1_contracts=state["pass1_contracts"],
        ckpt_dir=ckpt_dir,
        score_fn=score_fn,
        agent_factory=agent_factory,
        verifier=verifier,
        prompt_inputs=state["prompt_inputs"],
        system_prompts=state["system_prompts"],
        config=SearchConfig(
            max_turns_per_attempt=args.max_turns_per_attempt,
            max_attempts=args.max_attempts,
            check_order=args.check_order,
        ),
    )

    rust_evaluate = build_rust_evaluate_fn(
        work_dir=ckpt_dir / "autotune2" / "_rust_work",
        kernel_name=args.kernel,
        preset=args.preset,
        timing_only=True,
    )
    promotions = promote_top_k(
        root_library=result.root_library(),
        k=args.top_k,
        rust_evaluate_fn=rust_evaluate,
    )

    summary_path = ckpt_dir / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=result, rust_promotions=promotions,
        out_path=summary_path, include_sources=args.include_sources,
    )

    print(f"\nautotune2 summary -> {summary_path}")
    if promotions:
        best = promotions[0]
        print(f"  rust-best entry: cycles={best.rust_cycles} "
              f"(analytical={best.entry.cycles}, "
              f"on_chip={best.entry.on_chip}B, "
              f"provenance={best.entry.provenance!r})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="StepGenFlow autotune2 — bottom-up tree-DP autotuner "
                    "with per-node Pareto libraries + rust top-K promotion",
    )
    parser.add_argument(
        "outer_dir", type=str,
        help="Outer-dir checkpoint to resume from (must contain pass1/ + pass2/ "
             "from a previous run.py invocation).",
    )
    parser.add_argument(
        "--kernel", default=None,
        help="Kernel name (StepDB) — needed for rust simulator wiring. "
             "Defaults to the 'kernel' field in <outer_dir>/../config.json.",
    )
    parser.add_argument(
        "--preset", default=None,
        help="Preset name (StepDB) — needed for rust simulator wiring. "
             "Defaults to the 'preset' field in <outer_dir>/../config.json.",
    )
    parser.add_argument(
        "--autotune-config",
        default="/workspace/NextStep/StepGenFlow12/autotune_config_2.json",
        help="Path to autotune_config.json carrying the 'hw_config' block "
             "(same JSON run.py's --autotune-config consumes). Default: "
             "/workspace/NextStep/StepGenFlow12/autotune_config_2.json.",
    )
    parser.add_argument(
        "--model", default="gpt-oss-120b",
        help="LLM config profile name; resolves to "
             "<NextStep>/configs/<model>.json (default: gpt-oss-120b). "
             "Mirrors run.py.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Explicit path to an LLM config JSON (overrides --model). "
             "Mirrors run.py.",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Base directory under which a fresh '<timestamp>/' run dir is "
             "created; the source checkpoint (config.json + chosen outer_N) "
             "is copied into it and autotune2 writes only there, leaving the "
             "original outer_* untouched. Default: the parent of the source "
             "timestamp dir (typically <repo>/checkpoints/).",
    )
    parser.add_argument(
        "--max-turns-per-attempt", type=int, default=16,
        help="Max LLM turns within a single fresh-conversation attempt "
             "(default: 3). Each turn within an attempt accumulates "
             "gate-failure feedback.",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=5,
        help="Max fresh-conversation attempts per node, after the pass-1 "
             "baseline (default: 5). Use 0 to run only the pass-1 baseline.",
    )
    parser.add_argument(
        "--check-order", default="correctness-first",
        choices=("correctness-first", "compliance-first", "always-both"),
        help="Gate cascade order for the 4-gate verifier (default: "
             "correctness-first).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of root-level Pareto entries to promote through the "
             "rust simulator (default: 3).",
    )
    parser.add_argument(
        "--compute-bw", type=int, default=100_000,
        help="Global compute_bw budget for the analytical scorer (default: "
             "100k — the memory-bound regime per HANDOFF.md).",
    )
    parser.add_argument(
        "--include-sources", action="store_true",
        help="Embed the rust-best composed source string in the summary JSON "
             "(can be megabytes for large kernels).",
    )
    args = parser.parse_args()

    return asyncio.run(_run_autotune2(args))


if __name__ == "__main__":
    sys.exit(main())
