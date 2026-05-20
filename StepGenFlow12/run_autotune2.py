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


def _resolve_attempt_budgets(
    attempt_budgets: list, max_on_chip_memory: int, *, source: str,
) -> list[int | None]:
    """Validate + scale a list of attempt-budget multipliers to absolute
    byte budgets. ``None`` passes through unchanged (unlimited)."""
    assert isinstance(attempt_budgets, list) and attempt_budgets, (
        f"{source}: 'attempt_budgets' must be a non-empty list "
        f"of multipliers (null = unlimited), got {attempt_budgets!r}"
    )
    bytes_out: list[int | None] = []
    for i, m in enumerate(attempt_budgets):
        if m is None:
            bytes_out.append(None)
            continue
        assert isinstance(m, (int, float)) and m > 0, (
            f"{source}: attempt_budgets[{i}]={m!r} must be a "
            f"positive number or null"
        )
        bytes_out.append(int(round(max_on_chip_memory * float(m))))
    return bytes_out


_PASS_KNOBS = (
    "fewshot", "max_baselines_per_node", "baseline_selection",
    "attempt_budgets", "max_turns_per_attempt",
)
_PASS_SPEC_DEFAULTS = {
    "fewshot": "tile_shrink",
    "max_baselines_per_node": 4,
    "baseline_selection": "pareto_diverse",
    "max_turns_per_attempt": 16,
}


def _resolve_pass_specs(
    autotune_config: dict, *, cli_args: argparse.Namespace, source: str,
) -> tuple[list[dict], bool]:
    """Normalize the optional ``passes:[...]`` block into a list of
    fully-resolved pass specs.

    Resolution order per knob: per-pass key > top-level config key >
    CLI arg (where applicable) > library default. Returns
    ``(specs, is_multi_pass)`` where ``is_multi_pass`` reflects whether
    a ``passes`` key was present in the JSON — used to decide whether
    to lay artifacts out under per-pass subdirs.
    """
    max_on_chip_memory = autotune_config["max_on_chip_memory"]
    top_level_budgets = autotune_config["attempt_budgets"]

    # Per-knob fallback chain. CLI args win over library defaults but
    # lose to top-level config (top-level is a config-file decision).
    cli_overrides = {
        "fewshot": cli_args.fewshot,
        "max_turns_per_attempt": cli_args.max_turns_per_attempt,
    }
    config_defaults: dict = {}
    for knob in _PASS_KNOBS:
        if knob == "attempt_budgets":
            continue  # handled separately so byte resolution is per-pass
        if knob in autotune_config:
            config_defaults[knob] = autotune_config[knob]
        elif knob in cli_overrides:
            config_defaults[knob] = cli_overrides[knob]
        else:
            config_defaults[knob] = _PASS_SPEC_DEFAULTS[knob]

    raw_passes = autotune_config.get("passes")
    is_multi_pass = raw_passes is not None
    if not is_multi_pass:
        raw_passes = [{"name": "pass0"}]
    assert isinstance(raw_passes, list) and raw_passes, (
        f"{source}: 'passes' must be a non-empty list when present, "
        f"got {raw_passes!r}"
    )

    specs: list[dict] = []
    for i, p in enumerate(raw_passes):
        assert isinstance(p, dict), (
            f"{source}: passes[{i}] must be a dict, got {type(p).__name__}"
        )
        assert "name" in p and isinstance(p["name"], str) and p["name"], (
            f"{source}: passes[{i}] missing required 'name' string"
        )
        # Validate each per-pass key is a recognized knob.
        for k in p:
            assert k == "name" or k in _PASS_KNOBS, (
                f"{source}: passes[{i}] has unknown key {k!r}; "
                f"recognized: {('name',) + _PASS_KNOBS}"
            )
        spec = {"name": p["name"], **config_defaults}
        for knob in _PASS_KNOBS:
            if knob == "attempt_budgets":
                continue
            if knob in p:
                spec[knob] = p[knob]
        budgets = p.get("attempt_budgets", top_level_budgets)
        spec["attempt_budgets_bytes"] = _resolve_attempt_budgets(
            budgets, max_on_chip_memory,
            source=f"{source} passes[{i}].attempt_budgets",
        )
        specs.append(spec)
    return specs, is_multi_pass


def _build_system_prompts(
    tree, *, dsl_code: str, fewshot: str, max_tile: int | None = None,
) -> dict[str, str]:
    """Build ``{node_path: system_prompt}`` for one fewshot variant.

    Rebuilt per pass when multi-pass fewshots differ, so each pass's
    LLM agent sees the worked-example pack matching its goal.

    ``max_tile`` (when set) routes through ``build_autotune2_system_prompt``
    to inject the same pass-1 max-tile addendum so the autotune2 LLM
    knows about the bound.
    """
    from src.autotune2.prompts import build_autotune2_system_prompt
    return {
        node.path: build_autotune2_system_prompt(
            is_leaf=node.is_leaf, dsl_code=dsl_code, fewshot=fewshot,
            max_tile=max_tile,
        )
        for node in tree.iter_topological()
    }


def _resolve_max_tile(
    pass1_contracts: dict, *, cli_max_tile: int | None,
) -> int | None:
    """Reconcile a CLI ``--max-tile`` value with the bounds baked into
    each loaded Pass-1 ``Contract``.

    Pass-1 stamps every recorded contract with its ``max_tile`` field
    (or ``None`` when max-tile mode was off). All non-root contracts of
    a single run share that value; if the user passes ``--max-tile N``
    on the CLI it must either match the contracts' value, or the
    contracts must be unset (so autotune2 can add a bound that pass-1
    did not impose).

    Returns the resolved max_tile (``None`` if off).
    """
    contract_values = {c.max_tile for c in pass1_contracts.values()}
    assert len(contract_values) <= 1, (
        f"loaded pass-1 contracts disagree on max_tile: {contract_values!r}. "
        "All non-root contracts of one pass-1 run should share the same "
        "value — investigate the checkpoint."
    )
    contract_max_tile = next(iter(contract_values)) if contract_values else None
    if cli_max_tile is None:
        return contract_max_tile
    assert isinstance(cli_max_tile, int) and cli_max_tile >= 1, (
        f"--max-tile must be a positive int, got {cli_max_tile!r}"
    )
    if contract_max_tile is None:
        return cli_max_tile
    assert cli_max_tile == contract_max_tile, (
        f"--max-tile={cli_max_tile} disagrees with the bound baked into "
        f"the loaded pass-1 contracts (max_tile={contract_max_tile}). "
        "Pass-1 ran under a different bound; either rerun pass-1 with the "
        "new bound or drop --max-tile to reuse the existing one."
    )
    return cli_max_tile


def _activate_max_tile_dsl(max_tile: int) -> str:
    """Install ``step_dsl_max_tile`` as ``sys.modules['step_dsl']`` with
    the requested bound, and return the LLM-facing DSL source (with
    ``MAX_TILE_ROW`` / ``MAX_TILE_COL`` substituted to ``max_tile``).

    Mirrors the orchestrator's pass-1 hook
    (``orchestrator._refactor_one_node_pass1``, lines 2450-2454) and the
    pass-1 agent factory's prompt substitution
    (``agents.make_pass1_agent``, lines 530-541) so what the autotune2
    LLM reads is what the runtime enforces.
    """
    assert isinstance(max_tile, int) and max_tile >= 1, (
        f"_activate_max_tile_dsl: max_tile must be positive int, got {max_tile!r}"
    )
    from src import step_dsl_max_tile as _step_dsl_max_tile_mod
    _step_dsl_max_tile_mod.MAX_TILE_ROW = int(max_tile)
    _step_dsl_max_tile_mod.MAX_TILE_COL = int(max_tile)
    if sys.modules.get("step_dsl") is not _step_dsl_max_tile_mod:
        sys.modules["step_dsl"] = _step_dsl_max_tile_mod
    from src.prompts import _STEP_DSL_MAX_TILE_PY
    dsl_code = _STEP_DSL_MAX_TILE_PY.read_text()
    dsl_code = dsl_code.replace(
        "MAX_TILE_ROW = 1\n", f"MAX_TILE_ROW = {int(max_tile)}\n", 1
    )
    dsl_code = dsl_code.replace(
        "MAX_TILE_COL = 1\n", f"MAX_TILE_COL = {int(max_tile)}\n", 1
    )
    assert f"MAX_TILE_ROW = {int(max_tile)}" in dsl_code, (
        "max_tile substitution failed: literal 'MAX_TILE_ROW = 1\\n' not "
        "found in step_dsl_max_tile.py source. Did the default change?"
    )
    return dsl_code


def _load_pass1_state(
    outer_dir: Path,
    *,
    kernel: str,
    autotune_config_path: Path,
    llm_config: dict,
    cli_args: argparse.Namespace,
) -> dict:
    """Read the plan tree + per-node DSLs + per-node contracts from a
    saved outer_dir, and resolve the multi-pass spec from the autotune
    config.

    Returns ``{tree, pass1_dsls, pass1_contracts, dims, tensors,
    hw_config, max_on_chip_memory, dsl_code, llm_config, prompt_inputs,
    pass_specs, is_multi_pass}``. ``pass_specs`` is a list of one (legacy
    single-pass) or N (multi-pass) fully-resolved specs; the runner
    iterates over them.

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
    for required in ("hw_config", "max_on_chip_memory", "attempt_budgets"):
        assert required in autotune_config, (
            f"{autotune_config_path}: missing required {required!r} key"
        )
    hw_config = autotune_config["hw_config"]
    max_on_chip_memory = autotune_config["max_on_chip_memory"]
    assert (
        isinstance(max_on_chip_memory, int) and max_on_chip_memory > 0
    ), (
        f"{autotune_config_path}: 'max_on_chip_memory' must be a positive int "
        f"(bytes), got {max_on_chip_memory!r}"
    )
    pass_specs, is_multi_pass = _resolve_pass_specs(
        autotune_config, cli_args=cli_args, source=str(autotune_config_path),
    )

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

    # Per-node prompt inputs (user-prompt building blocks). System
    # prompts are built per pass by the runner since fewshot may differ
    # across passes; only the inputs that don't depend on fewshot live
    # here.
    dsl_code = _STEP_DSL_PY.read_text()
    prompt_inputs: dict = {}
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

    return {
        "tree": tree,
        "pass1_dsls": pass1_dsls,
        "pass1_contracts": pass1_contracts,
        "dims": dims,
        "tensors": tensors,
        "hw_config": hw_config,
        "max_on_chip_memory": max_on_chip_memory,
        "dsl_code": dsl_code,
        "pass_specs": pass_specs,
        "is_multi_pass": is_multi_pass,
        "llm_config": llm_config,
        "prompt_inputs": prompt_inputs,
    }


async def _run_autotune2(args: argparse.Namespace) -> int:
    from src.autotune2.compose import make_analytical_scorer
    from src.autotune2.runtime import (
        build_real_agent_fn,
        build_real_verifier_factory_fn,
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
        cli_args=args,
    )

    # Reconcile CLI --max-tile against the bound baked into the loaded
    # pass-1 contracts. When on, install step_dsl_max_tile as
    # sys.modules['step_dsl'] (so _exec_dsl_ref / _build_dsl_scaffold see
    # the bounded variant when verifying LLM-emitted code) and rebuild
    # state['dsl_code'] from the substituted source so the autotune2
    # system prompt shows the LLM what the runtime will enforce.
    max_tile = _resolve_max_tile(
        state["pass1_contracts"], cli_max_tile=args.max_tile,
    )
    if max_tile is not None:
        state["dsl_code"] = _activate_max_tile_dsl(max_tile)
        print(f"autotune2: max-tile mode active (MAX_TILE_ROW = "
              f"MAX_TILE_COL = {max_tile})")

    # The scorer's closed-over ``tensors`` dict has to match the per-node
    # arg names the synthetic wrapper references (e.g. ``tensors["Q"]``)
    # — see ``build_synthetic_wrapper_for_node``. The driver rebuilds the
    # scorer (wrapped in an ``AnalyticalOnly`` simulation manager) per
    # node via this factory; the root falls back to the kernel-level
    # dict passed in as ``root_tensors``. PR1 of the simulation-manager
    # work locks the in-loop policy to ``AnalyticalOnly`` (behavior-
    # equivalent to the legacy score_fn path) so the seam is in place
    # without changing what the autotuner does today.
    from src.autotune2.sim_manager import AnalyticalOnly

    def make_sim_manager(node_tensors: dict):
        score_fn = make_analytical_scorer(
            dims=state["dims"], tensors=node_tensors,
            hw_config=state["hw_config"],
            max_total_compute_bw=args.compute_bw,
        )
        return AnalyticalOnly(score_fn=score_fn)

    agent_factory = build_real_agent_fn(llm_config=state["llm_config"])
    # Per-node verifier factory. For the root the closure uses
    # ``compute_gold(args.kernel, dims, tensors)``; for non-root nodes
    # the factory builds a verifier that injects ``parent_contract
    # .tiled_outputs`` as gold and invokes the leaf directly. See
    # build_real_verifier_factory_fn for the per-node behavior.
    make_verifier = build_real_verifier_factory_fn(
        root_kernel=args.kernel,
        dims=state["dims"],
        check_order=args.check_order,
    )

    # Per-node resume stamps — fold in hw_config + compute_bw + check_order
    # so any change to scorer/verifier config invalidates every node's
    # cached library at once (their cycle/on_chip numbers were measured
    # against the old config). Pass-1 DSL and tree-shape changes ripple
    # up transitively inside compute_plan_stamps. Multi-pass uses a
    # cascading ``pass_specs[0..i]`` slice in ``extra`` so editing pass j
    # invalidates pass j and every later pass but leaves earlier passes
    # alone.
    from src.autotune2.persistence import compute_plan_stamps
    base_stamp_extra = {
        "hw_config": state["hw_config"],
        "compute_bw": args.compute_bw,
        "check_order": args.check_order,
        "max_on_chip_memory": state["max_on_chip_memory"],
        # Bumped when output_contracts switched from LLM-declared to
        # graph-derived. Old libraries were keyed by LLM-declared
        # output contracts that may not match the derived (stream+tile,
        # identity-permutation) form, so reuse would produce cell-key
        # collisions. Increment this tag to invalidate again.
        "output_contracts_source": "derived_v1",
    }

    pass_specs = state["pass_specs"]
    is_multi_pass = state["is_multi_pass"]
    print(
        f"autotune2: running {len(pass_specs)} pass(es): "
        + ", ".join(f"{i}={s['name']}({s['fewshot']})"
                    for i, s in enumerate(pass_specs))
    )

    prior_libraries: dict | None = None
    result = None  # type: ignore[assignment]
    for pass_idx, spec in enumerate(pass_specs):
        pass_subdir = (
            f"pass_{pass_idx}_{spec['name']}" if is_multi_pass else None
        )
        # Per-pass stamps cascade: hash of pass_specs[0..pass_idx+1] +
        # base extra. Editing spec j changes the cascade for pass j+,
        # invalidating their snapshots; earlier passes stay valid.
        pass_extra = {
            **base_stamp_extra,
            "attempt_budgets_bytes": spec["attempt_budgets_bytes"],
            "pass_specs": [
                {k: v for k, v in s.items() if k != "attempt_budgets_bytes"}
                for s in pass_specs[: pass_idx + 1]
            ],
        }
        node_stamps = compute_plan_stamps(
            plan_tree=state["tree"],
            pass1_dsls=state["pass1_dsls"],
            pass1_contracts=state["pass1_contracts"],
            extra=pass_extra,
        )
        system_prompts = _build_system_prompts(
            state["tree"], dsl_code=state["dsl_code"], fewshot=spec["fewshot"],
            max_tile=max_tile,
        )
        print(
            f"autotune2: pass {pass_idx} '{spec['name']}' — "
            f"fewshot={spec['fewshot']}, "
            f"max_baselines_per_node={spec['max_baselines_per_node']}, "
            f"baseline_selection={spec['baseline_selection']}"
        )
        result = await autotune(
            plan_tree=state["tree"],
            pass1_dsls=state["pass1_dsls"],
            pass1_contracts=state["pass1_contracts"],
            ckpt_dir=ckpt_dir,
            make_sim_manager=make_sim_manager,
            root_tensors=state["tensors"],
            agent_factory=agent_factory,
            make_verifier=make_verifier,
            prompt_inputs=state["prompt_inputs"],
            system_prompts=system_prompts,
            config=SearchConfig(
                max_turns_per_attempt=spec["max_turns_per_attempt"],
                attempt_budgets_bytes=spec["attempt_budgets_bytes"],
                check_order=args.check_order,
                fewshot=spec["fewshot"],
            ),
            node_stamps=node_stamps,
            initial_libraries=prior_libraries,
            max_baselines_per_node=spec["max_baselines_per_node"],
            baseline_selection=spec["baseline_selection"],
            pass_subdir=pass_subdir,
        )
        prior_libraries = result.libraries
    assert result is not None, "pass loop produced no result"

    rust_evaluate = build_rust_evaluate_fn(
        work_dir=ckpt_dir / "autotune2" / "_rust_work",
        kernel_name=args.kernel,
        preset=args.preset,
        timing_only=True,
        max_total_compute_bw=args.compute_bw,
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
    from src.process_group import setup_process_group
    setup_process_group()
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
             "(default: 16). Each turn within an attempt accumulates "
             "gate-failure feedback. The number of attempts per node is "
             "set by len(attempt_budgets) in the autotune-config JSON.",
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
    parser.add_argument(
        "--fewshot", default="tile_shrink",
        choices=("tile_shrink", "parallel"),
        help="Worked-example pack to inject into the autotune2 system "
             "prompt (default: tile_shrink). 'parallel' swaps in the "
             "shared-vs-independent parallelism examples and uses "
             "autotune2_system_parallel.txt as the template.",
    )
    parser.add_argument(
        "--max-tile", type=int, default=None, metavar="N",
        help="Run autotune2 against step_dsl_max_tile.py with both "
             "MAX_TILE_ROW and MAX_TILE_COL set to N. Mirrors run.py's "
             "--max-tile and applies the same load/store/tile-growing-"
             "reshape/stub-call bounds, plus the pass-1 max-tile "
             "addendum in the autotune2 system prompt. Default: derive "
             "from the bound baked into the loaded pass-1 contracts (so "
             "autotune2 inherits whatever pass-1 used). When set "
             "explicitly, must match the contracts' value, or the "
             "contracts must be unset (so autotune2 can add a bound "
             "pass-1 did not impose).",
    )
    args = parser.parse_args()

    return asyncio.run(_run_autotune2(args))


if __name__ == "__main__":
    sys.exit(main())
