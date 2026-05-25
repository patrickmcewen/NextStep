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
  - ``hw_config`` from a separate autotune config file (path passed
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

from src.autotune_config_loader import load_autotune_config


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
    "attempt_budgets", "time_limit_seconds", "ace_context_enabled",
    "ace_refresh_interval_turns", "rust_sim_timeout_seconds",
    "rust_timeout_cycles",
)
_PASS_SPEC_DEFAULTS = {
    "fewshot": "tile_shrink",
    "max_baselines_per_node": 4,
    "baseline_selection": "pareto_diverse",
    "time_limit_seconds": 1800.0,
    "ace_context_enabled": False,
    "ace_refresh_interval_turns": 4,
    "rust_sim_timeout_seconds": 1800,
    "rust_timeout_cycles": 10**15,
}

_STAMP_EXCLUDED_PASS_SPEC_KEYS = frozenset({
    "time_limit_seconds",
    "ace_context_enabled",
    "ace_refresh_interval_turns",
    "rust_sim_timeout_seconds",
    "rust_timeout_cycles",
})


def _stamp_pass_spec_payload(spec: dict) -> dict:
    """Return the pass-spec subset that should invalidate completed nodes.

    Operational resume knobs such as wall-clock time limit and ACE refresh
    cadence affect how much additional search can happen after resume, but
    they do not change whether an already-written library snapshot is
    semantically valid. Keeping them out of the stamp prevents a completed
    node from rerunning just because the user resumes with a shorter time
    budget.
    """
    return {
        k: v for k, v in spec.items()
        if k not in _STAMP_EXCLUDED_PASS_SPEC_KEYS
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
        "time_limit_seconds": cli_args.time_limit_seconds,
        "ace_context_enabled": getattr(cli_args, "ace_context", False),
        "ace_refresh_interval_turns": getattr(
            cli_args, "ace_refresh_interval_turns", 4,
        ),
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
        assert isinstance(spec["time_limit_seconds"], (int, float)) and (
            spec["time_limit_seconds"] > 0
        ), (
            f"{source} passes[{i}].time_limit_seconds must be a positive "
            f"number, got {spec['time_limit_seconds']!r}"
        )
        spec["time_limit_seconds"] = float(spec["time_limit_seconds"])
        assert isinstance(spec["ace_context_enabled"], bool), (
            f"{source} passes[{i}].ace_context_enabled must be a bool, "
            f"got {spec['ace_context_enabled']!r}"
        )
        assert isinstance(spec["ace_refresh_interval_turns"], int) and (
            spec["ace_refresh_interval_turns"] >= 1
        ), (
            f"{source} passes[{i}].ace_refresh_interval_turns must be an "
            f"integer >= 1, got {spec['ace_refresh_interval_turns']!r}"
        )
        assert spec["rust_sim_timeout_seconds"] is None or (
            isinstance(spec["rust_sim_timeout_seconds"], (int, float))
            and spec["rust_sim_timeout_seconds"] > 0
        ), (
            f"{source} passes[{i}].rust_sim_timeout_seconds must be null "
            f"or a positive number, got "
            f"{spec['rust_sim_timeout_seconds']!r}"
        )
        if spec["rust_sim_timeout_seconds"] is not None:
            spec["rust_sim_timeout_seconds"] = float(
                spec["rust_sim_timeout_seconds"]
            )
        assert (
            isinstance(spec["rust_timeout_cycles"], int)
            and spec["rust_timeout_cycles"] > 0
        ), (
            f"{source} passes[{i}].rust_timeout_cycles must be a positive "
            f"int, got {spec['rust_timeout_cycles']!r}"
        )
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
    autotune_config_name: str | None,
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

    autotune_config = load_autotune_config(
        autotune_config_path, autotune_config_name,
    )
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
        final_pick,
        promote_top_k,
        write_autotune2_summary,
    )
    from src.autotune2.ace_context import AceContextConfig, AceContextManager
    from src.autotune2.ace_curator import build_ace_context_refresh_fn
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

    # Install stdio redirect so the very chatty per-turn output (LLM
    # usage lines, "Saved <node>.npy" prints, rust simulator stdout, etc.)
    # lands in <snapshot>/autotune2.log instead of the user's terminal.
    # No-op when stdout is not a TTY (e.g. driven by an outer harness).
    from src.log_redirect import redirect_stdio_to, terminal_print
    snapshot_ts_dir = outer_dir.parent.parent
    log_path = snapshot_ts_dir / "autotune2.log"
    _redirected = redirect_stdio_to(log_path)
    if _redirected:
        terminal_print(f"autotune2 log -> {log_path}")

    # Resolve the llm config the same way run.py does so api_key and any
    # other profile-only fields are filled in (the checkpoint's embedded
    # llm_config blob lacks api_key).
    llm_config = load_llm_config(args.config, args.model)

    state = _load_pass1_state(
        outer_dir,
        kernel=args.kernel,
        autotune_config_path=Path(args.autotune_config).resolve(),
        autotune_config_name=args.autotune_config_name,
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
    # scorer per node via this factory; the root falls back to the
    # kernel-level dict passed in as ``root_tensors``.
    #
    # ``--sim-mode`` selects which simulation manager wraps the scorer:
    #   * ``analytical`` (default): ``AnalyticalOnly`` — every variant
    #     scored by the STeP timing model. Behavior-equivalent to the
    #     pre-PR1 score_fn path.
    #   * ``rust``: ``RustAll`` — every variant additionally rust-
    #     evaluated until the per-pass time limit runs out, with paired
    #     (analytical, rust) records appended to the calibration store.
    from src.autotune2.agent_telemetry import (
        AgentDecisionStore,
        write_sim_manager_system_prompts,
    )
    from src.autotune2.calibration import (
        CalibrationOverlayStore,
        CalibrationStore,
        hw_config_hash,
    )
    from src.autotune2.sim_manager import (
        AgentManager, AnalyticalOnly, DeterministicSplit, RustAll, TimeBudget,
    )

    assert args.sim_mode in (
        "analytical", "rust", "deterministic-split", "agent",
    ), (
        f"--sim-mode must be one of analytical/rust/deterministic-split/"
        f"agent, got {args.sim_mode!r}"
    )
    assert args.curation_max_candidates >= 1, (
        f"--curation-max-candidates must be >= 1, got "
        f"{args.curation_max_candidates!r}"
    )
    assert 1 <= args.curation_k <= args.curation_max_candidates, (
        f"--curation-k must be in [1, --curation-max-candidates="
        f"{args.curation_max_candidates}], got {args.curation_k!r}"
    )

    if args.sim_calibration_path:
        calibration_path = Path(args.sim_calibration_path).resolve()
    else:
        calibration_path = ckpt_dir / "autotune2" / "calibration.jsonl"
    calibration_append_store = CalibrationStore(path=calibration_path)
    seed_stores = []
    if args.sim_calibration_seed_path:
        seed_path = Path(args.sim_calibration_seed_path).resolve()
        if seed_path != calibration_path:
            seed_stores.append(CalibrationStore(path=seed_path))
    calibration_store = CalibrationOverlayStore(
        seed_stores=seed_stores,
        append_store=calibration_append_store,
    )
    calibration_sources_dir = calibration_path.parent / "calibration_sources"
    hw_hash = hw_config_hash(state["hw_config"])
    run_id = ckpt_dir.name  # the new timestamp dir uniquely identifies this run

    # Agent-decision telemetry — one row per AgentManager._decide call
    # and per final_pick_agent invocation, sibling to calibration.jsonl
    # so post-hoc audits can join the two stores by composed_source_hash.
    # Only wired when an LLM-backed path is active (sim-mode=agent or
    # root-pick=agent); analytical / rust / deterministic-split modes
    # don't make any agent decisions to log.
    needs_agent_telemetry = (
        args.sim_mode == "agent" or args.root_pick == "agent"
    )
    if needs_agent_telemetry:
        agent_decisions_path = ckpt_dir / "autotune2" / "agent_decisions.jsonl"
        agent_decision_store = AgentDecisionStore(path=agent_decisions_path)
    else:
        agent_decision_store = None

    # Built up-front (not after the autotune loop) so the RustAll factory
    # below can capture it. ``promote_top_k`` reuses the same closure.
    rust_evaluate = build_rust_evaluate_fn(
        work_dir=ckpt_dir / "autotune2" / "_rust_work",
        kernel_name=args.kernel,
        preset=args.preset,
        tensors=state["tensors"],
        timing_only=not args.rust_functional_check,
        max_total_compute_bw=args.compute_bw,
    )
    current_rust_timeout = {
        "seconds": None,
        "cycles": 10**15,
    }

    def root_rust_evaluate(
        composed_source: str,
        **kwargs,
    ):
        return rust_evaluate(
            composed_source,
            rust_sim_timeout_seconds=current_rust_timeout["seconds"],
            rust_timeout_cycles=current_rust_timeout["cycles"],
            **kwargs,
        )

    # Several modes need a per-target candidate-fetcher and one or more
    # of {curation, decision, final-pick} agents. Build them up-front so
    # every consumer captures the same callables.
    #   * ``--sim-mode=agent``: curation + decision agents (AgentManager).
    #   * ``--root-pick=agent``: curation + final-pick agents
    #     (final_pick_agent).
    # The candidate-fetcher is shared by both — it reads the calibration
    # store filtered by hw_config_hash. The curation agent likewise is
    # shared whenever either mode needs it.
    curation_agent_fn = None
    decision_agent_fn = None
    final_pick_agent_fn = None
    fetch_candidates_fn = None
    needs_curation = (
        args.sim_mode == "agent" or args.root_pick == "agent"
    )
    if needs_curation:
        from src.agents import make_curation_agent
        from src.autotune2.prompts import (
            CurationCandidate,
            build_curation_system_prompt,
            build_sim_decision_system_prompt,
        )
        from agents import ReasoningItem, Runner
        from src.autotune2.search import AgentResponse

        if args.sim_mode == "agent":
            write_sim_manager_system_prompts(
                ckpt_dir / "autotune2",
                curator_system_prompt=build_curation_system_prompt(),
                sim_manager_system_prompt=build_sim_decision_system_prompt(),
            )

        def _make_agent_call(agent):
            async def call(conversation: list) -> AgentResponse:
                result = await Runner.run(agent, conversation)
                reasoning_chunks: list[str] = []
                for item in result.new_items:
                    if isinstance(item, ReasoningItem):
                        for summary in item.raw_item.summary:
                            reasoning_chunks.append(summary.text)
                return AgentResponse(
                    text=result.final_output or "",
                    reasoning="\n\n".join(reasoning_chunks),
                    usage=result.context_wrapper.usage,
                )
            return call

        curation_agent = make_curation_agent(state["llm_config"])
        curation_agent_fn = _make_agent_call(curation_agent)

        def fetch_candidates_fn(_target_source: str) -> list:
            """Pull calibration records relevant to ``_target_source``.

            Steps: filter by hw_config_hash, collapse to one record per
            ``(source-hash, kernel, preset)`` keeping the most recent
            (the calibration store is append-only, so the same triple
            can legitimately appear more than once — e.g. a baseline
            re-rust-evaluated across passes), then cap at
            ``args.curation_max_candidates`` so the curation prompt
            stays bounded. Per the PR4 design discussion we
            intentionally skip semantic prefiltering — the curation
            agent is the prefilter.

            ``record_id`` includes kernel + preset so cross-preset rows
            with the same source-hash (StepDB stores one row per preset
            for each shared step_impl.py) survive as distinct candidates.
            Same source under different presets is real evidence about
            how the analytical-vs-rust gap scales with workload.
            """
            latest_by_id: dict[str, CurationCandidate] = {}
            for r in calibration_store.iter_records(hw_config_hash=hw_hash):
                src_path = Path(r.composed_source_path)
                if not src_path.exists():
                    # Stale record (sources_dir got cleaned, JSONL kept).
                    # Skip rather than crash — the agent's job is to rank
                    # whatever is intact today.
                    continue
                record_id = f"{src_path.stem}__{r.kernel}__{r.preset}"
                latest_by_id[record_id] = CurationCandidate(
                    record_id=record_id,
                    composed_source=src_path.read_text(encoding="utf-8"),
                    analytical_cycles=r.analytical_cycles,
                    rust_cycles=r.rust_cycles,
                    kernel=r.kernel,
                    preset=r.preset,
                )
            return list(latest_by_id.values())[: args.curation_max_candidates]

        if args.sim_mode == "agent":
            from src.agents import make_sim_decision_agent
            decision_agent = make_sim_decision_agent(state["llm_config"])
            decision_agent_fn = _make_agent_call(decision_agent)

        if args.root_pick == "agent":
            from src.agents import make_final_pick_agent
            final_pick_agent_inst = make_final_pick_agent(state["llm_config"])
            final_pick_agent_fn = _make_agent_call(final_pick_agent_inst)

    def make_sim_manager(node_tensors: dict):
        # Each autotune node gets its own simulator time budget. Sibling
        # nodes still run in parallel, but a long-running leaf can no
        # longer consume the rust/agent budget that a parent should get
        # after its children finish.
        node_time_budget = TimeBudget(total_seconds=None)
        score_fn = make_analytical_scorer(
            dims=state["dims"], tensors=node_tensors,
            hw_config=state["hw_config"],
            max_total_compute_bw=args.compute_bw,
        )
        def node_rust_evaluate(
            composed_source: str,
            *,
            node_path: str | None = None,
            run_label: str | None = None,
        ):
            return rust_evaluate(
                composed_source,
                tensors_override=node_tensors,
                node_path=node_path,
                run_label=run_label,
                rust_sim_timeout_seconds=current_rust_timeout["seconds"],
                rust_timeout_cycles=current_rust_timeout["cycles"],
            )

        if args.sim_mode == "analytical":
            return AnalyticalOnly(score_fn=score_fn)
        if args.sim_mode == "rust":
            return RustAll(
                score_fn=score_fn,
                rust_evaluate_fn=node_rust_evaluate,
                time_budget=node_time_budget,
                calibration_store=calibration_store,
                sources_dir=calibration_sources_dir,
                kernel=args.kernel,
                preset=args.preset,
                hw_config_hash=hw_hash,
                compute_bw=args.compute_bw,
                run_id=run_id,
            )
        if args.sim_mode == "deterministic-split":
            return DeterministicSplit(
                score_fn=score_fn,
                rust_evaluate_fn=node_rust_evaluate,
                time_budget=node_time_budget,
                calibration_store=calibration_store,
                sources_dir=calibration_sources_dir,
                kernel=args.kernel,
                preset=args.preset,
                hw_config_hash=hw_hash,
                compute_bw=args.compute_bw,
                run_id=run_id,
            )
        # agent
        return AgentManager(
            score_fn=score_fn,
            rust_evaluate_fn=node_rust_evaluate,
            time_budget=node_time_budget,
            calibration_store=calibration_store,
            sources_dir=calibration_sources_dir,
            kernel=args.kernel,
            preset=args.preset,
            hw_config_hash=hw_hash,
            compute_bw=args.compute_bw,
            run_id=run_id,
            curation_agent_fn=curation_agent_fn,
            decision_agent_fn=decision_agent_fn,
            fetch_candidates_fn=fetch_candidates_fn,
            max_curation_candidates=args.curation_max_candidates,
            curation_k=args.curation_k,
            telemetry_store=agent_decision_store,
        )

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
        # Folding ``sim_mode`` into the stamp invalidates every node's
        # cached library when the user toggles --sim-mode — analytical
        # and rust cycle counts are not directly comparable, and a
        # mixed-source library would silently merge incomparable
        # numbers in the next pass's Pareto admission.
        "sim_mode": args.sim_mode,
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
    ace_context_refresh_fn = None
    if any(spec["ace_context_enabled"] for spec in pass_specs):
        from agents import ReasoningItem, Runner
        from src.agents import make_ace_context_curator_agent
        from src.autotune2.prompts import build_ace_context_curator_system_prompt
        from src.autotune2.search import AgentResponse

        ace_agent = make_ace_context_curator_agent(state["llm_config"])

        async def ace_agent_call(conversation: list) -> AgentResponse:
            result = await Runner.run(ace_agent, conversation)
            reasoning_chunks: list[str] = []
            for item in result.new_items:
                if isinstance(item, ReasoningItem):
                    for summary in item.raw_item.summary:
                        reasoning_chunks.append(summary.text)
            return AgentResponse(
                text=result.final_output or "",
                reasoning="\n\n".join(reasoning_chunks),
                usage=result.context_wrapper.usage,
            )

        ace_context_refresh_fn = build_ace_context_refresh_fn(ace_agent_call)
        ace_prompt_dir = ckpt_dir / "autotune2" / "ace_context"
        ace_prompt_dir.mkdir(parents=True, exist_ok=True)
        (ace_prompt_dir / "curator_system_prompt.txt").write_text(
            build_ace_context_curator_system_prompt()
        )

    for pass_idx, spec in enumerate(pass_specs):
        current_rust_timeout["seconds"] = spec["rust_sim_timeout_seconds"]
        current_rust_timeout["cycles"] = spec["rust_timeout_cycles"]
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
                _stamp_pass_spec_payload({
                    k: v for k, v in s.items()
                    if k != "attempt_budgets_bytes"
                })
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
        ace_context = None
        if spec["ace_context_enabled"]:
            assert ace_context_refresh_fn is not None, (
                "ACE context enabled but curator refresh function was not built"
            )
            playbook_path = (
                ckpt_dir / "autotune2" / "ace_context"
                / f"pass_{pass_idx}_{spec['name']}_{spec['fewshot']}.md"
            )
            ace_context = AceContextManager(AceContextConfig(
                enabled=True,
                refresh_interval_turns=spec["ace_refresh_interval_turns"],
                playbook_path=playbook_path,
            ), refresh_fn=ace_context_refresh_fn)
        print(
            f"autotune2: pass {pass_idx} '{spec['name']}' — "
            f"fewshot={spec['fewshot']}, "
            f"max_baselines_per_node={spec['max_baselines_per_node']}, "
            f"baseline_selection={spec['baseline_selection']}, "
            f"sim_mode={args.sim_mode}, "
            f"time_limit_seconds={spec['time_limit_seconds']}, "
            f"ace_context_enabled={spec['ace_context_enabled']}, "
            f"ace_refresh_interval_turns="
            f"{spec['ace_refresh_interval_turns']}"
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
                time_limit_seconds=spec["time_limit_seconds"],
                attempt_budgets_bytes=spec["attempt_budgets_bytes"],
                check_order=args.check_order,
                fewshot=spec["fewshot"],
                ace_context=ace_context,
            ),
            node_stamps=node_stamps,
            initial_libraries=prior_libraries,
            max_baselines_per_node=spec["max_baselines_per_node"],
            baseline_selection=spec["baseline_selection"],
            pass_subdir=pass_subdir,
        )
        prior_libraries = result.libraries
    assert result is not None, "pass loop produced no result"

    if args.root_pick == "final_pick":
        promotions = final_pick(
            root_library=result.root_library(),
            rust_evaluate_fn=root_rust_evaluate,
            strategy="min_cycles",
        )
        root_pick_strategy = "final_pick:min_cycles"
    elif args.root_pick == "agent":
        from src.autotune2.runtime import final_pick_agent
        assert curation_agent_fn is not None and final_pick_agent_fn is not None, (
            "--root-pick=agent: curation + final-pick agent callables were "
            "not constructed; the up-front agent-helper block should have "
            "built them when args.root_pick == 'agent'."
        )
        promotions = await final_pick_agent(
            root_library=result.root_library(),
            rust_evaluate_fn=root_rust_evaluate,
            curation_agent_fn=curation_agent_fn,
            final_pick_agent_fn=final_pick_agent_fn,
            fetch_candidates_fn=fetch_candidates_fn,
            root_path=result.root_path,
            kernel=args.kernel,
            preset=args.preset,
            curation_k=args.curation_k,
            curation_max_candidates=args.curation_max_candidates,
            telemetry_store=agent_decision_store,
            hw_config_hash=hw_hash,
            run_id=run_id,
        )
        root_pick_strategy = "final_pick:agent"
    else:
        promotions = promote_top_k(
            root_library=result.root_library(),
            k=args.top_k,
            rust_evaluate_fn=root_rust_evaluate,
        )
        root_pick_strategy = f"top_k:{args.top_k}"

    summary_path = ckpt_dir / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=result, rust_promotions=promotions,
        out_path=summary_path, include_sources=args.include_sources,
        root_pick_strategy=root_pick_strategy,
    )

    print(f"\nautotune2 summary -> {summary_path}")
    if promotions:
        best = promotions[0]
        print(f"  rust-best entry: cycles={best.rust_cycles} "
              f"(analytical={best.entry.cycles}, "
              f"on_chip={best.entry.on_chip}B, "
              f"provenance={best.entry.provenance!r})")
    if _redirected:
        terminal_print(f"autotune2 done — summary: {summary_path}")
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
        default="/workspace/NextStep/StepGenFlow12/autotune_configs.yaml",
        help="Path to autotune config JSON/YAML carrying the 'hw_config' block "
             "(same file run.py's --autotune-config consumes). Default: "
             "/workspace/NextStep/StepGenFlow12/autotune_configs.yaml.",
    )
    parser.add_argument(
        "--autotune-config-name",
        default="autotune_config_2",
        help="Named config to resolve when --autotune-config is YAML.",
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
        "--time-limit-seconds", type=float, default=1800.0,
        help="Wall-clock limit for each autotuner pass (default: 1800). "
             "The search loop starts no new LLM turns after the pass "
             "deadline expires. Per-pass config entries override this.",
    )
    parser.add_argument(
        "--ace-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ACE-style context refresh for autotune2 attempts. "
             "Each lane starts a new logged session after "
             "--ace-refresh-interval-turns completed turns; per-pass "
             "config entries may override this.",
    )
    parser.add_argument(
        "--ace-refresh-interval-turns", type=int, default=10,
        help="Completed turns per lane before the shared ACE context "
             "manager refreshes the playbook and starts a new session "
             "(default: 4).",
    )
    parser.add_argument(
        "--check-order", default="correctness-first",
        choices=("correctness-first", "compliance-first", "always-both"),
        help="Gate cascade order for the 4-gate verifier (default: "
             "correctness-first).",
    )
    parser.add_argument(
        "--root-pick", default="final_pick",
        choices=("final_pick", "top_k", "agent"),
        help="How to choose the root-level variant(s) sent to the rust "
             "simulator at end of run (default: final_pick). 'final_pick' "
             "picks one entry from the root Pareto (lowest cycles, "
             "restricted to rust-sourced entries when the library is "
             "mixed) and rust-evaluates only that one — the resulting "
             "number is comparable across sim modes per HANDOFF design "
             "decision #6. 'agent' lets the FinalPickAgent choose one "
             "entry from the root Pareto given per-entry curated "
             "calibration evidence; falls back to the 'final_pick' "
             "deterministic pick on any agent failure. 'top_k' uses the "
             "legacy promote_top_k(k=--top-k) path for backward "
             "compatibility with existing benchmark scripts.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of root-level Pareto entries to promote through the "
             "rust simulator (default: 3). Only consulted when "
             "--root-pick=top_k.",
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
        choices=("tile_shrink", "parallel", "general"),
        help="Worked-example pack to inject into the autotune2 system "
             "prompt (default: tile_shrink). 'parallel' focuses on "
             "shared-vs-independent parallelism; 'general' combines "
             "tile-shrink and parallelism guidance.",
    )
    parser.add_argument(
        "--sim-mode", default="agent",
        choices=("analytical", "rust", "deterministic-split", "agent"),
        help="Which simulation manager wraps the per-node scorer "
             "(default: analytical). 'analytical' = STeP timing model "
             "only (legacy behavior). 'rust' = rust-evaluate every "
             "variant until the per-pass time limit is exhausted. "
             "'deterministic-split' = rust-evaluate just the baselines "
             "(anchors the Pareto cheaply); variants stay analytical. "
             "'agent' = an in-loop LLM decides per variant whether to "
             "spend rust budget, conditioned on past (analytical, "
             "rust) calibration records picked by a curation agent. "
             "All non-analytical modes append paired records to the "
             "calibration store and respect the per-pass budget set "
             "by each pass spec's 'time_limit_seconds'.",
    )
    parser.add_argument(
        "--sim-calibration-path", default=None,
        help="Path to the run-local calibration JSONL store the "
             "rust-backed simulation managers append paired "
             "(analytical, rust) records to. Default: "
             "<ckpt>/autotune2/calibration.jsonl inside the snapshotted "
             "run dir.",
    )
    parser.add_argument(
        "--sim-calibration-seed-path",
        default="/workspace/NextStep/StepDB/calibration_empty.jsonl",
        help="Optional read-only calibration JSONL used to seed the "
             "curation / decision agents. New records are not appended "
             "here unless this path is also passed as "
             "--sim-calibration-path. Pass an empty string to disable "
             "seed calibration.",
    )
    parser.add_argument(
        "--curation-max-candidates", type=int, default=50,
        help="Max calibration records the curation agent ranks per "
             "variant (default: 50). Larger values let the LLM see "
             "more divergence patterns at the cost of prompt size; "
             "smaller values are faster but starve the agent of "
             "signal. Only consulted when --sim-mode=agent or "
             "--root-pick=agent.",
    )
    parser.add_argument(
        "--curation-k", type=int, default=4,
        help="Number of records the curation agent picks (subset of "
             "the input; default: 4). Bounds the size of the evidence "
             "block fed to the simulation-decision / final-pick agents. "
             "Must satisfy 1 <= curation_k <= curation_max_candidates.",
    )
    parser.add_argument(
        "--rust-functional-check", dest="rust_functional_check",
        default=False, action=argparse.BooleanOptionalAction,
        help="Run the rust simulator with the functional sim enabled and "
             "compare its output tensor against the PyTorch gold reference "
             "for every variant (default: enabled). Catches structurally "
             "broken graphs that drain early and report meaningless cycle "
             "counts (see binary_map_accum_init_shape memory). Pass "
             "--no-rust-functional-check to skip the compare and record "
             "cycle counts only — faster, but bogus low-cycle 'winners' "
             "will not be flagged.",
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
