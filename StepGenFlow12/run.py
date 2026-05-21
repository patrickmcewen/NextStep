"""StepGenFlow: Agentic STeP program generation."""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config_loader import load_llm_config
from src.log_redirect import redirect_stdio_to, terminal_print
from src.orchestrator import run_kernel
from src.process_group import setup_process_group


def _build_autotune_options(args, autotune_cfg: dict) -> dict:
    """Construct autotune_options for the orchestrator.

    If autotune_cfg["passes"] is set, use it verbatim. Otherwise synthesize
    a single-element passes list from the legacy --autotune-agent /
    --autotune-max-turns flags so existing callers keep working.
    """
    if "passes" in autotune_cfg:
        passes = autotune_cfg["passes"]
        assert isinstance(passes, list) and passes, (
            "autotune_config['passes'] must be a non-empty list")
        for i, spec in enumerate(passes):
            assert "agent" in spec, f"pass {i} is missing 'agent' key: {spec}"
    else:
        passes = [{
            "agent": args.autotune_agent,
            "max_turns": args.autotune_max_turns,
        }]
    return {"config": autotune_cfg, "passes": passes}


def main():
    setup_process_group()
    parser = argparse.ArgumentParser(description="StepGenFlow — generate STeP programs from PyTorch references")
    parser.add_argument("kernel", help="Kernel name from StepDB (e.g., element_wise_add)")
    parser.add_argument("preset", help="Preset name (e.g., small)")
    parser.add_argument("--model", default="gpt-oss-120b",
                        help="Profile name under configs/ (loads configs/<name>.json). "
                             "Ignored if --config is given.")
    parser.add_argument("--config", default=None,
                        help="Explicit path to an LLM config JSON (overrides --model).")
    parser.add_argument("--max-outer", type=int, default=4, help="Max outer loop iterations")
    parser.add_argument("--max-turns", type=int, default=16, help="Max tool-call rounds per inner loop")
    parser.add_argument("--results-dir", default="/workspace/results", help="Directory for run logs")
    parser.add_argument("--experience-dir", default="experience", help="Directory for successful implementations")
    parser.add_argument("--checkpoint-dir", default="/workspace/checkpoints", help="Override checkpoint directory (default: checkpoints/<timestamp>)")
    parser.add_argument("--pipeline", default="standard", choices=["standard", "direct", "direct_no_functional"],
                        help="Pipeline mode: 'standard' (lowering + translate) or 'direct' (PyTorch → STeP in one step) or 'direct_no_functional' (PyTorch → STeP in one step without functional.py)")
    parser.add_argument("--translator", default="auto", choices=["llm", "auto"],
                        help="Translation backend: 'auto' (default; deterministic AST rewrite "
                             "from DSL to STeP) or 'llm' (LLM translate pass). "
                             "'auto' requires --pipeline=standard.")
    parser.add_argument("--resume", default=None, metavar="PATH",
                        help="Resume from a checkpoint where refactor_final succeeded. "
                             "Accepts: path to dsl_code.py, outer_N dir, or checkpoint root dir. "
                             "Skips lowering and starts directly from translation.")
    parser.add_argument("--resume-planner", default=None, metavar="OUTER_DIR",
                        help="Resume an outer iteration that crashed mid-planner-walk. "
                             "Loads the saved tree (from <OUTER_DIR>/plan/iteration_*/tree.json) "
                             "and any per-node verified DSLs (from <OUTER_DIR>/refactor/.../"
                             "extracted_code.py with status PASS), then re-runs only the "
                             "non-verified nodes. Mutually exclusive with --resume.")
    parser.add_argument("--resume-after-pass1", default=None, metavar="OUTER_DIR",
                        help="Resume an outer iteration after pass1 completed but pass2 "
                             "(or downstream translation) failed. Loads the saved tree "
                             "and per-node verified DSLs from <OUTER_DIR>, then runs ONLY "
                             "pass2 composition + translation — planner and pass1 LLM are "
                             "skipped entirely. Requires every node in the tree to have "
                             "a cached PASS DSL; fails fast with the missing list "
                             "otherwise. Forces --max-outer=1 since the operation is "
                             "deterministic. Mutually exclusive with --resume / "
                             "--resume-planner.")
    parser.add_argument("--few-shot", nargs="+", default=None, metavar="PATH",
                        help="Optional paths to previously completed step program "
                             "directories. Each contributes a PyTorch→DSL example pair "
                             "to the refactor_final system prompt. Useful for "
                             "kernel-family hints (e.g. RoPE/SDPA/MoE for transformers). "
                             "Accepts: dsl_code.py file, outer_N dir, or checkpoint root dir. "
                             "Off by default.")
    parser.add_argument("--bundle-dir", default=None, metavar="PATH",
                        help="Path to a bundle directory containing abstraction.py, "
                             "transpiler.py, and refactor_system.txt. When set, the "
                             "inner orchestrator loads these instead of the legacy "
                             "src/step_dsl.py / src/dsl_to_step.py / prompt files. "
                             "Required when invoked from the AbstractionOpt outer flow.")
    parser.add_argument("--autotune", action="store_true",
                        help="Run the autotuner on each outer's verified build_graph.")
    parser.add_argument("--autotune-config", default="autotune_config.json",
                        help="Path to autotune_config JSON (hw_config, constraints, max_turns).")
    parser.add_argument("--autotune-max-turns", type=int, default=None,
                        help="Override max_turns from autotune_config.")
    parser.add_argument("--autotune-agent", default="general",
                        help="Autotuner agent variant (e.g. 'general', 'parallel'). "
                             "Ignored when autotune_config.json defines 'passes'. "
                             "Validated by the agent factory at run time.")
    parser.add_argument(
        "--check-order", default="correctness-first",
        choices=["correctness-first", "compliance-first", "always-both"],
        help="Order of per-turn gates. 'correctness-first' (default) runs the "
             "code first, then compliance/judge/post-validator — exactly as before. "
             "'compliance-first' runs the regex compliance and LLM judge before "
             "executing the code, so structurally noncompliant code is rejected "
             "without being run. 'always-both' runs every gate (correctness, "
             "compliance, judge, post-validator) without short-circuiting on the "
             "first failure and concatenates all gate feedback into the next "
             "user turn — maximum signal per turn at the cost of extra judge "
             "calls when correctness already fails.",
    )
    parser.add_argument(
        "--no-plan", action="store_true",
        help="Disable Phase 0 (decomposition planner). Falls back to single-node refactor.",
    )
    parser.add_argument(
        "--max-replans", type=int, default=1,
        help="Global re-plan budget on Phase 1 failure. Default 3.",
    )
    parser.add_argument(
        "--node-attempts", type=int, default=4,
        help="Per-tree-node parallel refactor attempts. Default 1 (single "
             "attempt). Set higher to spawn N parallel refactor_final passes "
             "per node and use the first successful one — useful for nodes "
             "where the refactor pass is flaky.",
    )
    parser.add_argument(
        "--max-plan-depth", type=int, default=5,
        help="Maximum recursion depth of the planner tree. Depth 0 is the root, "
             "depth 1 is its direct children, etc. Any node at depth >= "
             "--max-plan-depth is forced to LEAF without consulting the LLM. "
             "Default 3 (root → mid → leaves).",
    )
    parser.add_argument(
        "--non-root-sequential", action=argparse.BooleanOptionalAction, default=False,
        help="When --node-attempts > 1, run the N attempts sequentially (with "
             "early-exit on success) for non-root nodes. Root nodes always run "
             "attempts in parallel. Default True — saves LLM tokens on easy "
             "nodes that succeed on the first try. Pass --no-non-root-sequential "
             "to spawn all N attempts in parallel even for non-root nodes.",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Disable the LLM judge stack entirely. Regex compliance still runs; "
             "refactor_final loses the line-specific judge feedback that "
             "compliance-first / always-both modes rely on.",
    )
    parser.add_argument(
        "--stateless-refactor", action="store_true",
        help="Run the refactor agent in stateless mode: each turn rebuilds the "
             "user prompt as (original prompt + latest failed code + latest "
             "feedback) and discards the rest of the chat history. Caps "
             "per-turn context at O(1) and keeps the prompt-cache prefix "
             "stable across turns. Off by default — accumulating chat history "
             "is the existing behavior.",
    )
    parser.add_argument(
        "--max-tile", type=int, default=None, metavar="N",
        help="Run pass-1 against step_dsl_max_tile.py with both MAX_TILE_ROW "
             "and MAX_TILE_COL set to N. Every load asserts tile_row,tile_col "
             "<= N; tile-growing reshapes (accum_retile_row/col, restream) "
             "assert the output tile stays <= (N, N); the final offchip_store "
             "input tile must satisfy the same bound. N=1 recovers strict 1x1 "
             "mode (everything at the stream level). Larger N gives the model "
             "more room while still letting the autotuner pick the final "
             "tile size via accum_retile_*. Off by default (uses stock "
             "step_dsl.py). Requires --no-plan to be OFF and is incompatible "
             "with --bundle-dir.",
    )
    args = parser.parse_args()

    _resume_modes_set = sum(
        1 for m in (args.resume, args.resume_planner, args.resume_after_pass1) if m
    )
    assert _resume_modes_set <= 1, (
        "--resume, --resume-planner, and --resume-after-pass1 are pairwise "
        "mutually exclusive. --resume skips lowering entirely (final "
        "dsl_code.py needed); --resume-planner re-runs pass1 using a saved "
        "tree + per-node verified DSLs (verified_cache is currently not wired "
        "through, so pass1 runs from scratch); --resume-after-pass1 skips "
        "planner + pass1 entirely and runs only pass2 composition + translation."
    )

    if args.resume_after_pass1 and args.max_outer != 1:
        # The resume operation is deterministic — running it across N parallel
        # outers just produces N identical results. Force 1 to save compute.
        print(f"--resume-after-pass1 forces --max-outer=1 (was {args.max_outer})")
        args.max_outer = 1

    llm_config = load_llm_config(args.config, args.model)

    autotune_options = None
    if args.autotune:
        with open(args.autotune_config) as f:
            autotune_cfg = json.load(f)
        autotune_options = _build_autotune_options(args, autotune_cfg)

    # Pre-generate the timestamp dir so we can install stdio redirection
    # against <ckpt>/<ts>/run.log before run_kernel starts printing. When
    # stdout is not a TTY (e.g. run_regression.py spawned this run.py with
    # piped stdout), redirect_stdio_to is a no-op so the parent's per-job
    # log keeps receiving output.
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    base_ckpt = args.checkpoint_dir if args.checkpoint_dir else "checkpoints"
    ckpt_ts_dir = Path(base_ckpt) / ts
    ckpt_ts_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_ts_dir / "run.log"
    redirected = redirect_stdio_to(log_path)
    if redirected:
        terminal_print(f"run.py log -> {log_path}")

    result = asyncio.run(run_kernel(
        kernel_name=args.kernel,
        preset=args.preset,
        llm_config=llm_config,
        max_outer=args.max_outer,
        max_turns=args.max_turns,
        results_dir=args.results_dir,
        experience_dir=args.experience_dir,
        checkpoint_dir=args.checkpoint_dir,
        pipeline=args.pipeline,
        resume_from=args.resume,
        translator=args.translator,
        few_shot_paths=args.few_shot,
        bundle_dir=args.bundle_dir,
        autotune_options=autotune_options,
        check_order=args.check_order,
        plan_enabled=not args.no_plan,
        max_replans=args.max_replans,
        node_attempts=args.node_attempts,
        non_root_sequential=args.non_root_sequential,
        max_plan_depth=args.max_plan_depth,
        resume_planner=args.resume_planner,
        resume_after_pass1=args.resume_after_pass1,
        stateless_refactor=args.stateless_refactor,
        judge_enabled=not args.no_judge,
        max_tile=args.max_tile,
        pregenerated_ts=ts,
    ))

    if result["success"]:
        print(f"\nSUCCESS — {args.kernel}/{args.preset}")
        print(f"  Outer iterations: {result['outer_iterations']}")
        print(f"  Total tool calls: {result['total_tool_calls']}")
        if result.get("cycle_count"):
            print(f"  Cycle count: {result['cycle_count']}")
    else:
        print(f"\nFAILED — {args.kernel}/{args.preset}")
        print(f"  Final diagnosis: {result.get('final_diagnosis', 'N/A')}")
    if redirected:
        status = "SUCCESS" if result["success"] else "FAILED"
        terminal_print(f"run.py {status} — {args.kernel}/{args.preset} (log: {log_path})")
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
