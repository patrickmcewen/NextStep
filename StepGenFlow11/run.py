"""StepGenFlow: Agentic STeP program generation."""

import argparse
import asyncio
import json
import sys

from src.config_loader import load_llm_config
from src.orchestrator import run_kernel


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
    parser = argparse.ArgumentParser(description="StepGenFlow — generate STeP programs from PyTorch references")
    parser.add_argument("kernel", help="Kernel name from StepDB (e.g., element_wise_add)")
    parser.add_argument("preset", help="Preset name (e.g., small)")
    parser.add_argument("--model", default="gpt-oss-120b",
                        help="Profile name under configs/ (loads configs/<name>.json). "
                             "Ignored if --config is given.")
    parser.add_argument("--config", default=None,
                        help="Explicit path to an LLM config JSON (overrides --model).")
    parser.add_argument("--max-outer", type=int, default=3, help="Max outer loop iterations")
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
        choices=["correctness-first", "compliance-first"],
        help="Order of per-turn gates. 'correctness-first' (default) runs the "
             "code first, then compliance/judge/post-validator — exactly as before. "
             "'compliance-first' runs the regex compliance and LLM judge before "
             "executing the code, so structurally noncompliant code is rejected "
             "without being run.",
    )
    parser.add_argument(
        "--no-plan", action="store_true",
        help="Disable Phase 0 (decomposition planner). Falls back to single-node refactor.",
    )
    parser.add_argument(
        "--max-replans", type=int, default=3,
        help="Global re-plan budget on Phase 1 failure. Default 3.",
    )
    parser.add_argument(
        "--node-attempts", type=int, default=5,
        help="Per-tree-node parallel refactor attempts. Default 1 (single "
             "attempt). Set higher to spawn N parallel refactor_final passes "
             "per node and use the first successful one — useful for nodes "
             "where the refactor pass is flaky.",
    )
    parser.add_argument(
        "--max-plan-depth", type=int, default=2,
        help="Maximum recursion depth of the planner tree. Depth 0 is the root, "
             "depth 1 is its direct children, etc. Any node at depth >= "
             "--max-plan-depth is forced to LEAF without consulting the LLM. "
             "Default 3 (root → mid → leaves).",
    )
    parser.add_argument(
        "--non-root-sequential", action=argparse.BooleanOptionalAction, default=True,
        help="When --node-attempts > 1, run the N attempts sequentially (with "
             "early-exit on success) for non-root nodes. Root nodes always run "
             "attempts in parallel. Default True — saves LLM tokens on easy "
             "nodes that succeed on the first try. Pass --no-non-root-sequential "
             "to spawn all N attempts in parallel even for non-root nodes.",
    )
    args = parser.parse_args()

    assert not (args.resume and args.resume_planner), (
        "--resume and --resume-planner are mutually exclusive: --resume skips "
        "lowering entirely (final dsl_code.py needed); --resume-planner re-runs "
        "lowering using a saved tree + per-node verified DSLs."
    )

    llm_config = load_llm_config(args.config, args.model)

    autotune_options = None
    if args.autotune:
        with open(args.autotune_config) as f:
            autotune_cfg = json.load(f)
        autotune_options = _build_autotune_options(args, autotune_cfg)

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
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
