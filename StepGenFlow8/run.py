"""StepGenFlow: Agentic STeP program generation."""

import argparse
import asyncio
import json
import sys

from src.config_loader import load_llm_config
from src.orchestrator import run_kernel


def main():
    parser = argparse.ArgumentParser(description="StepGenFlow — generate STeP programs from PyTorch references")
    parser.add_argument("kernel", help="Kernel name from StepDB (e.g., element_wise_add)")
    parser.add_argument("preset", help="Preset name (e.g., small)")
    parser.add_argument("--model", default="gpt-oss-120b",
                        help="Profile name under configs/ (loads configs/<name>.json). "
                             "Ignored if --config is given.")
    parser.add_argument("--config", default=None,
                        help="Explicit path to an LLM config JSON (overrides --model).")
    parser.add_argument("--max-outer", type=int, default=5, help="Max outer loop iterations")
    parser.add_argument("--max-turns", type=int, default=16, help="Max tool-call rounds per inner loop")
    parser.add_argument("--results-dir", default="results", help="Directory for run logs")
    parser.add_argument("--experience-dir", default="experience", help="Directory for successful implementations")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory (default: checkpoints/<timestamp>)")
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
    parser.add_argument("--few-shot", nargs="+", default=None, metavar="PATH",
                        help="Optional paths to previously completed step program "
                             "directories. Each contributes a PyTorch→DSL example pair "
                             "to the refactor_final system prompt. Useful for "
                             "kernel-family hints (e.g. RoPE/SDPA/MoE for transformers). "
                             "Accepts: dsl_code.py file, outer_N dir, or checkpoint root dir. "
                             "Off by default.")
    parser.add_argument("--autotune", action="store_true",
                        help="Run the autotuner on each outer's verified build_graph.")
    parser.add_argument("--autotune-config", default="autotune_config.json",
                        help="Path to autotune_config JSON (hw_config, constraints, max_turns).")
    parser.add_argument("--autotune-max-turns", type=int, default=None,
                        help="Override max_turns from autotune_config.")
    parser.add_argument("--autotune-agent", default="general",
                        choices=["general", "parallel"],
                        help="Which autotuner agent to run.")
    args = parser.parse_args()

    llm_config = load_llm_config(args.config, args.model)

    autotune_options = None
    if args.autotune:
        with open(args.autotune_config) as f:
            autotune_cfg = json.load(f)
        autotune_options = {
            "config": autotune_cfg,
            "max_turns": args.autotune_max_turns,
            "agent_variant": args.autotune_agent,
        }

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
        autotune_options=autotune_options,
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
