"""StepGenFlow: Agentic STeP program generation."""

import argparse
import asyncio
import sys

from src.config_loader import load_llm_config
from src.orchestrator import run_kernel


def main():
    parser = argparse.ArgumentParser(description="StepGenFlow — generate STeP programs from PyTorch references")
    parser.add_argument("kernel", help="Kernel name from StepDB (e.g., element_wise_add)")
    parser.add_argument("preset", help="Preset name (e.g., small)")
    parser.add_argument("--model", default="gpt-oss120b",
                        help="Profile name under configs/ (loads configs/<name>.json). "
                             "Ignored if --config is given.")
    parser.add_argument("--config", default=None,
                        help="Explicit path to an LLM config JSON (overrides --model).")
    parser.add_argument("--max-outer", type=int, default=1, help="Max outer loop iterations")
    parser.add_argument("--max-turns", type=int, default=11, help="Max tool-call rounds per inner loop")
    parser.add_argument("--results-dir", default="results", help="Directory for run logs")
    parser.add_argument("--experience-dir", default="experience", help="Directory for successful implementations")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory (default: checkpoints/<timestamp>)")
    parser.add_argument("--pipeline", default="standard", choices=["standard", "direct"],
                        help="Pipeline mode: 'standard' (lowering + translate) or 'direct' (PyTorch → STeP in one step)")
    parser.add_argument("--resume", default=None, metavar="PATH",
                        help="Resume from a checkpoint where refactor_final succeeded. "
                             "Accepts: path to dsl_code.py, outer_N dir, or checkpoint root dir. "
                             "Skips lowering and starts directly from translation.")
    args = parser.parse_args()

    llm_config = load_llm_config(args.config, args.model)

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
