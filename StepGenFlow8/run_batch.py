"""Run StepGenFlow on all easy/medium kernels and produce a summary."""

import argparse
import asyncio

from src.config_loader import load_llm_config
from src.orchestrator import run_kernel

EASY_KERNELS = ["element_wise_add", "copy_2d", "silu_activation", "chained_unary"]
MEDIUM_KERNELS = ["gemm", "rms_norm", "layernorm", "residual_add_norm"]


async def run_batch(llm_config, kernels, preset, max_outer, max_turns, results_dir, experience_dir, checkpoint_dir, pipeline="standard", resume_from=None):
    results = {}
    for kernel in kernels:
        print(f"\n{'='*60}")
        print(f"  {kernel} / {preset}")
        print(f"{'='*60}")
        result = await run_kernel(
            kernel_name=kernel,
            preset=preset,
            llm_config=llm_config,
            max_outer=max_outer,
            max_turns=max_turns,
            results_dir=results_dir,
            experience_dir=experience_dir,
            checkpoint_dir=checkpoint_dir,
            pipeline=pipeline,
            resume_from=resume_from,
        )
        results[kernel] = result

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    print(f"  {passed}/{total} passed\n")
    for kernel, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        iters = result["outer_iterations"]
        calls = result["total_tool_calls"]
        cycles = result.get("cycle_count", "-")
        print(f"  {status:4s}  {kernel:30s}  iters={iters}  tools={calls}  cycles={cycles}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch run StepGenFlow")
    parser.add_argument("--tier", choices=["easy", "medium", "all"], default="all")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--model", default="gpt-oss120b",
                        help="Profile name under configs/ (loads configs/<name>.json). "
                             "Ignored if --config is given.")
    parser.add_argument("--config", default=None,
                        help="Explicit path to an LLM config JSON (overrides --model).")
    parser.add_argument("--max-outer", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experience-dir", default="experience")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--pipeline", default="standard", choices=["standard", "direct"],
                        help="Pipeline mode: 'standard' or 'direct'")
    parser.add_argument("--resume", default=None, metavar="PATH",
                        help="Resume from a checkpoint where refactor_final succeeded. "
                             "Skips lowering and starts directly from translation.")
    args = parser.parse_args()

    llm_config = load_llm_config(args.config, args.model)

    kernels = []
    if args.tier in ("easy", "all"):
        kernels += EASY_KERNELS
    if args.tier in ("medium", "all"):
        kernels += MEDIUM_KERNELS

    asyncio.run(run_batch(
        llm_config, kernels, args.preset,
        args.max_outer, args.max_turns,
        args.results_dir, args.experience_dir,
        args.checkpoint_dir,
        pipeline=args.pipeline,
        resume_from=args.resume,
    ))


if __name__ == "__main__":
    main()
