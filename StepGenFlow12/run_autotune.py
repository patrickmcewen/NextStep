"""Autotuner CLI — run the performance-tuning agent on a verified build_graph.

The baseline build_graph must already exist as a successful checkpoint from
the implementer pipeline (run.py). Point --resume at the checkpoint root,
the kernel outer dir, or the extracted_code.py itself.
"""

import argparse
import asyncio
import sys

from src.autotune import run_autotune
from src.autotune_config_loader import load_autotune_config
from src.config_loader import load_llm_config
from src.process_group import setup_process_group


def main():
    setup_process_group()
    parser = argparse.ArgumentParser(description="Autotune a verified STeP build_graph.")
    parser.add_argument("kernel", help="Kernel name (must match StepDB bench_config.yaml)")
    parser.add_argument("preset", help="Preset name")
    parser.add_argument("--resume", required=True, metavar="PATH",
                        help="Path to a successful checkpoint — a .py file, a turn dir, "
                             "an outer_N dir, a kernel dir, or a checkpoint root.")
    parser.add_argument("--model", default="gpt-oss-120b",
                        help="Profile name under configs/ (loads configs/<name>.json). "
                             "Ignored if --config is given.")
    parser.add_argument("--config", default=None,
                        help="Explicit path to an LLM config JSON (overrides --model).")
    parser.add_argument("--autotune-config", default="autotune_configs.yaml",
                        help="Autotune config JSON/YAML (hw_config, constraints, max_turns)")
    parser.add_argument("--autotune-config-name", default="autotune_config",
                        help="Named config to resolve when --autotune-config is YAML.")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Override max_turns from the autotune config")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override autotune checkpoint dir (default: checkpoints_autotune/<ts>)")
    parser.add_argument("--agent", choices=["general", "parallel"], default="general",
                        help="Which autotuner agent to run: 'general' (default) covers "
                             "tile/compute/par_dispatch knobs and larger rewrites; "
                             "'parallel' only inserts/retunes Parallelize/StaticReassemble.")
    args = parser.parse_args()

    llm_config = load_llm_config(args.config, args.model)
    autotune_config = load_autotune_config(
        args.autotune_config, args.autotune_config_name,
    )

    result = asyncio.run(run_autotune(
        kernel_name=args.kernel,
        preset=args.preset,
        llm_config=llm_config,
        autotune_config=autotune_config,
        resume_from=args.resume,
        max_turns=args.max_turns,
        checkpoint_dir=args.checkpoint_dir,
        agent_variant=args.agent,
    ))

    print(f"\nbaseline_cycles = {result['baseline_cycles']}")
    print(f"best_cycles     = {result['best_cycles']}")
    if result.get("speedup") is not None:
        print(f"speedup         = {result['speedup']:.2f}x")
    print(f"checkpoint      = {result['checkpoint_dir']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
