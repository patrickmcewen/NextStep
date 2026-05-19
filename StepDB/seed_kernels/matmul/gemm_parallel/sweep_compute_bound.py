"""Sweep par_factor under low compute_bw (compute-bound regime).

Hypothesis: if compute is the bottleneck (not source bandwidth), shared
mode should give ~par_factor speedup. Earlier sweep at compute_bw=1024
showed shared mode capped at 1.06x — but that was source-bound. This
script drops compute_bw to force compute-bound conditions.

Run from /workspace/NextStep/StepDB:
    python seed_kernels/matmul/gemm_parallel/sweep_compute_bound.py
"""
import copy
import sys
from pathlib import Path

STEPDB_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STEPDB_DIR))

from validate_timing import load_config, validate_kernel


PRESET = "med_512"
COMPUTE_BWS = [64, 256, 1024]
PAR_FACTORS = [1, 2, 4, 8]
LOAD_MODE = "shared"


def main():
    base_config = load_config()
    KERNEL = "gemm_parallel"
    grid = {}
    for cb in COMPUTE_BWS:
        grid[cb] = {}
        for pf in PAR_FACTORS:
            cfg = copy.deepcopy(base_config)
            cfg[KERNEL]["presets"][PRESET]["par_factor"] = pf
            cfg[KERNEL]["presets"][PRESET]["load_mode"] = LOAD_MODE
            cfg[KERNEL]["presets"][PRESET]["compute_bw"] = cb
            print(f"\n--- {KERNEL} / {PRESET} / cb={cb} pf={pf} ---")
            _, _, predicted, actual, err, _ = validate_kernel(
                KERNEL, PRESET, cfg, verbose=False,
            )
            grid[cb][pf] = actual

    col_w = 14
    header = f"{'compute_bw':<14s}" + "".join(
        f"{f'pf={pf}':>{col_w}s}" for pf in PAR_FACTORS
    )
    print("\nSim cycles ({} / load_mode={})".format(PRESET, LOAD_MODE))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cb in COMPUTE_BWS:
        row = f"{cb:<14d}"
        for pf in PAR_FACTORS:
            row += f"{grid[cb][pf]:>{col_w}d}"
        print(row)
    print("=" * len(header))

    print("\nSpeedup vs pf=1 (same compute_bw)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cb in COMPUTE_BWS:
        baseline = grid[cb][1]
        row = f"{cb:<14d}"
        for pf in PAR_FACTORS:
            row += f"{baseline / grid[cb][pf]:>{col_w - 1}.2f}x"
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    main()
