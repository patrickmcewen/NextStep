"""Sweep par_factor for gemm_parallel under both load_modes on med_512.

shared: one A/B LinearOffChipLoad feeds a Parallelize op (current default).
indep:  one A/B LinearOffChipLoad per consumer (no shared upstream).

The shared mode appears to be capped by the source's token rate; the indep
mode replicates B traffic but lets each consumer's pipeline advance
independently.

Run from /workspace/NextStep/StepDB:
    python seed_kernels/matmul/gemm_parallel/sweep_par_factor.py
"""
import copy
import sys
from pathlib import Path

STEPDB_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STEPDB_DIR))

from validate_timing import load_config, validate_kernel


PRESET = "med_512"
PAR_FACTORS = [1, 2, 4, 8]
LOAD_MODES = ["shared", "indep"]


def main():
    base_config = load_config()
    KERNEL = "gemm_parallel"
    assert PRESET in base_config[KERNEL]["presets"]

    grid = {}  # grid[load_mode][par_factor] = (predicted, actual, err)
    for mode in LOAD_MODES:
        grid[mode] = {}
        for pf in PAR_FACTORS:
            cfg = copy.deepcopy(base_config)
            cfg[KERNEL]["presets"][PRESET]["par_factor"] = pf
            cfg[KERNEL]["presets"][PRESET]["load_mode"] = mode
            print(f"\n--- {KERNEL} / {PRESET} / load_mode={mode} par_factor={pf} ---")
            _, _, predicted, actual, err, _ = validate_kernel(
                KERNEL, PRESET, cfg, verbose=False,
            )
            grid[mode][pf] = (predicted, actual, err)

    def _print_table(title, get_cell):
        col_w = 14
        header = f"{'par_factor':<14s}" + "".join(
            f"{m:>{col_w}s}" for m in LOAD_MODES
        )
        print(f"\n{title}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for pf in PAR_FACTORS:
            row = f"{pf:<14d}"
            for mode in LOAD_MODES:
                row += f"{get_cell(mode, pf):>{col_w}s}"
            print(row)
        print("=" * len(header))

    # Speedup baseline is (load_mode=shared, par_factor=1) — the apples-to-
    # apples "no parallelism, no replication" point.
    baseline = grid["shared"][1][1]
    _print_table(
        f"Sim cycles ({KERNEL} / {PRESET})",
        lambda m, pf: f"{grid[m][pf][1]}",
    )
    _print_table(
        "Sim speedup vs shared/par_factor=1",
        lambda m, pf: f"{baseline / grid[m][pf][1]:.2f}x",
    )
    _print_table(
        "Analytical model error %",
        lambda m, pf: f"{grid[m][pf][2]:+.1f}%",
    )


if __name__ == "__main__":
    main()
