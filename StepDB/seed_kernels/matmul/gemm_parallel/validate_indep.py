"""Quick functional check: verify load_mode='indep' matches gemm's gold
for a few (preset, par_factor) combos. Uses validate_functional's loader
+ runner with an in-memory config override.
"""
import copy
import sys
from pathlib import Path

STEPDB_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STEPDB_DIR))

from validate_functional import load_config, validate_kernel


CHECKS = [
    ("small",    [1, 2]),    # M_t=2
    ("square",   [2, 4]),    # M_t=4
    ("med_256",  [2, 4]),    # M_t=4
    ("med_512",  [2, 4, 8]), # M_t=8
]


def main():
    base = load_config()
    K = "gemm_parallel"
    fail = 0
    for preset, pfs in CHECKS:
        for pf in pfs:
            cfg = copy.deepcopy(base)
            cfg[K]["presets"][preset]["par_factor"] = pf
            cfg[K]["presets"][preset]["load_mode"] = "indep"
            kernel, p_out, match, max_err, msg = validate_kernel(K, preset, cfg)
            status = "PASS" if match else "FAIL"
            print(f"  {status:4s}  {preset:10s} pf={pf}  {msg}")
            if not match:
                fail += 1
    print(f"\n{'OK' if fail == 0 else f'{fail} FAIL'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
