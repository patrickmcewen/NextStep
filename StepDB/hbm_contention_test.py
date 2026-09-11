"""Sweep N parallel HBM load->store chains and compare model vs sim.

Tests whether the analytical contention model in `_compute_hbm_oti` matches
the simulator when many independent HBM ops run concurrently. Each chain is
a `LinearOffChipLoad -> OffChipStore` of an independent tensor — no compute,
no PMU back-pressure between chains. This isolates the multi-op HBM model.

Usage:
    python hbm_contention_test.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_timing as vt


def build_n_chains(n_chains, M, K, tile_m, tile_k, par_dispatch):
    """Build a graph of N parallel LinearOffChipLoad -> OffChipStore chains."""
    # Re-import inside the IMPORT_SCAFFOLD environment (matches validate_timing.py)
    impl_code = f"""
SEED = 42

def build_graph(dims):
    N_chains = {n_chains}
    M, K = {M}, {K}
    tile_m, tile_k = {tile_m}, {tile_k}
    par_dispatch = {par_dispatch}
    torch.manual_seed(SEED)

    g = Graph()
    for i in range(N_chains):
        A = torch.randn(M, K)
        load = LinearOffChipLoad(
            underlying=A,
            stride=(K // tile_k, 1),
            out_shape_tiled=(M // tile_m, K // tile_k),
            tile_row=tile_m, tile_col=tile_k,
            par_dispatch=par_dispatch,
        )
        OffChipStore(graph=g, input=load, par_dispatch=par_dispatch,
                     store_file_name=f"out_{{i}}")
    g = infer_broadcast(g)
    # OffChipStore for last chain (we just need *some* sink for run_simulator)
    # but each chain has its own store; pick the last as output_op handle
    return g, list(g.nodes)[-1]
"""
    full = vt.IMPORT_SCAFFOLD + "\n" + impl_code
    ns = {}
    exec(full, ns)
    return ns["build_graph"]({})


def run_one(n_chains, M, K, tile_m, tile_k, par_dispatch, label):
    graph, output_op = build_n_chains(n_chains, M, K, tile_m, tile_k, par_dispatch)

    predicted, _ = vt.run_analytical_model(graph)

    work_dir = os.path.join(vt.STEPDB_DIR, "seed_kernels", "_hbm_test", label)
    actual = vt.run_simulator(graph, output_op, work_dir)
    if isinstance(actual, tuple):
        actual = actual[0]

    err = (predicted - actual) / max(actual, 1) * 100
    print(f"  {label:<35}  pred={predicted:>10d}  sim={actual:>10d}  "
          f"err={err:>+7.1f}%  ratio={actual/predicted:.2f}x")
    return predicted, actual


def main():
    print("== Sweep: N parallel chains, fixed tile size (256x256, par_dispatch=4) ==")
    for n in [1, 2, 4, 8, 16, 32]:
        run_one(n, M=256, K=256, tile_m=64, tile_k=64,
                par_dispatch=4, label=f"N={n}_med")

    print()
    print("== Sweep: N parallel chains, SMALL tiles (1024x1024, tile=64x16, par_dispatch=4) ==")
    print("== (mimics the moe-like regime: many small per-tile reads)             ==")
    for n in [1, 2, 4, 8, 16, 24]:
        run_one(n, M=1024, K=1024, tile_m=64, tile_k=16,
                par_dispatch=4, label=f"N={n}_small_tiles")

    print()
    print("== Sweep: N parallel chains, TINY tiles (par_dispatch=1) — moe tile_f=2 mimic ==")
    for n in [1, 2, 4, 8, 16, 24]:
        # tile_row=1024, tile_col=2: same shape as moe weight tile at tile_f=2
        run_one(n, M=1024, K=2048, tile_m=1024, tile_k=2,
                par_dispatch=16, label=f"N={n}_moe_like")


if __name__ == "__main__":
    main()
