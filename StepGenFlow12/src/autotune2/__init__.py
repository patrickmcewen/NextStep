"""Bottom-up tree-DP autotuner with per-node Pareto libraries.

Attaches after Pass-1/Pass-2 succeed. Walks the planner tree post-order,
building a library of (cycles, on_chip)-Pareto-optimal variants at each
node. The pass-1 design is always entry zero (rollback guarantee).

Each variant is identified by its ``(input_contracts, output_contracts)``
pair, where a contract is a ``(reshape, permutation)`` of the vanilla
PyTorch tensor. Contracts apply only to on-chip stream args; RAW args
(off-chip, loaded fresh inside the function) are always vanilla.

See ``design/autotuner2.md`` for the full design narrative (to be written).
"""
