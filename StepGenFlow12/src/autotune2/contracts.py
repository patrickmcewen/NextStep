"""Variant contracts and library data model.

A ``TensorContract`` is a ``(reshape, permutation)`` pair describing how a
vanilla PyTorch tensor is laid out at a function boundary. Identity =
``reshape == vanilla_shape`` and identity permutation = "vanilla".

A ``DesignEntry`` is one row in a node's library: a verified DSL skeleton
plus its analytical (cycles, on_chip) score and the input/output
contracts under which the score was measured. Contracts are tracked
*only for on-chip-stream args* — RAW args have no contract axis.

A ``NodeLibrary`` indexes entries by ``(input_contracts, output_contracts)``,
maintaining a Pareto front per cell.
"""

from dataclasses import dataclass, field
from math import prod


@dataclass(frozen=True)
class TensorContract:
    """One tensor's boundary layout: a reshape spec + a permutation of the
    post-reshape dims.

    ``reshape`` is a valid factorization of the original vanilla shape
    (product of dims must equal product of the vanilla shape). ``permutation``
    is a permutation of ``range(len(reshape))``.

    Mechanically a contract is applied to a vanilla tensor as:
        x_in_contract = x_vanilla.reshape(reshape).permute(*permutation)
    """

    reshape: tuple[int, ...]
    permutation: tuple[int, ...]

    def __post_init__(self):
        assert len(self.reshape) == len(self.permutation), (
            f"TensorContract: reshape rank {len(self.reshape)} != "
            f"permutation rank {len(self.permutation)}; reshape={self.reshape!r}, "
            f"permutation={self.permutation!r}"
        )
        for d in self.reshape:
            assert isinstance(d, int) and d > 0, (
                f"TensorContract.reshape contains non-positive-int dim {d!r}; "
                f"reshape={self.reshape!r}"
            )
        n = len(self.permutation)
        assert sorted(self.permutation) == list(range(n)), (
            f"TensorContract.permutation must be a permutation of "
            f"range({n}); got {self.permutation!r}"
        )

    def is_identity_of(self, vanilla_shape: tuple[int, ...]) -> bool:
        """True iff this contract is the identity reshape+permute of the
        given vanilla shape (i.e., produces a tensor identical to vanilla)."""
        return (
            self.reshape == tuple(vanilla_shape)
            and self.permutation == tuple(range(len(vanilla_shape)))
        )

    def post_permute_shape(self) -> tuple[int, ...]:
        """The actual tensor shape after applying reshape + permute."""
        return tuple(self.reshape[i] for i in self.permutation)

    def canonical_name(self, vanilla_shape: tuple[int, ...] | None = None) -> str:
        """Stable short identifier used in LLM prompts and stub-table keys.

        Returns ``"vanilla"`` when this contract is the identity of
        ``vanilla_shape``; otherwise returns ``"r(d1,d2,...)p(p1,p2,...)"``.
        ``vanilla_shape=None`` skips the identity check and always emits the
        r/p form (useful when vanilla isn't conveniently available).
        """
        if vanilla_shape is not None and self.is_identity_of(vanilla_shape):
            return "vanilla"
        rs = ",".join(str(d) for d in self.reshape)
        ps = ",".join(str(p) for p in self.permutation)
        return f"r({rs})p({ps})"


def vanilla_contract_for(vanilla_shape: tuple[int, ...]) -> TensorContract:
    """The identity contract for a given vanilla shape."""
    return TensorContract(
        reshape=tuple(vanilla_shape),
        permutation=tuple(range(len(vanilla_shape))),
    )


def validate_reshape(vanilla_shape: tuple[int, ...], reshape: tuple[int, ...]) -> None:
    """Assert that ``reshape`` is a valid factorization of ``vanilla_shape``.

    The only invariant a reshape must satisfy is element-count preservation —
    rank may change and dims may be split or merged freely. Direction-of-
    factorization (e.g., 12 -> (3, 4) vs 12 -> (4, 3)) is not constrained
    here; that's a stylistic choice up to the LLM and the validator only
    catches arithmetic errors.
    """
    vp = prod(vanilla_shape)
    rp = prod(reshape)
    assert vp == rp, (
        f"reshape {reshape!r} (product {rp}) does not preserve total element "
        f"count of vanilla {vanilla_shape!r} (product {vp})"
    )


# --- DesignEntry + library typing ----------------------------------------------


@dataclass
class DesignEntry:
    """One row in a node's library."""

    dsl: str
    """The verified DSL source text for this variant."""

    input_contracts: dict[str, "TensorContract"] = field(default_factory=dict)
    """Per-arg contracts. Keys are on-chip arg names only; RAW args omitted
    (they have no contract axis — always vanilla on entry by definition)."""

    output_contracts: dict[str, "TensorContract"] = field(default_factory=dict)
    """Per-output contracts. Outputs are always on-chip streams (no RAW
    outputs in pass-1's model), so every output has a contract."""

    cycles: int = 0
    """Recorded cycle estimate. Set by the autotuner after scoring. The
    estimator that produced this number is tagged in ``cycle_source``."""

    on_chip: int = 0
    """Analytical on-chip bytes estimate. Always from the analytical model
    — the rust simulator does not surface on-chip bytes today."""

    cycle_source: str = "analytical"
    """Which simulator produced ``cycles``: ``"analytical"`` (STeP timing
    model) or ``"rust"`` (cycle-approximate rust sim). When the simulation
    manager invokes both for one variant, only one value is recorded —
    callers reading ``cycles`` MUST consult this tag before comparing
    across entries, since mixed-source numbers are not directly
    comparable."""

    provenance: str = ""
    """Free-form tag indicating where the entry came from (e.g.,
    ``"pass1_baseline"``, ``"llm_call_3"``). Useful for debugging library
    growth; not consumed by any logic."""

    breakdown: str = ""
    """Per-node on-chip memory report from the analytical scorer for this
    entry's composed source. Populated when the entry is created against a
    scorer that exposes ``.breakdown`` (production); empty string when the
    scorer is a test stub without it. Cached here so multi-pass branches
    can render a starting-design memory breakdown into the LLM prompt
    without re-running ``compose_source`` + ``score_fn.breakdown``."""

    children_picks: dict[str, "DesignEntry"] = field(default_factory=dict)
    """For parent entries: child_path -> the DesignEntry chosen from that
    child's library at the time this entry was scored. Used by the search
    driver's ``gather_descendants_postorder`` to recursively reconstruct
    the full descendant DSL chain when composing this entry's parent.
    Leaf entries have an empty dict. Always populated via object reference
    (not index), so Pareto culling at the child level doesn't orphan the
    parent's reproducibility chain — Python's GC keeps referenced entries
    alive even after they're culled from their own cell."""


# A hashable freeze of a per-arg contract dict, used to key library cells.
# Sorted by arg name so two dicts with the same contents produce the same key.
ContractsKey = tuple[tuple[str, TensorContract], ...]


def freeze_contracts(contracts: dict[str, TensorContract]) -> ContractsKey:
    return tuple(sorted(contracts.items()))


# A node's library: nested dict keyed by (input contracts, output contracts),
# each cell holding a Pareto front of DesignEntry.
NodeLibrary = dict[ContractsKey, dict[ContractsKey, list[DesignEntry]]]


def library_cell(
    lib: NodeLibrary,
    input_contracts: dict[str, TensorContract],
    output_contracts: dict[str, TensorContract],
) -> list[DesignEntry]:
    """Return (creating if needed) the Pareto front list for the given
    (input_contracts, output_contracts) cell of ``lib``. Mutating the
    returned list mutates the library in place."""
    in_key = freeze_contracts(input_contracts)
    out_key = freeze_contracts(output_contracts)
    return lib.setdefault(in_key, {}).setdefault(out_key, [])
