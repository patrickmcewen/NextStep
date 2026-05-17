"""Pareto-front utilities over ``DesignEntry`` lists.

A front is a plain ``list[DesignEntry]``, conventionally non-dominated and
unsorted. Membership-changing operations (``insert``, ``cull_top_T``) mutate
in place. ``dominates`` is the lexicographic-min ordering on
``(cycles, on_chip)``: a strictly better in either axis, no worse in the other.
"""

import hashlib
import re

from src.autotune2.contracts import DesignEntry


def dominates(a: DesignEntry, b: DesignEntry) -> bool:
    """True iff ``a`` Pareto-dominates ``b`` on (cycles, on_chip)."""
    return (
        a.cycles <= b.cycles
        and a.on_chip <= b.on_chip
        and (a.cycles < b.cycles or a.on_chip < b.on_chip)
    )


PROTECTED_PROVENANCES: frozenset[str] = frozenset({"pass1_baseline"})
"""Provenance tags whose entries are never evicted from a Pareto front.

The pass-1 baseline is a load-bearing reference: parents must compose against
their children's pass-1 baseline (not whichever LLM variant happens to be
Pareto-best) because that's the only chain pass-1 already proved composes
end-to-end. Letting Pareto eviction drop it strands ``find_pass1_baseline_entry``
later in the search.
"""


def insert_pareto(front: list[DesignEntry], entry: DesignEntry) -> bool:
    """Insert ``entry`` into the front if it's not dominated by any existing
    member. Drop any members ``entry`` dominates, except those whose
    ``provenance`` is in ``PROTECTED_PROVENANCES`` — those stay regardless.
    Returns True iff inserted.

    Treats equal-on-both-axes entries as duplicates: the existing entry stays,
    the new one is rejected. This avoids unbounded front growth from
    re-discovering identical Pareto points across LLM calls.

    Refuses to insert any new entry whose provenance is protected: the
    seeding routines own the single insertion of those entries via raw
    ``cell.append``, and a duplicate would silently break the
    "exactly one baseline per library" invariant."""
    assert entry.provenance not in PROTECTED_PROVENANCES, (
        f"insert_pareto: refusing to insert entry with protected provenance "
        f"{entry.provenance!r}; protected entries are seeded directly via "
        f"cell.append, not through insert_pareto"
    )
    for x in front:
        if dominates(x, entry):
            return False
        if x.cycles == entry.cycles and x.on_chip == entry.on_chip:
            return False
    front[:] = [
        x for x in front
        if x.provenance in PROTECTED_PROVENANCES or not dominates(entry, x)
    ]
    front.append(entry)
    return True


def cull_top_T(front: list[DesignEntry], T: int) -> None:
    """Reduce ``front`` to at most ``T`` entries, spaced evenly along cycles.

    Even spacing biases coverage toward the curve's full extent rather than
    crowding the low-cycles tail. For T=8-16 this is plenty; for tighter
    budgets consider hypervolume-aware culling.
    """
    assert T >= 1, f"cull_top_T: T must be >= 1, got {T}"
    if len(front) <= T:
        return
    front.sort(key=lambda e: e.cycles)
    if T == 1:
        front[:] = front[:1]
        return
    n = len(front)
    idx = sorted({round(i * (n - 1) / (T - 1)) for i in range(T)})
    front[:] = [front[i] for i in idx]


# --- dedup ---------------------------------------------------------------------


_WS = re.compile(r"\s+")


def dsl_dedup_hash(dsl: str) -> str:
    """SHA256 of whitespace-normalized DSL.

    Used to skip LLM proposals that are byte-equivalent (modulo whitespace)
    to one we've already scored. Does NOT canonicalize structurally — two
    DSLs with the same op graph but different tile values produce different
    hashes, which is what we want (the LLM is free to vary tiles)."""
    return hashlib.sha256(_WS.sub(" ", dsl).strip().encode()).hexdigest()
