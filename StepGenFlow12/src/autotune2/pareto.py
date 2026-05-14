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


def insert_pareto(front: list[DesignEntry], entry: DesignEntry) -> bool:
    """Insert ``entry`` into the front if it's not dominated by any existing
    member. Drop any members ``entry`` dominates. Returns True iff inserted.

    Treats equal-on-both-axes entries as duplicates: the existing entry stays,
    the new one is rejected. This avoids unbounded front growth from
    re-discovering identical Pareto points across LLM calls."""
    for x in front:
        if dominates(x, entry):
            return False
        if x.cycles == entry.cycles and x.on_chip == entry.on_chip:
            return False
    front[:] = [x for x in front if not dominates(entry, x)]
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
