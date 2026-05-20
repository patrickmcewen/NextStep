"""Unit tests for autotune2 contract data model + Pareto utilities."""

import pytest

from src.autotune2.contracts import (
    ContractsKey,
    DesignEntry,
    NodeLibrary,
    TensorContract,
    freeze_contracts,
    library_cell,
    validate_reshape,
    vanilla_contract_for,
)
from src.autotune2.pareto import (
    PROTECTED_PROVENANCES,
    admit_to_cell,
    cull_top_T,
    dominates,
    dsl_dedup_hash,
    insert_pareto,
    pareto_front_of,
)


# --- TensorContract ------------------------------------------------------------


def test_tensor_contract_identity():
    c = TensorContract(reshape=(1, 64, 512), permutation=(0, 1, 2))
    assert c.is_identity_of((1, 64, 512))
    assert not c.is_identity_of((1, 64, 256))
    assert c.canonical_name((1, 64, 512)) == "vanilla"
    assert c.post_permute_shape() == (1, 64, 512)


def test_tensor_contract_reshape_and_permute():
    c = TensorContract(reshape=(1, 8, 8, 512), permutation=(2, 0, 1, 3))
    assert not c.is_identity_of((1, 64, 512))
    assert c.canonical_name((1, 64, 512)) == "r(1,8,8,512)p(2,0,1,3)"
    assert c.post_permute_shape() == (8, 1, 8, 512)


def test_tensor_contract_canonical_name_without_vanilla():
    c = TensorContract(reshape=(1, 64, 512), permutation=(0, 1, 2))
    # No vanilla supplied -> always emits r/p form (no identity collapse).
    assert c.canonical_name(None) == "r(1,64,512)p(0,1,2)"


def test_tensor_contract_rank_mismatch_rejected():
    with pytest.raises(AssertionError, match="reshape rank"):
        TensorContract(reshape=(1, 64, 512), permutation=(0, 1))


def test_tensor_contract_bad_permutation_rejected():
    with pytest.raises(AssertionError, match="permutation must be"):
        TensorContract(reshape=(1, 64, 512), permutation=(0, 0, 1))
    with pytest.raises(AssertionError, match="permutation must be"):
        TensorContract(reshape=(1, 64, 512), permutation=(0, 1, 5))


def test_tensor_contract_non_positive_dim_rejected():
    with pytest.raises(AssertionError, match="non-positive"):
        TensorContract(reshape=(1, 0, 512), permutation=(0, 1, 2))


def test_tensor_contract_is_hashable_and_equal():
    a = TensorContract(reshape=(1, 8, 8, 512), permutation=(2, 0, 1, 3))
    b = TensorContract(reshape=(1, 8, 8, 512), permutation=(2, 0, 1, 3))
    c = TensorContract(reshape=(1, 8, 8, 512), permutation=(0, 1, 2, 3))
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    # Usable as dict key.
    d = {a: "x"}
    assert d[b] == "x"


def test_vanilla_contract_for():
    c = vanilla_contract_for((64, 16, 32))
    assert c.reshape == (64, 16, 32)
    assert c.permutation == (0, 1, 2)
    assert c.is_identity_of((64, 16, 32))


# --- validate_reshape ----------------------------------------------------------


def test_validate_reshape_factorization_ok():
    validate_reshape((1, 64, 512), (1, 8, 8, 512))  # factor seq=64 into 8x8
    validate_reshape((1, 64, 512), (1, 32768))  # collapse hidden
    validate_reshape((1, 64, 512), (32768, 1))  # rank-1 + trailing 1


def test_validate_reshape_size_mismatch_rejected():
    with pytest.raises(AssertionError, match="does not preserve total"):
        validate_reshape((1, 64, 512), (1, 8, 4, 512))  # 16384 != 32768


# --- DesignEntry + library indexing -------------------------------------------


def test_design_entry_defaults():
    e = DesignEntry(dsl="def f(): pass")
    assert e.input_contracts == {}
    assert e.output_contracts == {}
    assert e.cycles == 0
    assert e.on_chip == 0
    assert e.provenance == ""


def test_freeze_contracts_stable_ordering():
    c1 = vanilla_contract_for((4,))
    c2 = vanilla_contract_for((8,))
    # Two dicts with the same content but inserted in different orders.
    a = {"b": c2, "a": c1}
    b = {"a": c1, "b": c2}
    assert freeze_contracts(a) == freeze_contracts(b)
    # Hashable.
    assert hash(freeze_contracts(a)) == hash(freeze_contracts(b))


def test_library_cell_creates_and_reuses():
    lib: NodeLibrary = {}
    inp = {"Q": vanilla_contract_for((64, 16, 32))}
    out = {"out": vanilla_contract_for((64, 512))}
    cell = library_cell(lib, inp, out)
    assert cell == []
    cell.append(DesignEntry(dsl="v0"))
    # Second call with the same contracts returns the same list.
    cell2 = library_cell(lib, inp, out)
    assert cell2 is cell
    assert len(cell2) == 1
    # Different contracts produce a different cell.
    out_alt = {"out": TensorContract(reshape=(64, 512), permutation=(1, 0))}
    cell3 = library_cell(lib, inp, out_alt)
    assert cell3 is not cell
    assert cell3 == []


# --- pareto --------------------------------------------------------------------


def _e(cycles, on_chip, dsl="x", **kw):
    return DesignEntry(dsl=dsl, cycles=cycles, on_chip=on_chip, **kw)


def test_dominates():
    a = _e(100, 100)
    b = _e(120, 120)
    c = _e(100, 120)
    d = _e(100, 100)  # equal
    assert dominates(a, b)
    assert not dominates(b, a)
    assert dominates(a, c)
    assert not dominates(a, d)  # equality is not domination
    assert not dominates(d, a)


def test_insert_pareto_accepts_nondominated():
    front = []
    assert insert_pareto(front, _e(100, 200))
    assert insert_pareto(front, _e(150, 150))
    assert insert_pareto(front, _e(200, 100))
    assert len(front) == 3


def test_insert_pareto_rejects_dominated():
    front = [_e(100, 100)]
    assert not insert_pareto(front, _e(120, 120))
    assert not insert_pareto(front, _e(100, 200))
    assert len(front) == 1


def test_insert_pareto_evicts_dominated_existing():
    front = [_e(150, 150), _e(200, 100)]
    assert insert_pareto(front, _e(100, 100))
    # Both originals are dominated by the new entry.
    assert len(front) == 1
    assert front[0].cycles == 100


def test_insert_pareto_rejects_duplicate_point():
    front = [_e(100, 100, dsl="v1")]
    assert not insert_pareto(front, _e(100, 100, dsl="v2"))
    assert len(front) == 1
    assert front[0].dsl == "v1"


def test_insert_pareto_does_not_evict_pass1_baseline():
    # The pass-1 baseline must survive Pareto culling even when an LLM
    # variant strictly dominates it on both axes: parent composition uses
    # this entry as the canonical reference (see find_pass1_baseline_entry).
    baseline = _e(200, 200, dsl="baseline", provenance="pass1_baseline")
    front = [baseline]
    assert insert_pareto(front, _e(100, 100, provenance="llm_0"))
    assert len(front) == 2
    # Baseline is still in the front by identity.
    assert any(e is baseline for e in front)


def test_insert_pareto_rejects_inserting_protected_provenance():
    # Protected entries are seeded via direct cell.append; routing them
    # through insert_pareto would silently violate "exactly one baseline".
    front = [_e(100, 100, provenance="pass1_baseline")]
    with pytest.raises(AssertionError, match="protected provenance"):
        insert_pareto(front, _e(50, 50, provenance="pass1_baseline"))


def test_protected_provenances_contains_pass1_baseline():
    # The string is referenced by name elsewhere (find_pass1_baseline_entry,
    # _seed_baseline). Catch accidental renames.
    assert "pass1_baseline" in PROTECTED_PROVENANCES


def test_cull_top_T_noop_when_under_T():
    front = [_e(100, 100), _e(120, 80)]
    cull_top_T(front, T=8)
    assert len(front) == 2


def test_cull_top_T_keeps_endpoints():
    front = [_e(c, 1000 - c) for c in (100, 110, 120, 130, 140, 150, 160, 170)]
    cull_top_T(front, T=3)
    assert len(front) == 3
    assert front[0].cycles == 100  # min-cycles endpoint
    assert front[-1].cycles == 170  # max-cycles endpoint


def test_cull_top_T_T_one():
    front = [_e(100, 100), _e(200, 50)]
    cull_top_T(front, T=1)
    # With T=1 we keep just one point; even-spacing collapses to the head.
    assert len(front) == 1


def test_dsl_dedup_hash_whitespace_invariant():
    a = "def f():\n  return  1\n"
    b = "def f():\n\treturn 1"
    assert dsl_dedup_hash(a) == dsl_dedup_hash(b)


def test_dsl_dedup_hash_distinguishes_content():
    a = "tile_row=16"
    b = "tile_row=32"
    assert dsl_dedup_hash(a) != dsl_dedup_hash(b)


# --- admit_to_cell (no-eviction accumulator) ----------------------------------


def test_admit_to_cell_appends_all_including_dominated():
    """``admit_to_cell`` accumulates every variant the search admits — no
    Pareto eviction. A dominated entry must still land in the cell so the
    library snapshot keeps it."""
    cell = []
    assert admit_to_cell(cell, _e(100, 100, dsl="a"))
    # Strictly worse on both axes — would be rejected by insert_pareto but
    # admit_to_cell keeps it.
    assert admit_to_cell(cell, _e(200, 200, dsl="b"))
    # Strictly better — would evict the worse two under insert_pareto; here
    # all three stay.
    assert admit_to_cell(cell, _e(50, 50, dsl="c"))
    assert len(cell) == 3


def test_admit_to_cell_skips_byte_identical_dsl():
    """Byte-equivalent DSL resubmissions are pure noise from a confused
    LLM — they get dropped to keep the accumulator from blowing up over
    many turns."""
    cell = []
    assert admit_to_cell(cell, _e(100, 100, dsl="def f():\n  return 1\n"))
    # Same logical DSL, different whitespace — same hash.
    assert not admit_to_cell(
        cell, _e(200, 999, dsl="def f():\n\treturn 1"),
    )
    assert len(cell) == 1


def test_admit_to_cell_rejects_protected_provenance():
    """Pass-1 baselines must be seeded via raw ``cell.append``; routing
    them through admit_to_cell would silently break the "exactly one
    pass-1 baseline" invariant."""
    cell = []
    with pytest.raises(AssertionError, match="protected provenance"):
        admit_to_cell(cell, _e(50, 50, provenance="pass1_baseline"))


# --- pareto_front_of (consumer-side filter) -----------------------------------


def test_pareto_front_of_drops_dominated():
    a = _e(100, 100)
    b = _e(150, 150)  # dominated by a
    c = _e(200, 50)
    front = pareto_front_of([a, b, c])
    assert a in front and c in front
    assert b not in front


def test_pareto_front_of_keeps_equal_score_duplicates():
    """Two entries with identical (cycles, on_chip) don't strictly
    dominate each other — both stay so callers can distinguish them by
    identity (e.g. for variant-index resolution)."""
    a = _e(100, 100, dsl="v1")
    b = _e(100, 100, dsl="v2")
    front = pareto_front_of([a, b])
    assert len(front) == 2
