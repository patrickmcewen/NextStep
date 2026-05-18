"""On-disk persistence for autotune2 node libraries.

Two pieces:

1. **Snapshot file** (``library.json``) — full ``NodeLibrary`` serialized
   with DSL source, contracts, scores, provenance, and child picks (as
   coordinates, not Python refs). Loaded post-order so each parent can
   re-resolve its ``children_picks`` against freshly-loaded child
   libraries.

2. **Stamp** — a transitive hash that identifies the inputs each node's
   library was computed against. Stamps fold in every descendant's
   stamp, so a deep DSL change ripples up the tree and invalidates every
   ancestor — preventing silent reuse of stale cycle/on_chip numbers
   that were measured against the old descendant.

The stamp's content is opaque to this module: callers assemble it from
whatever invariants they want to bind (typically plan tree shape, every
node's pass-1 DSL, parent contracts, and hw_config). ``compute_plan_stamps``
takes those pieces and produces the per-node map.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.autotune2.contracts import (
    ContractsKey,
    DesignEntry,
    NodeLibrary,
    TensorContract,
    library_cell,
)


SNAPSHOT_FILENAME = "library.json"
SNAPSHOT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Contract / key serialization
# ---------------------------------------------------------------------------


def _contract_to_json(c: TensorContract) -> dict:
    return {"reshape": list(c.reshape), "permutation": list(c.permutation)}


def _contract_from_json(d: dict) -> TensorContract:
    return TensorContract(
        reshape=tuple(d["reshape"]),
        permutation=tuple(d["permutation"]),
    )


def _contracts_dict_to_json(cs: dict[str, TensorContract]) -> list:
    # Sorted by arg name for deterministic output.
    return [[name, _contract_to_json(c)] for name, c in sorted(cs.items())]


def _contracts_dict_from_json(data: list) -> dict[str, TensorContract]:
    return {name: _contract_from_json(c) for name, c in data}


def _key_to_json(key: ContractsKey) -> list:
    return [[name, _contract_to_json(c)] for name, c in key]


def _key_from_json(data: list) -> ContractsKey:
    return tuple((name, _contract_from_json(c)) for name, c in data)


# ---------------------------------------------------------------------------
# Library save / load
# ---------------------------------------------------------------------------


def _find_entry_coords(
    child_lib: NodeLibrary, target: DesignEntry,
) -> tuple[ContractsKey, ContractsKey, int]:
    """Locate ``target`` inside ``child_lib`` and return its (in_key, out_key,
    idx_in_cell). Uses object identity — assumes the entry is currently
    present in some cell."""
    for in_key, by_out in child_lib.items():
        for out_key, cell in by_out.items():
            for idx, e in enumerate(cell):
                if e is target:
                    return in_key, out_key, idx
    raise AssertionError(
        "_find_entry_coords: entry not found in child library — "
        "children_picks may hold a reference that was Pareto-evicted. "
        "Protected provenances (e.g. pass1_baseline) must never be evicted; "
        "for LLM variants, snapshot the parent before its child cell is "
        "further mutated."
    )


def _entry_to_json(
    entry: DesignEntry,
    *,
    children_libraries: dict[str, NodeLibrary],
) -> dict:
    child_picks_json: dict[str, dict] = {}
    for child_path, child_entry in entry.children_picks.items():
        assert child_path in children_libraries, (
            f"_entry_to_json: children_picks references child_path "
            f"{child_path!r} not in children_libraries (keys: "
            f"{sorted(children_libraries)!r})"
        )
        in_k, out_k, idx = _find_entry_coords(
            children_libraries[child_path], child_entry,
        )
        child_picks_json[child_path] = {
            "in_key": _key_to_json(in_k),
            "out_key": _key_to_json(out_k),
            "idx": idx,
        }
    return {
        "dsl": entry.dsl,
        "input_contracts": _contracts_dict_to_json(entry.input_contracts),
        "output_contracts": _contracts_dict_to_json(entry.output_contracts),
        "cycles": entry.cycles,
        "on_chip": entry.on_chip,
        "provenance": entry.provenance,
        "children_picks": child_picks_json,
    }


def _entry_from_json(
    data: dict,
    *,
    children_libraries: dict[str, NodeLibrary],
) -> DesignEntry:
    children_picks: dict[str, DesignEntry] = {}
    for child_path, coord in data["children_picks"].items():
        assert child_path in children_libraries, (
            f"_entry_from_json: serialized children_picks references "
            f"child_path {child_path!r} not in children_libraries "
            f"(keys: {sorted(children_libraries)!r}); load children first"
        )
        in_k = _key_from_json(coord["in_key"])
        out_k = _key_from_json(coord["out_key"])
        children_picks[child_path] = (
            children_libraries[child_path][in_k][out_k][coord["idx"]]
        )
    return DesignEntry(
        dsl=data["dsl"],
        input_contracts=_contracts_dict_from_json(data["input_contracts"]),
        output_contracts=_contracts_dict_from_json(data["output_contracts"]),
        cycles=data["cycles"],
        on_chip=data["on_chip"],
        provenance=data["provenance"],
        children_picks=children_picks,
    )


def serialize_library(
    lib: NodeLibrary,
    *,
    stamp: str,
    children_libraries: dict[str, NodeLibrary],
) -> dict:
    """Build the JSON payload for ``lib``. ``children_libraries`` is needed
    to resolve each entry's ``children_picks`` references back to their
    coordinates (in_key, out_key, idx) in the child library."""
    # Skip empty cells: they're meaningless (no Pareto entries to score
    # against) and would trip render_library_as_variant_summaries on the
    # next load. The in-memory library shouldn't contain them either, but
    # filtering here keeps the snapshot self-consistent regardless of the
    # producer.
    cells_json = []
    for in_key, by_out in lib.items():
        for out_key, cell in by_out.items():
            if not cell:
                continue
            cells_json.append({
                "in_key": _key_to_json(in_key),
                "out_key": _key_to_json(out_key),
                "entries": [
                    _entry_to_json(e, children_libraries=children_libraries)
                    for e in cell
                ],
            })
    return {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "stamp": stamp,
        "cells": cells_json,
    }


def deserialize_library(
    data: dict,
    *,
    children_libraries: dict[str, NodeLibrary],
) -> NodeLibrary:
    assert data["format_version"] == SNAPSHOT_FORMAT_VERSION, (
        f"deserialize_library: unsupported format_version "
        f"{data['format_version']!r}; expected {SNAPSHOT_FORMAT_VERSION}"
    )
    lib: NodeLibrary = {}
    for cell_data in data["cells"]:
        # Skip empty cells from older snapshots written before search_parent
        # learned to lazily allocate cells. Reconstituting them would trip
        # render_library_as_variant_summaries on the very next prompt build.
        if not cell_data["entries"]:
            continue
        in_contracts = _contracts_dict_from_json(cell_data["in_key"])
        out_contracts = _contracts_dict_from_json(cell_data["out_key"])
        cell = library_cell(lib, in_contracts, out_contracts)
        for entry_data in cell_data["entries"]:
            cell.append(_entry_from_json(
                entry_data, children_libraries=children_libraries,
            ))
    return lib


def save_library_snapshot(
    lib: NodeLibrary,
    *,
    path: Path,
    stamp: str,
    children_libraries: dict[str, NodeLibrary],
) -> None:
    """Atomically write ``lib`` to ``path`` (tmp + os.replace).

    Atomic rename guarantees that an interrupted run leaves either the
    old file or the new one — never a half-written snapshot that would
    parse but produce a wrong library on resume.
    """
    payload = serialize_library(
        lib, stamp=stamp, children_libraries=children_libraries,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def try_load_library_snapshot(
    path: Path,
    *,
    expected_stamp: str,
    children_libraries: dict[str, NodeLibrary],
) -> NodeLibrary | None:
    """Return the loaded library iff ``path`` exists and its stamp matches
    ``expected_stamp``. Returns None otherwise — caller treats that as
    "snapshot unusable; re-run this node"."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("stamp") != expected_stamp:
        return None
    return deserialize_library(data, children_libraries=children_libraries)


# ---------------------------------------------------------------------------
# Stamp computation
# ---------------------------------------------------------------------------


def _stable_hash(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def compute_node_stamp(
    *,
    node_path: str,
    pass1_dsl: str,
    parent_contract_repr: Any,
    extra: dict[str, Any],
    children_stamps: dict[str, str],
) -> str:
    """Single-node stamp. ``parent_contract_repr`` is a JSON-serializable
    structural summary of the parent contract (or None for the root).
    ``extra`` folds in anything else that should invalidate this stamp
    (e.g., ``hw_config`` and any scorer/verifier config that changes
    cycle/on_chip outputs). ``children_stamps`` is the per-immediate-child
    stamp map — transitively embeds the whole subtree."""
    parts = {
        "node_path": node_path,
        "pass1_dsl": pass1_dsl,
        "parent_contract": parent_contract_repr,
        "extra": extra,
        "children_stamps": dict(sorted(children_stamps.items())),
    }
    return _stable_hash(parts)


def contract_structural_repr(contract: Any) -> Any:
    """JSON-friendly structural summary of a ``Contract`` for stamping.

    Includes only shape/spec metadata — not the live ``tiled_values`` /
    ``tiled_outputs`` tensors (those have non-deterministic memory ids
    that would defeat stamp stability). Returns ``None`` when given
    ``None`` so callers can pass the root's missing contract through.
    """
    if contract is None:
        return None
    return {
        "arg_names": list(contract.arg_names),
        "vanilla_shapes": [list(s) for s in contract.vanilla_shapes],
        "tiled_shapes": [list(s) for s in contract.tiled_shapes],
        "out_shapes": [list(s) for s in contract.out_shapes],
        "arg_specs": [repr(s) for s in contract.arg_specs],
        "arg_is_raw": list(contract.arg_is_raw),
        "out_is_tuple": bool(contract.out_is_tuple),
    }


def compute_plan_stamps(
    *,
    plan_tree,
    pass1_dsls: dict[str, str],
    pass1_contracts: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, str]:
    """Walk ``plan_tree`` post-order and produce ``{node_path: stamp}``.

    Stamps are transitive — each parent folds in its children's stamps —
    so a single descendant DSL change ripples up to invalidate every
    ancestor. ``extra`` is mixed into every node's stamp so any change
    to hw_config / scorer config invalidates the whole tree at once.
    """
    stamps: dict[str, str] = {}
    for node in plan_tree.iter_topological():
        children_stamps = {c.path: stamps[c.path] for c in node.children}
        parent_contract = pass1_contracts.get(node.path)
        stamps[node.path] = compute_node_stamp(
            node_path=node.path,
            pass1_dsl=pass1_dsls[node.path],
            parent_contract_repr=contract_structural_repr(parent_contract),
            extra=extra,
            children_stamps=children_stamps,
        )
    return stamps
