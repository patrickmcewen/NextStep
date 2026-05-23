"""Diagnostic: detect "reducing-diamond" deadlocks in a STeP graph.

The Rust simulator's ``BroadcastContext`` (step-perf/src/operator/broadcast.rs)
applies backpressure on **all** target FIFOs before emitting the next token.
That gives a deadlock recipe:

    X ──Broadcast──┬──► Accum(R) ──► ... ──► op ─┐
                   │                              ▼
                   └──────────────────────────► merge_op

If one arm of the diamond reduces over R tokens before producing its first
output, the other arm must hold those R tokens in flight before the merge can
fire. With ``SimConfig(channel_depth=C)`` and ``C < R``, every Broadcast target
fills up at depth C, the Broadcast stalls, and the reduction can never finish.

This module's ``find_deadlocks`` reports the pattern (skew > channel_depth).
It is NOT wired into the eval pipeline — ``evaluate.py`` runs the rust sim
with ``channel_depth=1024``, big enough that the broadcast never stalls on
the in-flight windows we see in real LLM-generated kernels. The function
is kept for ad-hoc diagnosis (e.g. when investigating a new hang or porting
to a real-hardware FIFO depth).

The check only covers the SDF subset (static rates, no cycles, no dynamic
ops) and uses ``max`` over an arm rather than the true multiplicative
composition. See ``_PHASE_2_NOTES`` for the gap between this and a complete
one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import networkx as nx


# Notes for Phase 2 (proper SDF buffer-sizing). When the MVP starts producing
# false positives or missing real deadlocks, the next steps are:
#   - Replace ``max`` over an arm with multiplicative composition of holdbacks.
#     Chained reductions multiply: Accum(R1) -> Accum(R2) gives first-output
#     latency R1*R2, not max(R1, R2).
#   - Handle rate-expanding ops (RepeatStatic, RepeatRef) by tracking
#     per-edge token rates. Solve the SDF balance equations to recover the
#     true in-flight count per edge.
#   - Credit the fast arm with FIFO slack from intermediate ops: an arm with
#     m intermediate ops has (m+1)*channel_depth capacity, not just C.
#     The MVP under-counts slack and may over-report on graphs that would
#     actually run (false positives).
#   - Generalize "Broadcast" to any node whose stream is consumed by >=2
#     downstream ops (in case some lowering path skips ``infer_broadcast``).
#   - Dynamic-rate ops (DynMatmul, DynLinearOffChipLoad, ExpertAddrGen,
#     FlatmapFilterRowStreamify): the rate depends on runtime data, so a
#     static check can't be exact. Emit ``UNANALYZABLE`` flags rather than
#     silently passing.
_PHASE_2_NOTES = __doc__  # keep alongside for grep-ability.


@dataclass
class Violation:
    """One reducing-diamond deadlock found in the graph.

    A violation describes a single (broadcast, arm-pair, merge) triple where
    the per-arm holdback skew exceeds ``channel_depth``.
    """
    broadcast_id: int
    broadcast_label: str
    slow_arm_id: int
    slow_arm_label: str
    slow_arm_holdback: int
    fast_arm_id: int
    fast_arm_label: str
    fast_arm_holdback: int
    merge_id: int
    merge_label: str
    channel_depth: int

    @property
    def required_depth(self) -> int:
        return abs(self.slow_arm_holdback - self.fast_arm_holdback)

    def explain(self) -> str:
        return (
            f"reducing-diamond deadlock through {self.broadcast_label}: "
            f"slow arm via {self.slow_arm_label} holds back "
            f"{self.slow_arm_holdback} tokens, fast arm via "
            f"{self.fast_arm_label} holds back {self.fast_arm_holdback} "
            f"before merge at {self.merge_label} "
            f"(channel_depth={self.channel_depth}, required={self.required_depth})"
        )


def _op_class_name(node) -> str:
    return type(node).__name__


def _input_stream(node):
    """Return the input stream of an op for shape lookup.

    Uses the same ``get_stream`` helper that step_py ops use internally.
    Imported lazily so this module doesn't pull in step_py at import time
    (validate_deadlock can be loaded by tooling that doesn't have step_py
    on its path).
    """
    from step_py.ops import get_stream  # type: ignore
    raw = node._input if hasattr(node, "_input") else node.input
    return get_stream(raw)


def _holdback(node) -> int:
    """Broadcast-tokens consumed before ``node`` produces its first output.

    For Accum (the only op in ``_HOLDBACK_OPS``), this is the reduction
    window size — product of input-stream dims along the reduced axes.
    Symbolic / dynamic dims return ``math.inf`` so the resulting skew
    always exceeds any finite ``channel_depth`` and is conservatively
    flagged. All other ops are passthrough for first-output timing and
    return 1.
    """
    cls = _op_class_name(node)
    if cls == "Accum":
        assert hasattr(node, "accum_rank"), (
            "Accum node missing accum_rank attribute; step_py contract violated"
        )
        in_stream = _input_stream(node)
        prod = 1
        for d in in_stream.shape[-node.accum_rank:]:
            if isinstance(d, int):
                prod *= d
            else:
                return math.inf  # symbolic / dynamic dim
        return prod
    return 1


# Ops that don't change the *first-output-to-merge* timing on an arm.
#
# For the deadlock check we care about: starting from the broadcast pushing
# its first token, how many broadcast tokens must be consumed before this
# arm produces its first output reaching the merge? Token-rate-expanding
# ops (RetileStreamify 1->N, RepeatStatic 1->K, ExpandRef 1->ref) all still
# emit their FIRST output after just 1 input token; the expansion shows up
# in later tokens. Likewise rank-rearranging ops (Promote, PromoteOuter,
# Reshape) and Buffer<->Tile transforms (Streamify) are 1-in/1-out for the
# first emission. So all of these are safe to skip past when computing
# first-output holdback.
#
# Excluded: FlatPartition/FlatReassemble/Parallelize/StaticReassemble have
# data-dependent token routing the MVP can't model, and Bufferize has its
# own R-token holdback already covered by _HOLDBACK_OPS.
_RATE_PASSTHROUGH = frozenset({
    "UnaryMap", "BinaryMap", "Flatten",
    "Promote", "PromoteOuter", "Reshape",
    "ExpandRef", "RetileStreamify", "RepeatStatic", "RepeatRef",
    "Streamify",
    # Token-rate-preserving fanout / routing. A nested Broadcast on an arm
    # routes the same per-broadcast-token rate to its branches (shortest_path
    # picks one branch toward the merge). Parallelize splits round-robin to
    # one of N branches per token, StaticReassemble pulls round-robin from N
    # inputs — net 1-in/1-out at the outer broadcast's token cadence. These
    # appear on the slow arm of rotate_half-style diamonds (e.g. Broadcast#201
    # in prefill_transformer_simple); without them the Accum's holdback is
    # invisible to the check. FlatPartition/FlatReassemble are the data-
    # dependent analogues (MoE expert routing): only the selected branch
    # sees the token, but per-arm rate is still 1-in/1-out at first-emission.
    "Broadcast", "Parallelize", "StaticReassemble",
    "FlatPartition", "FlatReassemble",
    # Bufferize absorbs tokens to PMU without backpressuring its input —
    # the broadcast feeding it can push freely. Bufferize on either arm
    # therefore breaks the deadlock chain. The token-cadence holdback
    # Bufferize creates is DOWNSTREAM of itself, not upstream, so it
    # doesn't gate the broadcast's pushes — Bufferize must NOT be in
    # _HOLDBACK_OPS.
    "Bufferize",
})

# Ops whose upstream input is blocked until they consume R tokens. Accum
# accumulates R input tokens before producing its first output, and while
# accumulating it backpressures its input FIFO. That's the upstream
# backpressure the broadcast can't escape. Bufferize is conspicuously NOT
# here — see the comment in _RATE_PASSTHROUGH above.
_HOLDBACK_OPS = frozenset({"Accum"})


def _arm_holdback(graph: nx.MultiDiGraph, src, sink):
    """Holdback in broadcast-token units along the path ``src -> ... -> sink``.

    Returns an int when the arm is rate-comparable to the broadcast (every
    intermediate op is pass-through or a recognized holdback op), or
    ``None`` when the arm contains a rate-altering op the MVP can't model
    safely. ``None`` causes the caller to skip the violation — this is the
    conservative behavior; some true deadlocks will be missed (see
    ``_PHASE_2_NOTES``).

    The MVP also uses ``max`` over holdback ops on the arm rather than the
    true multiplicative composition; with the rate-comparability gate above,
    chained reductions are extremely rare on a single arm.

    ``sink`` is the merge op itself, on neither arm — only nodes strictly
    between the broadcast successor and the merge count.
    """
    path = nx.shortest_path(graph, src, sink)
    arm_nodes = path[:-1]  # exclude the merge sink; include src
    if not arm_nodes:
        return 1
    arm_max = 1
    for n in arm_nodes:
        cls = _op_class_name(n)
        if cls in _HOLDBACK_OPS:
            arm_max = max(arm_max, _holdback(n))
        elif cls in _RATE_PASSTHROUGH:
            continue
        else:
            # Rate-altering or unknown op — bail out conservatively.
            return None
    return arm_max


def _arm_absorbs(graph: nx.MultiDiGraph, src, sink) -> bool:
    """True if any op on the path ``src -> ... -> sink`` is a Bufferize.

    Bufferize writes its input to PMU at memory bandwidth without
    backpressuring its upstream. A Broadcast feeding into an absorbing arm
    can therefore push as many tokens as PMU has room for, regardless of
    what's gating downstream consumers. So an arm with Bufferize is safe
    from the reducing-diamond pattern even when the other arm has a slow
    Accum — there's no FIFO to fill against the broadcast.

    Both arms-with-user-written Bufferize (intentional fanout shielding)
    and arms with our injected ``Bufferize + Streamify`` repair return True
    here, which is why ``find_deadlocks`` converges after repair.
    """
    path = nx.shortest_path(graph, src, sink)
    return any(_op_class_name(n) == "Bufferize" for n in path[:-1])


def find_deadlocks(
    graph: nx.MultiDiGraph,
    channel_depth: int = 2,
) -> List[Violation]:
    """Scan ``graph`` for reducing-diamond deadlocks.

    Returns a list of :class:`Violation`, one per (broadcast, arm-pair,
    merge) triple where the holdback skew exceeds ``channel_depth``.
    An empty list means no deadlocks under the MVP model — it does NOT
    guarantee the graph is safe in general (see ``_PHASE_2_NOTES``).

    Currently only inspects ``Broadcast`` nodes; the MVP assumes
    ``infer_broadcast`` has been run so all multi-consumer fanouts are
    explicit.
    """
    assert channel_depth >= 1, f"channel_depth must be >= 1, got {channel_depth}"
    violations: List[Violation] = []
    for node in list(graph.nodes):
        if _op_class_name(node) != "Broadcast":
            continue
        consumers = list(graph.successors(node))
        if len(consumers) < 2:
            continue
        # Filter broadcasts whose stream is too short to ever stall — if
        # the broadcast pushes <= channel_depth tokens total, every
        # consumer FIFO accepts everything and no backpressure arises
        # regardless of downstream Accum holdbacks. Avoids tiny-preset
        # false positives like Broadcast over a (1,1) stream where a
        # nominal Accum.R=16 reflects the post-RetileStreamify rate,
        # not the broadcast rate the FIFO actually sees.
        bcast_stream = node.stream_idx(0)
        if all(isinstance(d, int) for d in bcast_stream.shape):
            total_tokens = 1
            for d in bcast_stream.shape:
                total_tokens *= d
            if total_tokens <= channel_depth:
                continue
        for i in range(len(consumers)):
            for j in range(i + 1, len(consumers)):
                c_i, c_j = consumers[i], consumers[j]
                desc_i = nx.descendants(graph, c_i) | {c_i}
                desc_j = nx.descendants(graph, c_j) | {c_j}
                merges = desc_i & desc_j
                if not merges:
                    continue
                # Pick the topologically-closest merge: among shared
                # descendants, the one with smallest combined path
                # length from both arms. This is the first place the
                # diamond closes, which is where the FIFO requirement
                # actually bites.
                merge = min(
                    merges,
                    key=lambda m: (
                        nx.shortest_path_length(graph, c_i, m)
                        + nx.shortest_path_length(graph, c_j, m)
                    ),
                )
                R_i = _arm_holdback(graph, c_i, merge)
                R_j = _arm_holdback(graph, c_j, merge)
                if R_i is None or R_j is None:
                    # One arm has a rate-altering op the MVP can't model;
                    # skip rather than risk a false positive.
                    continue
                if _arm_absorbs(graph, c_i, merge) or _arm_absorbs(graph, c_j, merge):
                    # A Bufferize on either arm decouples the broadcast from
                    # the merge's gating — no FIFO can fill against the
                    # broadcast, so the diamond can't deadlock. Avoids false
                    # positives when a kernel author hand-bufferizes a fanout.
                    continue
                skew = abs(R_i - R_j) if math.isfinite(R_i) and math.isfinite(R_j) else math.inf
                if skew > channel_depth:
                    slow_c, slow_R, fast_c, fast_R = (
                        (c_i, R_i, c_j, R_j) if R_i > R_j else (c_j, R_j, c_i, R_i)
                    )
                    violations.append(Violation(
                        broadcast_id=node.instance_id,
                        broadcast_label=str(node),
                        slow_arm_id=slow_c.instance_id,
                        slow_arm_label=str(slow_c),
                        slow_arm_holdback=int(slow_R) if math.isfinite(slow_R) else -1,
                        fast_arm_id=fast_c.instance_id,
                        fast_arm_label=str(fast_c),
                        fast_arm_holdback=int(fast_R) if math.isfinite(fast_R) else -1,
                        merge_id=merge.instance_id,
                        merge_label=str(merge),
                        channel_depth=channel_depth,
                    ))
    return violations
