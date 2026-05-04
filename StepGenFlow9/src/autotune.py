"""Autotuner orchestration.

Takes a correctness-verified `build_graph(dims, tensors)` produced by the
StepGenFlow pipeline (or loaded from a past successful checkpoint) and runs
an agent loop that iteratively proposes performance-oriented rewrites.
Each proposal is checked for correctness against the reference, then fed
through the analytical timing model; the report is sent back to the agent.

The autotuner never mutates the algorithm — it only changes knobs like
tile_row / tile_col, par_dispatch, compute_bw, write_back_mu, and the
choice of buffering / broadcast / retile ops.
"""

import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import sympy
import yaml

from agents import Runner

from src.agents import make_autotune_agent
from src.prompts import build_autotune_user_prompt
from src.tools import _exec_build_graph

# ---------------------------------------------------------------------------
# Path setup — mirror orchestrator.py
# ---------------------------------------------------------------------------
_DEIO_ROOT = Path(__file__).resolve().parent.parent.parent
_STEPDB_DIR = _DEIO_ROOT / "StepDB"
_STEP_TL_SRC = _DEIO_ROOT / "step_tl" / "src"
_STEP_TL_PROTO = _STEP_TL_SRC / "proto"

for p in (_STEPDB_DIR, _STEP_TL_SRC, _STEP_TL_PROTO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from precompute import precompute_tensors  # noqa: E402  (StepDB/precompute.py)

from timing_and_emulator.timing import analyze_timing  # noqa: E402

# Reuse the orchestrator's correctness checker so we verify identically to
# the implementer pipeline.
from src.orchestrator import (  # noqa: E402
    _run_dsl_correctness,
    _run_graph_correctness,
    _write,
    _extract_code,
    _reasoning_text,
    _resolve_resume_dsl,
)
from src.dsl_to_step import translate  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint resume — locate a passing build_graph from a previous run
# ---------------------------------------------------------------------------

def _find_passing_extract(search_root: Path) -> Path:
    """Find the newest `extracted_code.py` whose sibling `status.txt` says PASS.

    Used to locate the final build_graph produced by a successful translate
    pass under a given checkpoint sub-tree.
    """
    candidates = []
    for status_file in search_root.rglob("status.txt"):
        if status_file.read_text().strip().startswith("PASS"):
            code_file = status_file.parent / "extracted_code.py"
            if code_file.is_file():
                candidates.append(code_file)
    assert candidates, (
        f"No turn with status.txt=='PASS' + extracted_code.py found under {search_root}. "
        f"Autotuning requires a successful translate pass as its starting point."
    )
    # Prefer the latest-modified match (deepest outer/turn iteration usually wins).
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _resolve_resume_build_graph(resume_from: str, kernel_name: str) -> tuple[str, Path]:
    """Resolve `resume_from` to (build_graph_code, source_path).

    Accepts:
      - Path to a .py file directly.
      - Path to a turn directory containing extracted_code.py + PASSing status.txt.
      - Path to an outer_N, translate/, or kernel directory — searches recursively
        for the last passing extracted_code.py.
      - Path to a checkpoint root (the timestamped dir above the kernel name).
    """
    p = Path(resume_from)
    assert p.exists(), f"resume_from path does not exist: {p}"

    if p.is_file():
        assert p.suffix == ".py", f"resume_from file must be .py: {p}"
        return p.read_text(), p

    # Directory case. Prefer a kernel-scoped subtree if present so we don't
    # accidentally pick up a passing turn from a different kernel.
    kernel_dir = p / kernel_name
    search_root = kernel_dir if kernel_dir.is_dir() else p
    chosen = _find_passing_extract(search_root)
    print(f"  Resolved resume path: {chosen}")
    return chosen.read_text(), chosen


# ---------------------------------------------------------------------------
# StepDB config — dims for kernel/preset
# ---------------------------------------------------------------------------

def _load_stepdb_config() -> dict:
    config_path = _STEPDB_DIR / "bench_config.yaml"
    assert config_path.exists(), f"bench_config.yaml not found at {config_path}"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Timing model wrapper — run analyze_timing and format a compact report
# ---------------------------------------------------------------------------

def _sym_to_int(expr) -> int:
    """Collapse a sympy expression to a concrete int, substituting any
    remaining free symbols with 1 so downstream code can compare cycles.
    The main `analyze_timing` run already substitutes DynDim values, so
    this is a defensive fallback for leftover symbols.
    """
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        expr = expr.xreplace({s: 1 for s in expr.free_symbols})
    return int(sympy.N(expr))


def _sym_to_float(expr) -> float:
    """Float version of _sym_to_int (for OTI and other potentially-fractional quantities)."""
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        expr = expr.xreplace({s: 1 for s in expr.free_symbols})
    return float(sympy.N(expr))


def _node_label(n) -> str:
    """Pretty label for a StepOps node: '[id] OpType' or '[id] OpType<FnName>'."""
    label = f"[{n.instance_id}] {n.__class__.__name__}"
    fn = getattr(n, "fn", None)
    if fn is not None:
        label += f"<{fn.__class__.__name__}>"
    return label


def _build_verbose_report(graph, result, hw_config: dict) -> str:
    """Verbose per-node timing + memory report.

    Sections:
      1. Memory totals — summed on-chip requirement and off-chip traffic
         across the whole graph (with PMU-cap utilization if known).
      2. Graph structure — predecessors/successors for each node.
      3. Per-node block — timing (T_fire/OTPC/N_fire/OCI/OTI/ICI/ICD/st/
         fto/end), per-node on-chip / off-chip memory, an OCI
         max-breakdown labeled by the predecessor each candidate term
         comes from.
      4. Critical path — latency chain (st argmax) and throughput origin
         (OCI argmax) from the last-finishing leaf.
    """
    info = result["per_node"]
    sym_subs = result.get("sym_subs", {}) or {}

    def _sub(expr):
        if sym_subs and hasattr(expr, "free_symbols") and expr.free_symbols:
            return expr.xreplace(sym_subs)
        return expr

    # ------------------------------------------------------------------
    # Per-node memory metrics. Cached so the totals section and the
    # per-node section read identical values.
    # ------------------------------------------------------------------
    on_chip_bytes: dict[int, int] = {}
    off_chip_bytes: dict[int, int] = {}
    for nid, i in info.items():
        n = i["node"]
        on_chip_bytes[nid] = _sym_to_int(_sub(n.on_chip_requirement(count_fifos=False)))
        off_chip_bytes[nid] = _sym_to_int(_sub(n.off_chip_traffic()))
    total_on_chip = sum(on_chip_bytes.values())
    total_off_chip = sum(off_chip_bytes.values())
    pmu_cap = hw_config.get("pmu_buffer_bytes")

    total = _sym_to_int(result["total_cycles"])
    lines = [f"total_cycles={total}", ""]

    # ------------------------------------------------------------------
    # 1. Memory totals
    # ------------------------------------------------------------------
    lines.append("=" * 72)
    lines.append("MEMORY USAGE  (bytes)")
    lines.append("=" * 72)
    if pmu_cap:
        pct = 100.0 * total_on_chip / pmu_cap
        lines.append(
            f"  on-chip  (sum of on_chip_requirement, count_fifos=False): "
            f"{total_on_chip}  ({pct:.1f}% of pmu_buffer_bytes={pmu_cap})"
        )
    else:
        lines.append(
            f"  on-chip  (sum of on_chip_requirement, count_fifos=False): {total_on_chip}"
        )
    lines.append(
        f"  off-chip (sum of off_chip_traffic over kernel run):        {total_off_chip}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # 1. Graph structure
    # ------------------------------------------------------------------
    lines.append("=" * 72)
    lines.append("GRAPH STRUCTURE  (topological)")
    lines.append("=" * 72)
    for nid, i in info.items():
        n = i["node"]
        preds = sorted({p.instance_id for p in graph.predecessors(n)})
        succs = sorted({s.instance_id for s in graph.successors(n)})
        pred_str = ",".join(str(x) for x in preds) if preds else "-"
        succ_str = ",".join(str(x) for x in succs) if succs else "-"
        lines.append(f"  {_node_label(n):<40}  preds=[{pred_str}]  succs=[{succ_str}]")
    lines.append("")

    # ------------------------------------------------------------------
    # 2. Per-node detail
    # ------------------------------------------------------------------
    lines.append("=" * 72)
    lines.append("PER-NODE TIMING  (cycles)")
    lines.append("=" * 72)

    # Collect per-node NIT maps for the critical-path section.
    nit_maps: dict[int, dict[int, int]] = {}

    for nid, i in info.items():
        n = i["node"]
        t_fire = _sym_to_int(i["T_fire"])
        n_fire = _sym_to_int(i["N_fire"])
        otpc = _sym_to_int(i["OTPC"])
        oci = _sym_to_int(i["OCI"])
        oti_f = _sym_to_float(i["OTI"])
        ici = _sym_to_int(i["ICI"]) if "ICI" in i else 0
        icd = _sym_to_int(i["ICD"])
        st = _sym_to_int(i["st"])
        fto = _sym_to_int(i["fto"])
        end = _sym_to_int(i["end"])

        nit_raw = n.NIT()
        nit_map = {k: int(sympy.N(_sub(v))) for k, v in nit_raw.items()}
        nit_maps[nid] = nit_map

        preds = list({p.instance_id: p for p in graph.predecessors(n)}.values())

        lines.append(_node_label(n))
        lines.append(f"  params:   T_fire={t_fire}  OTPC={otpc}  N_fire={n_fire}")
        lines.append(f"  times:    st={st}  ICD={icd}  fto={fto}  end={end}")
        lines.append(f"  rates:    OCI={oci}  OTI={oti_f:.2f}  ICI={ici}")
        lines.append(
            f"  memory:   on_chip={on_chip_bytes[nid]} B  off_chip={off_chip_bytes[nid]} B"
        )

        if preds:
            lines.append(f"  inputs:")
            for p in preds:
                pi = info[p.instance_id]
                p_oti = _sym_to_float(pi["OTI"])
                nit = nit_map.get(p.instance_id, 1)
                lines.append(
                    f"    ← [{p.instance_id}] {p.__class__.__name__:<28}  OTI={p_oti:>8.2f}  NIT={nit}"
                )

        # OCI max-breakdown, with each candidate labeled by its source.
        #   compute op:   OCI = max( T_fire,   max_p  NIT_p * max(T_fire, pred_OTI_p) )
        #   off-chip op:  OCI = max( HBM_term, max_p  NIT_p * pred_OTI_p              )
        # (One-shot pred case — pred_n_fire <= NIT — uses NIT * T_fire; see timing.py:238-241.)
        candidates = []  # list of (source_tag, value, detail)
        if n.is_offchip_memory_op():
            for p in preds:
                pi = info[p.instance_id]
                p_oti = _sym_to_float(pi["OTI"])
                nit = nit_map.get(p.instance_id, 1)
                val = nit * p_oti
                candidates.append((
                    f"← [{p.instance_id}]",
                    val,
                    f"NIT({nit}) * OTI({p_oti:.2f})  (ICI term)",
                ))
            ici_max = max((v for _, v, _ in candidates), default=0.0)
            if oci > ici_max:
                hbm_oti_est = oci / max(1, otpc)
                candidates.append((
                    "[self]",
                    float(oci),
                    f"OTPC({otpc}) * hbm_oti(~{hbm_oti_est:.1f})  (HBM term)",
                ))
        else:
            candidates.append((
                "[self]",
                float(t_fire),
                f"T_fire({t_fire})",
            ))
            for p in preds:
                pi = info[p.instance_id]
                p_oti = _sym_to_float(pi["OTI"])
                p_n_fire = _sym_to_int(pi["N_fire"])
                nit = nit_map.get(p.instance_id, 1)
                if p_n_fire <= nit:
                    val = nit * t_fire
                    detail = (f"NIT({nit}) * T_fire({t_fire})  "
                              f"(one-shot: pred N_fire={p_n_fire} ≤ NIT)")
                else:
                    gate_val = max(t_fire, p_oti)
                    inner = "T_fire" if t_fire >= p_oti else "pred_OTI"
                    val = nit * gate_val
                    detail = (f"NIT({nit}) * max(T_fire={t_fire}, OTI={p_oti:.2f})  "
                              f"= {nit} * {gate_val:.2f}   (inner max won by {inner})")
                candidates.append((
                    f"← [{p.instance_id}]",
                    val,
                    detail,
                ))

        if candidates:
            lines.append(f"  OCI = max of:")
            best_val = max(v for _, v, _ in candidates)
            for tag, val, detail in candidates:
                win = abs(val - best_val) < 1e-6
                marker = "  <-- WINS" if win else ""
                lines.append(
                    f"    {tag:<10} {val:>10.2f}   {detail}{marker}"
                )
        lines.append("")

    # ------------------------------------------------------------------
    # 3. Critical path
    # ------------------------------------------------------------------
    lines.append("=" * 72)
    lines.append("CRITICAL PATH")
    lines.append("=" * 72)

    leaf_nid, leaf_info = max(info.items(), key=lambda kv: _sym_to_int(kv[1]["end"]))
    leaf_n = leaf_info["node"]
    leaf_fto = _sym_to_int(leaf_info["fto"])
    leaf_oti = _sym_to_float(leaf_info["OTI"])
    leaf_n_fire = _sym_to_int(leaf_info["N_fire"])
    leaf_steady = max(0, total - leaf_fto)
    pct_steady = 100.0 * leaf_steady / max(1, total)
    regime = "throughput-bound" if pct_steady >= 50 else "latency-bound"

    lines.append(f"  last-finishing node: {_node_label(leaf_n)}  (end={total})")
    lines.append(
        f"  end = fto + (N_fire-1) * OTI = {leaf_fto} + {leaf_n_fire-1} * {leaf_oti:.2f} "
        f"= {leaf_fto} + {leaf_steady}"
    )
    lines.append(f"  regime: {regime}  (steady-state {pct_steady:.1f}% of total)")
    lines.append("")

    # -- Latency chain: walk st = max_p fto(p) back from leaf --------------
    lines.append("  latency chain  (walks st = max_pred fto(pred)):")
    latency_chain = []  # source → leaf
    cur = leaf_n
    latency_chain.append(cur)
    while True:
        cur_st = _sym_to_int(info[cur.instance_id]["st"])
        if cur_st == 0:
            break
        chosen = None
        for p in graph.predecessors(cur):
            if _sym_to_int(info[p.instance_id]["fto"]) == cur_st:
                chosen = p
                break
        if chosen is None:
            break
        latency_chain.append(chosen)
        cur = chosen
    latency_chain.reverse()

    running = 0
    for idx, node in enumerate(latency_chain):
        nid = node.instance_id
        i = info[nid]
        tf = _sym_to_int(i["T_fire"])
        icd_n = _sym_to_int(i["ICD"])
        st_n = _sym_to_int(i["st"])
        fto_n = _sym_to_int(i["fto"])
        arrow = "    " if idx == 0 else " →  "
        lines.append(
            f"  {arrow}{_node_label(node):<40}  st={st_n:<6}+ ICD={icd_n:<6}+ T_fire={tf:<6}= fto={fto_n}"
        )
        running = fto_n
    lines.append(f"     → total fto (startup delay) = {running}")
    lines.append("")

    # -- Throughput origin: walk OCI argmax back to the node whose own
    #    T_fire (or HBM rate) sets the bottleneck OTI ----------------------
    lines.append("  throughput origin  (walks OCI argmax until a self-gated node):")
    thru_chain = []
    cur = leaf_n
    thru_chain.append(cur)
    while True:
        n = cur
        nid_c = n.instance_id
        i = info[nid_c]
        oci_c = _sym_to_int(i["OCI"])
        tf_c = _sym_to_int(i["T_fire"])

        if n.is_offchip_memory_op():
            break  # HBM / off-chip origin

        preds = list({p.instance_id: p for p in graph.predecessors(n)}.values())
        # Did a pred's OTI win the inner max (pred_OTI > T_fire) AND drive OCI?
        best_pred = None
        best_val = -1.0
        for p in preds:
            pi = info[p.instance_id]
            p_oti = _sym_to_float(pi["OTI"])
            nit = nit_maps[nid_c].get(p.instance_id, 1)
            if p_oti > tf_c:
                val = nit * p_oti
                if val > best_val and abs(val - oci_c) < 1.0:
                    best_val = val
                    best_pred = p
        if best_pred is None:
            break  # self-gated
        thru_chain.append(best_pred)
        cur = best_pred
    thru_chain.reverse()

    origin = thru_chain[0]
    origin_info = info[origin.instance_id]
    origin_oti = _sym_to_float(origin_info["OTI"])
    origin_t_fire = _sym_to_int(origin_info["T_fire"])

    for idx, node in enumerate(thru_chain):
        nid = node.instance_id
        i = info[nid]
        oti_n = _sym_to_float(i["OTI"])
        oci_n = _sym_to_int(i["OCI"])
        tf_n = _sym_to_int(i["T_fire"])
        arrow = "    " if idx == 0 else " →  "
        lines.append(
            f"  {arrow}{_node_label(node):<40}  T_fire={tf_n:<6}OCI={oci_n:<8}OTI={oti_n:.2f}"
        )

    if origin.is_offchip_memory_op():
        reason = f"HBM-gated (off-chip memory op)"
    else:
        reason = f"self-gated by T_fire={origin_t_fire}"
    lines.append(f"     → OTI origin: {_node_label(origin)}  ({reason})")
    lines.append(f"     → steady-state cost = (N_fire_leaf - 1) * OTI = "
                 f"{leaf_n_fire-1} * {leaf_oti:.2f} = {leaf_steady}")

    return "\n".join(lines)


def _normalize_compute_bw(graph, max_total_compute_bw: int) -> list[tuple[int, str, int, int]]:
    """Rescale every compute op's `compute_bw` so their sum equals `max_total_compute_bw`.

    The autotuner uses this to enforce the global compute-bandwidth budget
    instead of trusting the model to do per-op arithmetic. The model picks
    relative shares; we apply the uniform scale that turns the total
    allocation into the budget. Each `compute_bw` is floored at 1 (which
    is also the operator-level minimum in ops.py), so the resulting sum
    may exceed `max_total_compute_bw` slightly when many tiny shares hit
    the floor — acceptable for a budget enforcement pass.

    Returns a list of (instance_id, op_label, old_bw, new_bw) for the
    rescaled nodes (graph traversal order) so callers can show the
    rescaling in the timing report.
    """
    assert max_total_compute_bw >= 1, (
        f"max_total_compute_bw must be >= 1, got {max_total_compute_bw}"
    )
    compute_nodes = [n for n in graph.nodes if hasattr(n, "compute_bw")]
    if not compute_nodes:
        return []
    total = sum(n.compute_bw for n in compute_nodes)
    assert total >= 1, "Sum of compute_bw across compute ops is zero — invalid graph"
    scale = max_total_compute_bw / total
    rescaling = []
    for n in compute_nodes:
        old = n.compute_bw
        new = max(1, int(round(old * scale)))
        n.compute_bw = new
        rescaling.append((n.instance_id, _node_label(n), old, new))
    return rescaling


def _format_rescaling(rescaling, max_total_compute_bw: int) -> str:
    """Pretty-print the compute_bw rescaling table prepended to the timing report."""
    if not rescaling:
        return ""
    lines = [
        "=" * 72,
        f"COMPUTE_BW RESCALING  (sum normalized to max_total_compute_bw={max_total_compute_bw})",
        "=" * 72,
    ]
    old_total = sum(old for _, _, old, _ in rescaling)
    new_total = sum(new for _, _, _, new in rescaling)
    lines.append(f"  before: sum(compute_bw) = {old_total}")
    lines.append(f"  after:  sum(compute_bw) = {new_total}")
    for _, label, old, new in rescaling:
        lines.append(f"    {label:<40}  compute_bw: {old:>6} -> {new:>6}")
    lines.append("")
    return "\n".join(lines)


def _evaluate_dsl_turn(
    dsl_code: str,
    kernel_name: str,
    dims: dict,
    tensors: dict,
    hw_config: dict,
    max_total_compute_bw: int,
) -> dict:
    """Run the triple-gate (DSL -> translate -> IR) chain plus the timing model.

    Each gate is independent; we short-circuit at the first failure so later
    artifacts are absent in that case (which the caller relies on to choose
    feedback for the next turn). Exceptions raised by the gate functions are
    captured into the corresponding text field so the LLM gets a full
    traceback rather than a bare status.
    """
    out = {
        "status": None,
        "dsl_correctness_text": None,
        "translated_code": None,
        "translate_error_text": None,
        "graph_correctness_text": None,
        "timing_error_text": None,
        "new_cycles": None,
        "new_report": None,
    }

    # Gate 1: DSL exec vs gold.
    try:
        dsl_text = _run_dsl_correctness(dsl_code, kernel_name, dims, tensors)
    except Exception:
        dsl_text = "ERROR:\n" + traceback.format_exc()
    out["dsl_correctness_text"] = dsl_text
    if "match=True" not in dsl_text:
        out["status"] = "DSL_FAIL"
        return out

    # Gate 2: deterministic translate.
    try:
        translated = translate(dsl_code)
    except Exception:
        out["translate_error_text"] = traceback.format_exc()
        out["status"] = "TRANSLATE_ERROR"
        return out
    out["translated_code"] = translated

    # Gate 3: IR sim vs gold.
    try:
        graph_text = _run_graph_correctness(translated, kernel_name, dims, tensors)
    except Exception:
        graph_text = "ERROR:\n" + traceback.format_exc()
    out["graph_correctness_text"] = graph_text
    if "match=True" not in graph_text:
        out["status"] = "IR_FAIL"
        return out

    # Step 4: timing model on the translated graph.
    try:
        new_cycles, new_report = _measure(
            translated, kernel_name, dims, tensors, hw_config, max_total_compute_bw,
        )
    except Exception:
        out["timing_error_text"] = traceback.format_exc()
        out["status"] = "TIMING_ERROR"
        return out

    out["new_cycles"] = new_cycles
    out["new_report"] = new_report
    out["status"] = "PASS"
    return out


def _measure(code: str, kernel_name: str, dims: dict, tensors: dict,
             hw_config: dict, max_total_compute_bw: int) -> tuple[int, str]:
    """Run the analytical timing model on `code`. Returns (total_cycles, report).

    Before timing, every compute op's `compute_bw` is rescaled so the sum
    equals `max_total_compute_bw` (see `_normalize_compute_bw`). The
    timing report begins with a summary of that rescaling so the agent
    sees the post-scaled values.

    Callers are expected to have already verified correctness of `code`.
    """
    graph, _out = _exec_build_graph(code, dims, tensors)
    rescaling = _normalize_compute_bw(graph, max_total_compute_bw)
    result = analyze_timing(graph, hw_config=hw_config)
    total = _sym_to_int(result["total_cycles"])
    report = ""#_build_verbose_report(graph, result, hw_config)
    prefix = _format_rescaling(rescaling, max_total_compute_bw)
    if prefix:
        report = prefix + report
    return total, report


def _write_progress(ckpt_root: Path, *, baseline_cycles: int, best_cycles: int,
                    turn: int, last_status: str) -> None:
    """Write running progress so external code can recover best-so-far on crash.

    Called once after baseline measurement (turn=-1, last_status='BASELINE')
    and again at the end of every turn-loop iteration. Overwrites prior writes.
    """
    _write(ckpt_root / "progress.json", json.dumps({
        "baseline_cycles": baseline_cycles,
        "best_cycles": best_cycles,
        "turn": turn,
        "last_status": last_status,
    }, indent=2))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_autotune(
    kernel_name: str,
    preset: str,
    llm_config: dict,
    autotune_config: dict,
    resume_from: str,
    max_turns: int = None,
    checkpoint_dir: str = None,
    agent_variant: str = "general",
) -> dict:
    """Autotune a verified build_graph() starting from a past checkpoint.

    Args:
        autotune_config: dict with keys `hw_config`, `constraints`, `max_turns`
            (see autotune_config.json).
        resume_from: path to the successful implementer checkpoint — a file,
            turn dir, outer dir, or checkpoint root. See _resolve_resume_build_graph.
        max_turns: overrides autotune_config["max_turns"] if provided.
        agent_variant: which autotuner agent to run — "general" (default) or
            "parallel" (Parallelize/StaticReassemble specialist).
    """
    hw_config = autotune_config["hw_config"]
    constraints = autotune_config["constraints"]
    max_total_compute_bw = constraints["max_total_compute_bw"]
    if max_turns is None:
        max_turns = autotune_config.get("max_turns", 8)

    # Resolve dims from StepDB
    config = _load_stepdb_config()
    assert kernel_name in config, f"Kernel '{kernel_name}' not found"
    assert preset in config[kernel_name]["presets"], f"Preset '{preset}' not found"
    dims = config[kernel_name]["presets"][preset]

    # Load the baseline build_graph from the resume checkpoint
    baseline_code, baseline_src = _resolve_resume_build_graph(resume_from, kernel_name)
    print(f"Loaded baseline build_graph ({len(baseline_code)} chars) from {baseline_src}")

    # Precompute tensors (same call used by the implementer pipeline)
    tensors = precompute_tensors(kernel_name, dims)
    print(f"Pre-computed tensors: {sorted(tensors.keys())}")

    # Verify baseline correctness up front — required invariant
    baseline_check = _run_graph_correctness(baseline_code, kernel_name, dims, tensors)
    assert "match=True" in baseline_check, (
        f"Baseline code from {baseline_src} failed correctness:\n{baseline_check}"
    )

    # Measure baseline
    baseline_cycles, baseline_report = _measure(
        baseline_code, kernel_name, dims, tensors, hw_config, max_total_compute_bw)
    print(f"Baseline total_cycles = {baseline_cycles}")

    # Merge hw_config + constraints for the system prompt's {hw_constraints} block
    prompt_constraints = {**hw_config, **constraints}

    # Create the autotuner agent
    agent = make_autotune_agent(llm_config, prompt_constraints, variant=agent_variant)

    # Checkpoint setup
    if checkpoint_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        checkpoint_dir = str(Path("checkpoints_autotune") / ts)
    ckpt_root = Path(checkpoint_dir) / kernel_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    _write(ckpt_root / "config.json", json.dumps({
        "kernel": kernel_name,
        "preset": preset,
        "dims": dims,
        "llm_config": {k: v for k, v in llm_config.items() if k != "api_key"},
        "autotune_config": autotune_config,
        "resume_from": str(resume_from),
        "resume_resolved_to": str(baseline_src),
        "max_turns": max_turns,
        "agent_variant": agent_variant,
    }, indent=2))
    _write(ckpt_root / "baseline.py", baseline_code)
    _write(ckpt_root / "baseline_timing.txt", baseline_report)

    best_code = baseline_code
    best_cycles = baseline_cycles
    current_code = baseline_code
    current_report = baseline_report
    _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                    best_cycles=best_cycles, turn=-1, last_status="BASELINE")

    user_prompt = build_autotune_user_prompt(
        kernel_name, dims, current_code, current_report,
        baseline_cycles=baseline_cycles, best_cycles=best_cycles)
    conversation = [{"role": "user", "content": user_prompt}]

    for turn in range(max_turns):
        turn_dir = ckpt_root / f"turn_{turn}"
        print(f"[autotune] Turn {turn + 1}/{max_turns} — current best={best_cycles}")

        _write(turn_dir / "user_prompt.txt", conversation[-1]["content"])

        run_result = await Runner.run(agent, conversation)
        assistant_text = run_result.final_output or ""
        conversation.append({"role": "assistant", "content": assistant_text})
        _write(turn_dir / "response.txt", assistant_text)
        reasoning = _reasoning_text(run_result)
        if reasoning:
            _write(turn_dir / "reasoning.txt", reasoning)

        proposal = _extract_code(assistant_text)
        if not proposal:
            print("  no code block — skipping")
            _write(turn_dir / "status.txt", "NO_CODE")
            conversation.append({"role": "user", "content":
                "Your response did not contain a ```python code block. "
                "Please emit the full updated build_graph(dims, tensors)."})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="NO_CODE")
            continue
        _write(turn_dir / "extracted_code.py", proposal)

        # Correctness first
        try:
            correctness = _run_graph_correctness(proposal, kernel_name, dims, tensors)
        except Exception:
            correctness = "ERROR:\n" + traceback.format_exc()
        _write(turn_dir / "correctness_result.txt", correctness)

        if "match=True" not in correctness:
            print(f"  correctness FAILED: {correctness.splitlines()[0]}")
            _write(turn_dir / "status.txt", "CORRECTNESS_FAIL")
            conversation.append({"role": "user", "content":
                "## Correctness: FAIL\n\n"
                "Your proposal no longer matches the reference. \n\n"
                f"```\n{correctness}\n```\n\n"
                f"### Last correct build_graph (use this as the base)\n\n"
                f"```python\n{current_code}\n```"})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="CORRECTNESS_FAIL")
            continue

        # Correctness OK — measure
        try:
            new_cycles, new_report = _measure(
                proposal, kernel_name, dims, tensors, hw_config, max_total_compute_bw)
        except Exception:
            err = traceback.format_exc()
            _write(turn_dir / "status.txt", "TIMING_ERROR")
            conversation.append({"role": "user", "content":
                f"## Timing model error\n\n```\n{err}\n```\n\n"
                "Correctness passed but analyze_timing raised. This usually "
                "means a knob is out of range."})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="TIMING_ERROR")
            continue

        _write(turn_dir / "timing.txt", new_report)
        delta = new_cycles - best_cycles
        tag = "NEW_BEST" if new_cycles < best_cycles else ("SAME" if new_cycles == best_cycles else "REGRESSION")
        _write(turn_dir / "status.txt", f"PASS {tag} cycles={new_cycles} delta={delta:+d}")
        print(f"  correctness=PASS cycles={new_cycles} ({tag}, Δ={delta:+d})")

        current_code = proposal
        current_report = new_report
        if new_cycles < best_cycles:
            best_cycles = new_cycles
            best_code = proposal
            _write(ckpt_root / "best.py", best_code)
            _write(ckpt_root / "best_timing.txt", new_report)

        # Feed the next prompt
        conversation.append({"role": "user", "content": build_autotune_user_prompt(
            kernel_name, dims, current_code, current_report,
            baseline_cycles=baseline_cycles, best_cycles=best_cycles)})
        _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                        best_cycles=best_cycles, turn=turn, last_status=tag)

    result = {
        "success": True,
        "kernel": kernel_name,
        "preset": preset,
        "baseline_cycles": baseline_cycles,
        "best_cycles": best_cycles,
        "speedup": baseline_cycles / best_cycles if best_cycles > 0 else None,
        "turns": max_turns,
        "resume_from": str(baseline_src),
        "checkpoint_dir": str(ckpt_root),
    }
    _write(ckpt_root / "result.json", json.dumps(result, indent=2))
    if best_code is not baseline_code:
        _write(ckpt_root / "best.py", best_code)
    print(f"\n[autotune] done — baseline={baseline_cycles} best={best_cycles} "
          f"speedup={result['speedup']:.2f}x" if result["speedup"] else "")
    return result
