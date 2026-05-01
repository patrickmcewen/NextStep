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

from step_py.timing import analyze_timing  # noqa: E402

# Reuse the orchestrator's correctness checker so we verify identically to
# the implementer pipeline.
from src.orchestrator import _run_graph_correctness, _write, _extract_code, _reasoning_text  # noqa: E402


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


def _build_verbose_report(graph, result) -> str:
    """Verbose per-node timing report.

    Sections:
      1. Graph structure — predecessors/successors for each node.
      2. Per-node block — incoming OTI/NIT, T_fire, OTPC, N_fire, derived
         (OCI/OTI/ICI/ICD/st/fto/end), and an OCI max-breakdown labeled by
         the predecessor each candidate term comes from.
      3. Critical path — latency chain (st argmax) and throughput origin
         (OCI argmax) from the last-finishing leaf.
    """
    info = result["per_node"]
    sym_subs = result.get("sym_subs", {}) or {}

    def _sub(expr):
        if sym_subs and hasattr(expr, "free_symbols") and expr.free_symbols:
            return expr.xreplace(sym_subs)
        return expr

    total = _sym_to_int(result["total_cycles"])
    lines = [f"total_cycles={total}", ""]

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


def _measure(code: str, kernel_name: str, dims: dict, tensors: dict,
             hw_config: dict) -> tuple[int, str]:
    """Run the analytical timing model on `code`. Returns (total_cycles, report).

    Callers are expected to have already verified correctness of `code`.
    """
    graph, _out = _exec_build_graph(code, dims, tensors)
    result = analyze_timing(graph, hw_config=hw_config)
    total = _sym_to_int(result["total_cycles"])
    return total, _build_verbose_report(graph, result)


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
        baseline_code, kernel_name, dims, tensors, hw_config)
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

    # ---- Agent loop ----
    best_code = baseline_code
    best_cycles = baseline_cycles
    current_code = baseline_code
    current_report = baseline_report

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
            continue

        # Correctness OK — measure
        try:
            new_cycles, new_report = _measure(
                proposal, kernel_name, dims, tensors, hw_config)
        except Exception:
            err = traceback.format_exc()
            _write(turn_dir / "status.txt", "TIMING_ERROR")
            conversation.append({"role": "user", "content":
                f"## Timing model error\n\n```\n{err}\n```\n\n"
                "Correctness passed but analyze_timing raised. This usually "
                "means a knob is out of range."})
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
