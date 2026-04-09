"""Compare cycle-accurate simulator timing vs analytical model, per-node.

Builds the STeP graph, runs the Rust DAM simulator (with MongoDB logging),
runs the analytical timing model, and produces:
  - Side-by-side comparison CSV
  - Simulator Gantt chart (HTML, per-tile from MongoDB)
  - Analytical Gantt chart (HTML, synthetic tiles from model)
  - Raw tile CSV

Usage:
    python sim_timing.py gemm small              # one kernel + preset
    python sim_timing.py gemm --all-presets       # all presets for one kernel
    python sim_timing.py --all                    # everything
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sympy

from loader import load_config, get_dims, list_kernels, list_presets, load_problem, load_step_impl
from evaluate import IMPORT_SCAFFOLD, _strip_imports, STEP_TL_SRC, STEP_TL_PROTO, SIM_TIMEOUT_SECONDS


MONGO_URI = "mongodb://127.0.0.1:27017"


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def _ensure_paths():
    sys.path = [p for p in sys.path if "PytorchStepFlow/" not in p or "PytorchStepFlowNew" in p]
    if STEP_TL_SRC not in sys.path:
        sys.path.insert(0, STEP_TL_SRC)
    if STEP_TL_PROTO not in sys.path:
        sys.path.append(STEP_TL_PROTO)


def build_graph(kernel_name: str, preset: str):
    """Build graph and return (graph, output_op, dims)."""
    dims = get_dims(kernel_name, preset)
    step_code = load_step_impl(kernel_name)

    _ensure_paths()

    full_code = IMPORT_SCAFFOLD + step_code#_strip_imports(step_code)
    namespace = {}
    exec(full_code, namespace)
    assert namespace.get("build_graph") is not None, "step_impl.py does not define build_graph"

    graph, output_op = namespace["build_graph"](dims)
    return graph, output_op, dims


# ---------------------------------------------------------------------------
# Cycle-accurate simulator
# ---------------------------------------------------------------------------

def run_simulator_with_logging(graph, kernel_name: str, preset: str, timeout: int = SIM_TIMEOUT_SECONDS) -> tuple[int, str]:
    """Serialize graph, run Rust simulator with logging=True, return (cycles, db_name)."""
    from sim import serialize, SimConfig, HBMConfig

    work_dir = str(Path(__file__).resolve().parent / "kernels" / kernel_name / f"_work_{preset}_timed")
    os.makedirs(work_dir, exist_ok=True)

    orig_dir = os.getcwd()
    os.chdir(work_dir)
    pb_path = os.path.join(os.getcwd(), "graph.pb")

    sim_config = SimConfig(channel_depth=2, functional_sim=True, mock_bf16=False)
    hbm_config = HBMConfig(
        addr_offset=64, channel_num=32,
        per_channel_latency=2, per_channel_init_interval=2,
        per_channel_outstanding=1, per_channel_start_up_time=0,
    )

    serialize(graph, pb_path, sim_config.functional_sim)

    db_name = f"{kernel_name}_{preset}"

    # Drop stale data from any previous run
    from pymongo import MongoClient
    MongoClient(MONGO_URI).drop_database(db_name)

    sim_runner_script = (
        "import json, sys, os\n"
        "os.chdir(sys.argv[1])\n"
        "from sim import HBMConfig, SimConfig\n"
        "import step_perf\n"
        "pb_path = sys.argv[2]\n"
        "hbm_cfg = json.loads(sys.argv[3])\n"
        "sim_cfg = json.loads(sys.argv[4])\n"
        "db_name = sys.argv[5]\n"
        "hbm = HBMConfig(**hbm_cfg)\n"
        "sim = SimConfig(**sim_cfg)\n"
        "ret = step_perf.run_graph(pb_path, True, hbm, sim, db_name)\n"
        "print(json.dumps({'passed': ret[0], 'cycles': ret[1]}))\n"
    )

    pythonpath = STEP_TL_SRC + ":" + STEP_TL_PROTO + ":" + os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath

    proc = subprocess.run(
        [sys.executable, "-c", sim_runner_script,
         os.getcwd(), pb_path,
         json.dumps(asdict(hbm_config)),
         json.dumps({"channel_depth": sim_config.channel_depth,
                      "functional_sim": sim_config.functional_sim,
                      "mock_bf16": sim_config.mock_bf16}),
         db_name],
        capture_output=True, text=True, timeout=timeout,
        env=env,
    )

    os.chdir(orig_dir)

    assert proc.returncode == 0, (
        f"Simulator failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
    )

    result = json.loads(proc.stdout.strip().split("\n")[-1])
    return result["cycles"], db_name


def extract_sim_timing(db_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query MongoDB for per-node and per-tile timing from the simulator."""
    from pymongo import MongoClient

    client = MongoClient(MONGO_URI)
    db = client[db_name]
    collection = db["log"]

    tile_cursor = collection.find(
        {"event_type": "SimpleEvent"},
        {"_id": 0, "context": 1, "event_data": 1},
    )

    tile_rows = []
    for doc in tile_cursor:
        ed = doc["event_data"]
        tile_rows.append({
            "context_id": int(doc["context"]),
            "name": ed["name"],
            "id": int(ed["id"]),
            "start_ns": int(ed["start_ns"]),
            "end_ns": int(ed["end_ns"]),
            "is_stop": ed.get("is_stop", False),
        })

    tile_df = pd.DataFrame(tile_rows)
    assert len(tile_df) > 0, f"No SimpleEvent entries found in db '{db_name}'"

    tile_df["duration"] = tile_df["end_ns"] - tile_df["start_ns"]
    tile_df.sort_values(["id", "start_ns"], inplace=True, ignore_index=True)

    node_df = (
        tile_df.groupby(["id", "name"])
        .agg(
            sim_start=("start_ns", "min"),
            sim_end=("end_ns", "max"),
            sim_tiles=("start_ns", "count"),
            sim_avg_tile_dur=("duration", "mean"),
        )
        .reset_index()
        .rename(columns={"id": "op_id", "name": "op_name"})
    )
    node_df.sort_values("op_id", inplace=True, ignore_index=True)

    client.close()
    return node_df, tile_df


# ---------------------------------------------------------------------------
# Analytical timing model
# ---------------------------------------------------------------------------

def run_analytical_model(graph) -> tuple[int, dict]:
    """Run analytical timing model, return (total_cycles, per_node_info).

    Any remaining symbolic dimensions (from FlatPartition dynamic routing)
    are substituted with 1 (expected uniform value).
    """
    from step_py.timing import analyze_timing

    result = analyze_timing(graph)
    total = result["total_cycles"]

    # Collect ALL free symbols across total and every per-node field
    all_symbols = set(total.free_symbols) if hasattr(total, 'free_symbols') else set()
    sub_keys = ("end", "fto", "st", "OCI", "OTI", "ICI", "ICD", "T_fire", "N_fire")
    for nid in result["per_node"]:
        for key in sub_keys:
            val = result["per_node"][nid].get(key)
            if val is not None and hasattr(val, 'free_symbols'):
                all_symbols |= val.free_symbols

    if all_symbols:
        # Use the expected-value substitutions computed by analyze_timing
        # (e.g. FlatPartition DynDims → input_N_fire / num_consumers).
        # Fall back to 1 only for symbols not covered by the timing model.
        timing_subs = result.get("sym_subs") or {}
        sym_subs = {s: timing_subs.get(s, 1) for s in all_symbols}
        total = total.subs(sym_subs)
        for nid in result["per_node"]:
            for key in sub_keys:
                val = result["per_node"][nid].get(key)
                if val is not None and hasattr(val, 'free_symbols') and val.free_symbols:
                    result["per_node"][nid][key] = val.subs(sym_subs)

    return int(sympy.N(total)), result["per_node"]


def build_analytical_tiles(graph, ana_per_node: dict) -> pd.DataFrame:
    """Synthesize per-tile events from the analytical model's per-node timing.

    The analytical model says each node produces N_fire tiles, with:
      - first tile completing at fto
      - each subsequent tile completing OCI cycles later
      - each tile taking T_fire cycles of compute (or OTI for off-chip ops)

    We generate one row per tile to match the simulator's tile event format.
    """
    from step_py.timing import topological_sort

    rows = []
    for n in topological_sort(graph):
        nid = n.instance_id
        info = ana_per_node.get(nid)
        if info is None:
            continue

        n_fire = int(sympy.N(info["N_fire"]))
        fto = int(sympy.N(info["fto"]))
        oci = int(sympy.N(info["OCI"]))
        t_fire = int(sympy.N(info["T_fire"]))
        oti = int(sympy.N(info["OTI"]))
        op_name = type(n).__name__

        # Skip pass-through nodes with no duration
        if n_fire == 0:
            continue

        # Tile duration for visualization:
        #   Compute ops: T_fire (actual compute per tile)
        #   Off-chip ops: OTI (HBM access time per tile)
        # But clamp so tile_start never precedes the node's start time.
        tile_dur = oti if n.is_offchip_memory_op() else max(t_fire, 1)
        st = int(sympy.N(info["st"]))
        icd = int(sympy.N(info["ICD"]))

        for i in range(n_fire):
            tile_end = fto + i * oci
            tile_start = max(tile_end - tile_dur, st + icd)
            rows.append({
                "name": op_name,
                "id": nid,
                "start_ns": tile_start,
                "end_ns": tile_end,
                "is_stop": False,
            })

    df = pd.DataFrame(rows)
    df.sort_values(["id", "start_ns"], inplace=True, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison(graph, sim_node_df: pd.DataFrame, ana_per_node: dict) -> pd.DataFrame:
    """Merge simulator and analytical per-node timing into one DataFrame."""
    from step_py.timing import topological_sort

    analytical_rows = []
    for n in topological_sort(graph):
        nid = n.instance_id
        info = ana_per_node.get(nid)
        if info is None:
            continue
        analytical_rows.append({
            "op_id": nid,
            "op_name": type(n).__name__,
            "op_str": str(n),
            "ana_start": int(sympy.N(info["st"])),
            "ana_fto": int(sympy.N(info["fto"])),
            "ana_end": int(sympy.N(info["end"])),
            "ana_N_fire": int(sympy.N(info["N_fire"])),
            "ana_T_fire": int(sympy.N(info["T_fire"])),
            "ana_OCI": int(sympy.N(info["OCI"])),
            "ana_OTI": int(sympy.N(info["OTI"])),
            "ana_ICI": int(sympy.N(info["ICI"])),
            "ana_ICD": int(sympy.N(info["ICD"])),
        })

    ana_df = pd.DataFrame(analytical_rows)

    sim_for_merge = sim_node_df.drop(columns=["op_name"])
    merged = sim_for_merge.merge(ana_df, on="op_id", how="outer")

    merged["sim_duration"] = merged["sim_end"] - merged["sim_start"]
    merged["ana_duration"] = merged["ana_end"] - merged["ana_start"]

    for field in ("start", "end"):
        sim_col = f"sim_{field}"
        ana_col = f"ana_{field}"
        if sim_col in merged.columns and ana_col in merged.columns:
            merged[f"delta_{field}"] = merged[ana_col] - merged[sim_col]

    merged["delta_duration"] = merged["ana_duration"] - merged["sim_duration"]
    merged.sort_values("op_id", inplace=True, ignore_index=True)

    cols = [
        "op_id", "op_name", "op_str",
        "sim_start", "ana_start", "delta_start",
        "sim_end", "ana_end", "delta_end",
        "sim_duration", "ana_duration", "delta_duration",
        "sim_tiles", "ana_N_fire", "sim_avg_tile_dur", "ana_T_fire",
        "ana_OCI", "ana_OTI", "ana_ICI", "ana_ICD", "ana_fto",
    ]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols]


# ---------------------------------------------------------------------------
# Gantt chart PNG generation
# ---------------------------------------------------------------------------

# One color per op type so both sim and analytical charts use the same palette.
_OP_COLORS = {}
_COLOR_CYCLE = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#CDDC39", "#FF5722",
]


def _color_for_op(op_name: str) -> str:
    if op_name not in _OP_COLORS:
        _OP_COLORS[op_name] = _COLOR_CYCLE[len(_OP_COLORS) % len(_COLOR_CYCLE)]
    return _OP_COLORS[op_name]


def generate_gantt_png(tile_df: pd.DataFrame, title: str, out_path: str):
    """Render a Gantt chart PNG from a tile DataFrame.

    Expects columns: name, id, start_ns, end_ns.
    Each unique (name, id) gets its own row; tiles are drawn as horizontal bars.
    """
    # Build row labels sorted by id
    groups = tile_df.groupby(["id", "name"], sort=True)
    row_keys = sorted(groups.groups.keys(), key=lambda k: k[0])
    row_labels = [f"{name}_{op_id}" for op_id, name in row_keys]
    n_rows = len(row_labels)

    bar_height = 0.6
    fig_height = max(2.5, 0.5 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for row_idx, (op_id, name) in enumerate(row_keys):
        grp = groups.get_group((op_id, name))
        color = _color_for_op(name)
        for _, tile in grp.iterrows():
            start = tile["start_ns"]
            dur = tile["end_ns"] - start
            ax.barh(row_idx, dur, left=start, height=bar_height,
                    color=color, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cycles")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Legend: one entry per unique op name
    seen = []
    handles = []
    for _, name in row_keys:
        if name not in seen:
            seen.append(name)
            handles.append(mpatches.Patch(color=_color_for_op(name), label=name))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_comparison_gantt_png(sim_tile_df: pd.DataFrame, ana_tile_df: pd.DataFrame,
                                  title: str, out_path: str):
    """Render sim and analytical Gantt charts stacked in one PNG for easy comparison.

    Top panel = simulator, bottom panel = analytical. Same x-axis scale.
    """
    def _build_rows(tile_df):
        groups = tile_df.groupby(["id", "name"], sort=True)
        row_keys = sorted(groups.groups.keys(), key=lambda k: k[0])
        return groups, row_keys

    sim_groups, sim_keys = _build_rows(sim_tile_df)
    ana_groups, ana_keys = _build_rows(ana_tile_df)

    n_sim = len(sim_keys)
    n_ana = len(ana_keys)
    bar_height = 0.6

    fig_height = max(4, 0.45 * (n_sim + n_ana) + 2.5)
    fig, (ax_sim, ax_ana) = plt.subplots(2, 1, figsize=(14, fig_height),
                                          gridspec_kw={"height_ratios": [n_sim, n_ana]},
                                          sharex=True)

    # Shared x-axis range
    x_max = max(sim_tile_df["end_ns"].max(), ana_tile_df["end_ns"].max()) * 1.02

    for ax, groups, row_keys, subtitle in [
        (ax_sim, sim_groups, sim_keys, "Simulator"),
        (ax_ana, ana_groups, ana_keys, "Analytical"),
    ]:
        labels = [f"{name}_{op_id}" for op_id, name in row_keys]
        for row_idx, (op_id, name) in enumerate(row_keys):
            grp = groups.get_group((op_id, name))
            color = _color_for_op(name)
            for _, tile in grp.iterrows():
                start = tile["start_ns"]
                dur = tile["end_ns"] - start
                ax.barh(row_idx, dur, left=start, height=bar_height,
                        color=color, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, x_max)
        ax.set_title(subtitle, fontsize=10, fontweight="bold", loc="left")
        ax.grid(axis="x", alpha=0.3)

    ax_ana.set_xlabel("Cycles")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Shared legend
    seen = []
    handles = []
    all_keys = list(dict.fromkeys(sim_keys + ana_keys))
    for _, name in all_keys:
        if name not in seen:
            seen.append(name)
            handles.append(mpatches.Patch(color=_color_for_op(name), label=name))
    fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gantt chart HTML generation (adapted from step-perf/scripts/gantt_chart_generator.py)
# ---------------------------------------------------------------------------

def generate_gantt_html(tile_df: pd.DataFrame, title: str) -> str:
    """Generate an interactive Gantt chart HTML from a tile DataFrame.

    Expects columns: name, id, start_ns, end_ns, is_stop.
    """
    data_by_name_id = defaultdict(list)
    global_min = float("inf")
    global_max = 0
    unique_keys = set()

    for _, row in tile_df.iterrows():
        key = f"{row['name']}_{row['id']}"
        unique_keys.add(key)
        count = len(data_by_name_id[key]) + 1
        item = {
            "file_id": key,
            "name": str(row["name"]),
            "id": str(int(row["id"])),
            "prefix": "event",
            "identifier": f"event_{row['name']}_{row['id']}_{count}",
            "start": float(row["start_ns"]),
            "end": float(row["end_ns"]),
            "is_stop": bool(row.get("is_stop", False)),
        }
        data_by_name_id[key].append(item)
        global_min = min(global_min, item["start"])
        global_max = max(global_max, item["end"])

    sorted_keys = sorted(
        unique_keys,
        key=lambda x: int(x.rsplit("_", 1)[1]) if x.rsplit("_", 1)[1].isdigit() else x,
    )

    all_data = []
    for key in sorted_keys:
        all_data.extend(sorted(data_by_name_id[key], key=lambda x: x["start"]))

    return _GANTT_HTML_TEMPLATE.format(
        title=title,
        data_json=json.dumps(all_data),
        name_id_json=json.dumps(sorted_keys),
        min_time=global_min,
        max_time=global_max,
    )


_GANTT_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; overflow: hidden; }}
        h1 {{ margin-top: 0; }}
        #container {{ width: 100%; height: calc(100vh - 120px); overflow: auto; position: relative; }}
        #timeline {{ position: relative; margin-top: 40px; }}
        .file-row {{ height: 50px; margin-bottom: 5px; position: relative; border-bottom: 1px solid #eee; }}
        .file-label {{ position: absolute; left: 0; top: 10px; width: 300px; font-weight: bold;
                       text-align: right; padding-right: 10px; overflow: visible; word-wrap: break-word; line-height: 1.2; }}
        .timeline-container {{ margin-left: 320px; position: relative; height: 100%; }}
        .block {{ position: absolute; height: 30px; top: 10px; border-radius: 3px; text-align: center;
                  font-size: 10px; overflow: hidden; white-space: nowrap; color: white; display: flex;
                  align-items: center; justify-content: center; cursor: pointer; transition: opacity 0.2s; }}
        .block:hover {{ opacity: 0.8; }}
        .event-normal {{ background-color: #33A8FF; }}
        .event-stop {{ background-color: #FF5733; }}
        .timeline-marker {{ position: absolute; width: 1px; height: 100%; background-color: rgba(0,0,0,0.1); top: 0; }}
        .timeline-label {{ position: absolute; font-size: 10px; color: #666; top: -20px; transform: translateX(-50%); }}
        .controls {{ margin-bottom: 20px; }}
        .tooltip {{ position: absolute; background-color: rgba(0,0,0,0.8); color: white; padding: 8px;
                    border-radius: 4px; font-size: 12px; z-index: 100; pointer-events: none; display: none; }}
        button {{ margin-right: 10px; padding: 5px 10px; cursor: pointer; }}
        #scale-slider {{ width: 200px; display: inline-block; vertical-align: middle; }}
        .legend {{ margin-top: 10px; display: flex; align-items: center; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; margin-right: 20px; margin-bottom: 5px; }}
        .legend-color {{ width: 15px; height: 15px; border-radius: 3px; margin-right: 5px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="controls">
        <button id="zoom-in">Zoom In</button>
        <button id="zoom-out">Zoom Out</button>
        <button id="reset">Reset</button>
        <label for="scale-slider">Scale: </label>
        <input type="range" id="scale-slider" min="1" max="100" value="10">
        <span id="scale-value">1x</span>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color event-normal"></div><span>Normal Event</span></div>
        <div class="legend-item"><div class="legend-color event-stop"></div><span>Stop Event</span></div>
    </div>
    <div id="container"><div id="timeline"></div></div>
    <div id="tooltip" class="tooltip"></div>
    <script>
        const data = {data_json};
        const nameIdList = {name_id_json};
        const minTime = {min_time};
        const maxTime = {max_time};
        let scale = 0.5;
        let timelineEl = document.getElementById('timeline');
        let containerEl = document.getElementById('container');
        let tooltipEl = document.getElementById('tooltip');
        let scaleSlider = document.getElementById('scale-slider');
        let scaleValue = document.getElementById('scale-value');

        function renderTimeline() {{
            timelineEl.innerHTML = '';
            const timelineWidth = (maxTime - minTime) * scale;
            nameIdList.forEach(nameId => {{
                const fileRow = document.createElement('div');
                fileRow.className = 'file-row';
                const fileLabel = document.createElement('div');
                fileLabel.className = 'file-label';
                fileLabel.textContent = nameId;
                fileLabel.title = nameId;
                const tc = document.createElement('div');
                tc.className = 'timeline-container';
                tc.style.width = `${{timelineWidth}}px`;
                fileRow.appendChild(fileLabel);
                fileRow.appendChild(tc);
                timelineEl.appendChild(fileRow);
                const items = data.filter(item => item.file_id === nameId);
                items.forEach(item => {{
                    const block = document.createElement('div');
                    block.className = item.is_stop ? 'block event-stop' : 'block event-normal';
                    const left = (item.start - minTime) * scale;
                    const width = (item.end - item.start) * scale;
                    block.style.left = `${{left}}px`;
                    block.style.width = `${{Math.max(width, 1)}}px`;
                    if (width > 40) block.textContent = item.identifier;
                    block.dataset.identifier = item.identifier;
                    block.dataset.name = item.name;
                    block.dataset.id = item.id;
                    block.dataset.start = item.start;
                    block.dataset.end = item.end;
                    block.dataset.isStop = item.is_stop;
                    block.addEventListener('mouseover', showTooltip);
                    block.addEventListener('mousemove', moveTooltip);
                    block.addEventListener('mouseout', hideTooltip);
                    tc.appendChild(block);
                }});
            }});
            const stepSize = calculateStepSize(maxTime - minTime);
            for (let t = minTime; t <= maxTime; t += stepSize) {{
                const marker = document.createElement('div');
                marker.className = 'timeline-marker';
                marker.style.left = `${{(t - minTime) * scale + 320}}px`;
                const label = document.createElement('div');
                label.className = 'timeline-label';
                label.textContent = t.toFixed(0) + ' cyc';
                marker.appendChild(label);
                timelineEl.appendChild(marker);
            }}
        }}

        function calculateStepSize(range) {{
            const roughStep = range / 10;
            const mag = Math.pow(10, Math.floor(Math.log10(roughStep)));
            const norm = roughStep / mag;
            if (norm < 1.5) return mag;
            if (norm < 3.5) return 2 * mag;
            if (norm < 7.5) return 5 * mag;
            return 10 * mag;
        }}

        function showTooltip(e) {{
            const b = e.target;
            tooltipEl.innerHTML = `
                Name: ${{b.dataset.name}}<br>
                ID: ${{b.dataset.id}}<br>
                Start: ${{parseFloat(b.dataset.start).toFixed(0)}} cyc<br>
                End: ${{parseFloat(b.dataset.end).toFixed(0)}} cyc<br>
                Duration: ${{(parseFloat(b.dataset.end) - parseFloat(b.dataset.start)).toFixed(0)}} cyc<br>
                Type: ${{b.dataset.isStop === 'true' ? 'Stop' : 'Normal'}}
            `;
            tooltipEl.style.display = 'block';
            moveTooltip(e);
        }}
        function moveTooltip(e) {{ tooltipEl.style.left = `${{e.pageX + 10}}px`; tooltipEl.style.top = `${{e.pageY + 10}}px`; }}
        function hideTooltip() {{ tooltipEl.style.display = 'none'; }}

        document.getElementById('zoom-in').addEventListener('click', () => {{ scale *= 1.5; updateScale(); renderTimeline(); }});
        document.getElementById('zoom-out').addEventListener('click', () => {{ scale /= 1.5; updateScale(); renderTimeline(); }});
        document.getElementById('reset').addEventListener('click', () => {{ scale = 0.5; updateScale(); renderTimeline(); containerEl.scrollLeft = 0; }});
        scaleSlider.addEventListener('input', () => {{ scale = scaleSlider.value / 10; updateScale(); renderTimeline(); }});
        function updateScale() {{ scaleValue.textContent = `${{scale.toFixed(2)}}x`; scaleSlider.value = Math.round(scale * 10); }}

        updateScale();
        renderTimeline();
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare simulator vs analytical timing per-node")
    parser.add_argument("kernel", nargs="?", help="Kernel name")
    parser.add_argument("preset", nargs="?", help="Preset name")
    parser.add_argument("--all", action="store_true", help="All kernels, all presets")
    parser.add_argument("--all-presets", action="store_true", help="All presets for one kernel")
    parser.add_argument("--out-dir", default="timing_results", help="Output directory")
    parser.add_argument("--timeout", type=int, default=3600, help="Simulator timeout in seconds (default: 90)")
    args = parser.parse_args()

    pairs = []
    if args.all:
        for name in list_kernels():
            for preset in list_presets(name):
                pairs.append((name, preset))
    elif args.all_presets:
        assert args.kernel, "Specify a kernel name with --all-presets"
        for preset in list_presets(args.kernel):
            pairs.append((args.kernel, preset))
    else:
        assert args.kernel and args.preset, "Specify <kernel> <preset>, or use --all / --all-presets"
        pairs.append((args.kernel, args.preset))

    os.makedirs(args.out_dir, exist_ok=True)

    for kernel_name, preset in pairs:
        print(f"\n{'='*60}")
        print(f"  {kernel_name} / {preset}")
        print(f"{'='*60}")

        graph, output_op, dims = build_graph(kernel_name, preset)

        # Cycle-accurate simulator
        sim_cycles, db_name = run_simulator_with_logging(graph, kernel_name, preset, timeout=args.timeout)
        sim_node_df, sim_tile_df = extract_sim_timing(db_name)

        # Analytical model
        ana_cycles, ana_per_node = run_analytical_model(graph)
        ana_tile_df = build_analytical_tiles(graph, ana_per_node)

        # Output directory per kernel/preset
        out_sub = os.path.join(args.out_dir, f"{kernel_name}_{preset}")
        os.makedirs(out_sub, exist_ok=True)

        # Comparison CSV
        comparison = build_comparison(graph, sim_node_df, ana_per_node)
        comp_csv = os.path.join(out_sub, "comparison.csv")
        comparison.to_csv(comp_csv, index=False)

        # Tile CSVs
        sim_tile_df.to_csv(os.path.join(out_sub, "sim_tiles.csv"), index=False)
        ana_tile_df.to_csv(os.path.join(out_sub, "ana_tiles.csv"), index=False)

        # Gantt charts (HTML)
        with open(os.path.join(out_sub, "sim_gantt.html"), "w") as f:
            f.write(generate_gantt_html(sim_tile_df, f"Simulator: {kernel_name}/{preset} ({sim_cycles} cycles)"))
        with open(os.path.join(out_sub, "ana_gantt.html"), "w") as f:
            f.write(generate_gantt_html(ana_tile_df, f"Analytical: {kernel_name}/{preset} ({ana_cycles} cycles)"))

        # Gantt charts (PNG)
        _OP_COLORS.clear()
        error_pct = abs(ana_cycles - sim_cycles) / max(sim_cycles, 1) * 100
        generate_gantt_png(sim_tile_df, f"Simulator: {kernel_name}/{preset} ({sim_cycles} cyc)",
                           os.path.join(out_sub, "sim_gantt.png"))
        generate_gantt_png(ana_tile_df, f"Analytical: {kernel_name}/{preset} ({ana_cycles} cyc)",
                           os.path.join(out_sub, "ana_gantt.png"))
        combo_png = os.path.join(out_sub, "gantt.png")
        generate_comparison_gantt_png(
            sim_tile_df, ana_tile_df,
            f"{kernel_name}/{preset} — Sim: {sim_cycles} cyc | Ana: {ana_cycles} cyc | Err: {error_pct:.1f}%",
            combo_png,
        )

        # Print
        print(f"  Sim: {sim_cycles} cycles | Analytical: {ana_cycles} cycles | Error: {error_pct:.1f}%")
        print()
        print(comparison.to_string(index=False))
        print(f"\n  Output: {out_sub}/")


if __name__ == "__main__":
    main()
