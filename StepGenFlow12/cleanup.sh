#!/bin/bash
# cleanup.sh — kill stray StepGenFlow processes left over after a
# run.py / run_autotune.py / run_autotune2.py / run_regression.py /
# run_chained.py invocation was killed or crashed.
#
# Matches:
#   * Top-level runners: python .../StepGenFlow*/(run|run_autotune|
#     run_autotune2|run_regression|run_chained).py
#   * Rust simulator subprocesses: python -c "...step_perf.run_graph(...)..."
#     (spawned by StepDB/{evaluate,sim_timing,validate_timing,
#     tap_kernel,verify_parallelize}.py).
#
# Every descendant of a matched PID is also collected.
#
# Default behaviour is dry-run. Pass --kill or -y to actually terminate.
# Kill cascade: SIGTERM all, wait 5s, then SIGKILL anything still alive.
#
# Usage:
#   ./cleanup.sh           # dry-run: print the list, do nothing
#   ./cleanup.sh --kill    # actually terminate
#   ./cleanup.sh -y        # alias for --kill

set -u

DO_KILL=0
case "${1:-}" in
    --kill|-y) DO_KILL=1 ;;
    "") ;;
    -h|--help)
        sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "usage: $0 [--kill|-y]" >&2
        exit 2
        ;;
esac

SELF_PID=$$
SELF_PPID=$(ps -o ppid= -p "$SELF_PID" | tr -d ' ')

# Matches by reading /proc/<pid>/cmdline (null-separated argv) and
# applying bash glob `case` patterns. Done this way to avoid the awk
# self-reference trap: if the regex were stored as a literal awk string
# in this script, ps would surface it as part of awk's argv and the
# pattern would match itself.
matches() {
    local pid=$1 cmd
    [ -r "/proc/$pid/cmdline" ] || return 1
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    [ -n "$cmd" ] || return 1
    case "$cmd" in
        *python*StepGenFlow*/run.py*|\
        *python*StepGenFlow*/run_autotune.py*|\
        *python*StepGenFlow*/run_autotune2.py*|\
        *python*StepGenFlow*/run_regression.py*|\
        *python*StepGenFlow*/run_chained.py*) return 0 ;;
    esac
    case "$cmd" in
        *python*step_perf.run_graph*) return 0 ;;
    esac
    return 1
}

# Step 1: enumerate PIDs and find roots that match.
roots=()
for entry in /proc/[0-9]*; do
    pid=${entry##*/}
    [ "$pid" = "$SELF_PID" ] && continue
    [ "$pid" = "$SELF_PPID" ] && continue
    if matches "$pid"; then
        roots+=("$pid")
    fi
done

if [ ${#roots[@]} -eq 0 ]; then
    echo "No stray StepGenFlow processes found."
    exit 0
fi

# Step 2: walk the descendant tree of each root.
collect_descendants() {
    local pid=$1 kids k
    kids=$(ps -o pid= --ppid "$pid" 2>/dev/null)
    for k in $kids; do
        echo "$k"
        collect_descendants "$k"
    done
}

all_pids=()
seen=" "
for r in "${roots[@]}"; do
    if [[ "$seen" != *" $r "* ]]; then
        all_pids+=("$r")
        seen+="$r "
    fi
    for d in $(collect_descendants "$r"); do
        if [[ "$seen" != *" $d "* ]] && [ "$d" != "$SELF_PID" ] && [ "$d" != "$SELF_PPID" ]; then
            all_pids+=("$d")
            seen+="$d "
        fi
    done
done

if [ ${#all_pids[@]} -eq 0 ]; then
    echo "No stray StepGenFlow processes found."
    exit 0
fi

# Step 3: report.
echo "Found the following processes:"
echo
printf '%-7s %-7s %-9s %-9s %s\n' PID PPID ETIME RSS CMD
ps -o pid=,ppid=,etime=,rss=,args= -p "${all_pids[@]}" 2>/dev/null | awk '
    {
        pid=$1; ppid=$2; etime=$3; rss=$4
        cmd=""
        for (i=5; i<=NF; i++) cmd = cmd " " $i
        if (length(cmd) > 140) cmd = substr(cmd, 1, 137) "..."
        printf "%-7s %-7s %-9s %-9s %s\n", pid, ppid, etime, rss, cmd
    }'

echo
echo "Total: ${#all_pids[@]} process(es)."

if [ "$DO_KILL" -eq 0 ]; then
    echo
    echo "Dry run — pass --kill (or -y) to actually terminate."
    exit 0
fi

# Step 4: SIGTERM, wait, SIGKILL stragglers.
echo
echo "Sending SIGTERM..."
kill "${all_pids[@]}" 2>/dev/null || true

alive=""
for _ in 1 2 3 4 5; do
    sleep 1
    alive=$(ps -o pid= -p "${all_pids[@]}" 2>/dev/null | tr -d ' ')
    if [ -z "$alive" ]; then
        echo "All processes terminated."
        exit 0
    fi
done

echo "Some processes survived SIGTERM; sending SIGKILL to: $alive"
kill -9 $alive 2>/dev/null || true
sleep 1
remaining=$(ps -o pid= -p $alive 2>/dev/null | tr -d ' ')
if [ -n "$remaining" ]; then
    echo "Warning: still alive after SIGKILL: $remaining"
    exit 1
fi
echo "All processes terminated."
