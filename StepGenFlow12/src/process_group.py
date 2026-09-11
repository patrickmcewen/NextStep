"""Wire the top-level runner into a self-cleaning process group.

``setup_process_group()`` does three things:

1. ``os.setpgrp()`` — become a process-group leader, so a ``killpg`` on
   our pgid reaches every subprocess we spawn (they inherit pgid by
   default) without also signalling the parent shell. Skipped when this
   process is nested inside another runner that already set the group
   up (signalled via ``STEPGENFLOW_PGROUP_OWNER`` env var), so chained
   invocations like ``run_chained.py -> run_regression.py -> run.py``
   stay in one shared group.

2. Installs SIGTERM / SIGINT / SIGHUP handlers that fan the signal out
   to the whole pgid via ``os.killpg`` before letting the default
   action kill us. Handles ``kill <runner_pid>``, terminal close, and
   Ctrl-C from synchronous code.

3. Registers an ``atexit`` fallback that walks ``/proc`` and SIGTERMs
   our descendants. Needed because ``asyncio.run()`` replaces our
   SIGINT handler with its own graceful-cancel handler on Python 3.11+,
   so Ctrl-C inside an async runner unwinds via KeyboardInterrupt
   without our group-kill firing — atexit catches that path.

SIGKILL can't be caught, so a ``kill -9`` on the runner still leaks
descendants — fall back to ``cleanup.sh`` for that case.
"""
from __future__ import annotations

import atexit
import os
import signal


_ENV_KEY = "STEPGENFLOW_PGROUP_OWNER"
_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def _propagate_then_die(signum, _frame):
    # Reset to default before fanning out so the copy of the signal that
    # comes back to us (killpg includes the caller) runs the default
    # action instead of re-entering this handler.
    signal.signal(signum, signal.SIG_DFL)
    os.killpg(os.getpgrp(), signum)


def _read_ppid(pid_str: str) -> int | None:
    # /proc/<pid>/stat format: "<pid> (<comm>) <state> <ppid> ..."; comm
    # can contain spaces and parens so we split after the LAST ')'.
    try:
        with open(f"/proc/{pid_str}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    rparen = data.rfind(b")")
    if rparen < 0:
        return None
    fields = data[rparen + 1:].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _kill_descendants_atexit() -> None:
    """SIGTERM every descendant of this process. Best-effort: races
    against /proc churn (processes exiting mid-walk) are silently
    tolerated since the goal is mop-up, not exactness.
    """
    my_pid = os.getpid()
    pid_to_ppid: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        ppid = _read_ppid(entry)
        if ppid is None:
            continue
        pid_to_ppid[int(entry)] = ppid

    descendants: set[int] = set()
    frontier = {my_pid}
    while frontier:
        nxt = {
            pid for pid, ppid in pid_to_ppid.items()
            if ppid in frontier and pid not in descendants and pid != my_pid
        }
        descendants |= nxt
        frontier = nxt

    for pid in descendants:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def setup_process_group() -> None:
    """Idempotent setup: call once from ``main()``.

    The signal handlers and atexit hook are installed unconditionally so
    a nested runner can still fan signals out within the shared group
    set up by its parent.
    """
    if _ENV_KEY not in os.environ:
        os.setpgrp()
        os.environ[_ENV_KEY] = str(os.getpid())
    for sig in _SIGNALS:
        signal.signal(sig, _propagate_then_die)
    atexit.register(_kill_descendants_atexit)
