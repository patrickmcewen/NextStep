"""Redirect process stdout/stderr to a log file, keeping the terminal quiet
except for crashes and explicit ``terminal_print`` lines.

Entry-point scripts (``run.py``, ``run_autotune2.py``, ``run_regression.py``)
call ``redirect_stdio_to(<ckpt>/<name>.log)`` once the run's checkpoint
directory is known. After that:

  * File descriptors 1 (stdout) and 2 (stderr) point at the log file, so
    ``print()``, the ``logging`` module's default StreamHandler, AND any
    subprocess / native-extension write that inherits fd 1/2 lands in the
    log file.
  * ``terminal_print(msg)`` writes a single line back to the original
    terminal — use it for start/end markers the user should still see.
  * Uncaught exceptions are dumped (full traceback) to the *original*
    terminal stderr via ``sys.excepthook`` before propagating, so crashes
    stay visible without digging through the log.

When ``sys.stdout`` is not a TTY (e.g. ``run_regression.py`` spawns
``run.py`` with piped stdout) ``redirect_stdio_to`` is a no-op — the
caller already wired the streams and overriding would silently hide
output from the parent's per-job log.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_INSTALLED = False
_TERMINAL_STDOUT = None  # file-like wrapping the original fd 1 (None until install)
_TERMINAL_STDERR = None  # file-like wrapping the original fd 2


def redirect_stdio_to(path: str | Path) -> bool:
    """Reopen fd 1/2 against ``path``. Returns True if redirected, False
    if skipped (non-TTY stdout). Asserts on double-install."""
    global _INSTALLED, _TERMINAL_STDOUT, _TERMINAL_STDERR
    assert not _INSTALLED, "log_redirect.redirect_stdio_to called twice"
    if not sys.stdout.isatty():
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Drain any pending writes against the original fds before remapping.
    sys.stdout.flush()
    sys.stderr.flush()

    # Snapshot the original terminal fds so terminal_print and the
    # excepthook can still reach the user's screen.
    _TERMINAL_STDOUT = os.fdopen(os.dup(1), "w", buffering=1)
    _TERMINAL_STDERR = os.fdopen(os.dup(2), "w", buffering=1)

    log_fd = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    def _excepthook(exc_type, exc_value, exc_tb):
        traceback.print_exception(
            exc_type, exc_value, exc_tb, file=_TERMINAL_STDERR
        )
        _TERMINAL_STDERR.flush()
        # Also persist in the log via fd 2 (now -> log file).
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        sys.stderr.flush()

    sys.excepthook = _excepthook
    _INSTALLED = True
    return True


def terminal_print(msg: str) -> None:
    """Write one line to the original terminal stdout when a redirect is
    installed; otherwise falls back to the current ``sys.stdout``."""
    out = _TERMINAL_STDOUT if _TERMINAL_STDOUT is not None else sys.stdout
    print(msg, file=out, flush=True)


def is_installed() -> bool:
    return _INSTALLED
