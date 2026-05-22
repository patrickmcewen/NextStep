"""ACE-style context refresh support for autotune2.

This module deliberately manages *context*, not proposal generation. The
autotune2 LLM still emits strict YAML + Python through the existing agent
path; this manager supplies a compact playbook block and serializes
refresh updates so parallel lanes do not overwrite each other.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable


RefreshFn = Callable[..., str | Awaitable[str]]


@dataclass(frozen=True)
class AceContextConfig:
    enabled: bool = False
    refresh_interval_turns: int = 4
    initial_playbook: str = ""
    playbook_path: Path | None = None


class AceContextManager:
    """Serialized playbook/context refresh service.

    Each optimization lane owns its session counter and event batches. The
    manager owns the shared playbook and protects updates with one async
    lock, so lanes can refresh independently without concurrent writes.
    """

    def __init__(
        self,
        config: AceContextConfig,
        *,
        refresh_fn: RefreshFn | None = None,
    ) -> None:
        assert config.refresh_interval_turns >= 1, (
            f"AceContextManager: refresh_interval_turns must be >= 1, "
            f"got {config.refresh_interval_turns!r}"
        )
        self.config = config
        self._refresh_fn = refresh_fn
        self._lock = asyncio.Lock()
        self._playbook = self._load_initial_playbook(config)
        self._version = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def refresh_interval_turns(self) -> int:
        return self.config.refresh_interval_turns

    @property
    def version(self) -> int:
        return self._version

    def context_text(self) -> str:
        return self._playbook

    def write_session_start(
        self,
        attempt_dir: Path,
        *,
        session_index: int,
        metadata: dict,
    ) -> None:
        self._append_session_record(
            attempt_dir,
            {
                "event": "session_start",
                "session_index": int(session_index),
                "playbook_version": int(self._version),
                "metadata": dict(metadata),
            },
        )

    def write_turn_event(self, attempt_dir: Path, event: dict) -> None:
        self._append_jsonl(
            Path(attempt_dir) / "ace_events.jsonl",
            {
                "playbook_version": int(self._version),
                **dict(event),
            },
        )

    async def refresh(
        self,
        attempt_dir: Path,
        *,
        completed_session_index: int,
        next_session_index: int,
        events: list[dict],
        metadata: dict,
    ) -> str:
        assert events, "AceContextManager.refresh: events must be non-empty"
        async with self._lock:
            old_version = self._version
            if self._refresh_fn is None:
                updated = self._default_refresh(events)
            else:
                updated = self._refresh_fn(
                    playbook=self._playbook,
                    events=list(events),
                    metadata=dict(metadata),
                )
                if hasattr(updated, "__await__"):
                    updated = await updated
            assert isinstance(updated, str), (
                f"AceContextManager.refresh: refresh_fn must return str, "
                f"got {type(updated).__name__}"
            )
            self._playbook = updated
            self._version += 1
            self._write_playbook()
            self._append_session_record(
                attempt_dir,
                {
                    "event": "refresh",
                    "completed_session_index": int(completed_session_index),
                    "next_session_index": int(next_session_index),
                    "old_playbook_version": int(old_version),
                    "new_playbook_version": int(self._version),
                    "event_count": len(events),
                    "metadata": dict(metadata),
                },
            )
            return self._playbook

    def _load_initial_playbook(self, config: AceContextConfig) -> str:
        if config.playbook_path is None:
            return config.initial_playbook
        path = Path(config.playbook_path)
        return path.read_text(encoding="utf-8") if path.exists() else config.initial_playbook

    def _write_playbook(self) -> None:
        if self.config.playbook_path is None:
            return
        path = Path(self.config.playbook_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._playbook, encoding="utf-8")

    def _default_refresh(self, events: list[dict]) -> str:
        counts: dict[str, int] = {}
        for event in events:
            status = str(event["status"])
            counts[status] = counts.get(status, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if not self._playbook:
            return f"- Recent autotune outcomes: {summary}"
        return self._playbook + f"\n- Recent autotune outcomes: {summary}"

    def _append_session_record(self, attempt_dir: Path, record: dict) -> None:
        self._append_jsonl(Path(attempt_dir) / "ace_sessions.jsonl", record)

    def _append_jsonl(self, path: Path, record: dict) -> None:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
