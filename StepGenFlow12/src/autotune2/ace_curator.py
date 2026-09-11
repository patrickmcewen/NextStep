"""ACE context curator adapter for autotune2.

The search loop owns optimization turns and session boundaries. This module
turns a completed session window into one curator-agent call that returns the
next shared playbook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from src.autotune2.agent_telemetry import write_agent_call_artifacts
from src.autotune2.prompts import (
    build_ace_context_curator_user_prompt,
    parse_ace_context_curator_response,
)


AceCuratorAgentFn = Callable[[list[dict]], Awaitable[object]]
MAX_PLAYBOOK_CHARS = 24_000


def _turn_dir(attempt_dir: Path, event: dict) -> Path:
    return (
        Path(attempt_dir)
        / f"session_{int(event['session_index'])}"
        / f"turn_{int(event['session_turn'])}"
    )


def summarize_ace_turn_artifacts(attempt_dir: Path, events: list[dict]) -> list[dict]:
    """Read compact summaries for the completed turns in ``events``."""
    summaries: list[dict] = []
    for event in events:
        turn_dir = _turn_dir(attempt_dir, event)
        status_path = turn_dir / "status.txt"
        assert status_path.exists(), (
            f"summarize_ace_turn_artifacts: missing status artifact for "
            f"event {event!r}: {status_path}"
        )
        summary = {
            "session_index": int(event["session_index"]),
            "session_turn": int(event["session_turn"]),
            "global_turn": int(event["global_turn"]),
            "status": event["status"],
            "status_text": status_path.read_text(),
        }
        verify_path = turn_dir / "verify_result.txt"
        if verify_path.exists():
            summary["verify_result"] = verify_path.read_text()
        score_path = turn_dir / "score.json"
        if score_path.exists():
            summary["score"] = json.loads(score_path.read_text())
        summaries.append(summary)
    return summaries


def build_ace_context_refresh_fn(agent_fn: AceCuratorAgentFn):
    """Create an ``AceContextManager`` refresh function backed by an agent."""

    async def refresh(
        *,
        playbook: str,
        events: list[dict],
        metadata: dict,
        attempt_dir: Path,
        completed_session_index: int,
        next_session_index: int,
    ) -> str:
        prior_playbook = _cap_playbook(playbook)
        turn_summaries = summarize_ace_turn_artifacts(attempt_dir, events)
        user_prompt = build_ace_context_curator_user_prompt(
            current_playbook=prior_playbook,
            events=events,
            metadata=metadata,
            turn_summaries=turn_summaries,
        )
        response = await agent_fn([{"role": "user", "content": user_prompt}])
        call_dir = (
            Path(attempt_dir)
            / "ace_curator"
            / f"session_{int(completed_session_index)}_to_{int(next_session_index)}"
        )
        write_agent_call_artifacts(
            call_dir,
            user_prompt=user_prompt,
            agent_response=response,
        )
        response_text = _response_text(response)
        try:
            updated_playbook = parse_ace_context_curator_response(response_text)
        except (AssertionError, json.JSONDecodeError) as e:
            first_error = e
            repair_prompt = (
                "Your previous response failed schema validation.\n\n"
                f"Error:\n{type(e).__name__}: {e}\n\n"
                "Re-emit only a JSON object matching this schema. Do not "
                "include markdown, commentary, or code fences.\n\n"
                '{"playbook": "the complete updated playbook", '
                '"notes": "brief summary of changes"}\n'
            )
            repair_conversation = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": repair_prompt},
            ]
            response = await agent_fn(repair_conversation)
            repair_dir = (
                Path(attempt_dir)
                / "ace_curator"
                / (
                    f"session_{int(completed_session_index)}_to_"
                    f"{int(next_session_index)}_repair_1"
                )
            )
            write_agent_call_artifacts(
                repair_dir,
                user_prompt=repair_prompt,
                agent_response=response,
            )
            try:
                updated_playbook = parse_ace_context_curator_response(
                    _response_text(response)
                )
            except (AssertionError, json.JSONDecodeError) as repair_error:
                updated_playbook = prior_playbook
                _write_fallback_artifact(
                    call_dir,
                    first_error=first_error,
                    repair_error=repair_error,
                    fallback_playbook=updated_playbook,
                )
        updated_playbook = _cap_playbook(updated_playbook)
        (call_dir / "updated_playbook.txt").write_text(updated_playbook)
        return updated_playbook

    return refresh


def _cap_playbook(playbook: str) -> str:
    playbook = playbook.strip()
    if len(playbook) <= MAX_PLAYBOOK_CHARS:
        return playbook
    marker = (
        "[ACE playbook truncated to fit curator/proposal context; "
        "retaining most recent guidance.]\n"
    )
    keep = max(0, MAX_PLAYBOOK_CHARS - len(marker))
    return marker + playbook[-keep:]


def _write_fallback_artifact(
    call_dir: Path,
    *,
    first_error: Exception,
    repair_error: Exception,
    fallback_playbook: str,
) -> None:
    payload = {
        "event": "ace_curator_fallback",
        "fallback": "previous_playbook",
        "first_error_type": type(first_error).__name__,
        "first_error": str(first_error),
        "repair_error_type": type(repair_error).__name__,
        "repair_error": str(repair_error),
        "fallback_playbook_chars": len(fallback_playbook),
    }
    (call_dir / "fallback.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    (call_dir / "fallback.txt").write_text(
        "ACE curator fallback: failed initial parse and repair parse; "
        "falling back to the previous playbook.\n"
        f"Initial error: {type(first_error).__name__}: {first_error}\n"
        f"Repair error: {type(repair_error).__name__}: {repair_error}\n"
    )


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    assert isinstance(text, str), (
        f"ACE curator response must be a string or carry a string .text "
        f"field, got {type(response).__name__}"
    )
    return text
