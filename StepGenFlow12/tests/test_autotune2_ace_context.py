import asyncio
import json


def run(coro):
    return asyncio.run(coro)


def test_ace_context_refresh_fn_reads_turn_artifacts_and_logs_call(tmp_path):
    from src.autotune2.ace_curator import build_ace_context_refresh_fn

    attempt_dir = tmp_path / "attempt"
    turn_dir = attempt_dir / "session_0" / "turn_0"
    turn_dir.mkdir(parents=True)
    (turn_dir / "status.txt").write_text("ACCEPTED")
    (turn_dir / "verify_result.txt").write_text("PASS")
    (turn_dir / "score.json").write_text(json.dumps({
        "entries": [{
            "cycles": 10,
            "on_chip": 20,
            "provenance": "llm_baseline_0_attempt_0_binf_turn_0",
        }]
    }))

    captured_prompt = ""

    async def agent(conversation):
        nonlocal captured_prompt
        captured_prompt = conversation[0]["content"]
        return (
            "```json\n"
            "{\"playbook\": \"## STRATEGIES\\n"
            "[gen-00001] helpful=0 harmful=0 :: keep accepted tile reuse\", "
            "\"notes\": \"added accepted lesson\"}\n"
            "```"
        )

    refresh_fn = build_ace_context_refresh_fn(agent)
    playbook = run(refresh_fn(
        playbook="",
        events=[{
            "node_path": "root/leaf",
            "status": "ACCEPTED",
            "session_index": 0,
            "session_turn": 0,
            "global_turn": 0,
            "cycles": 10,
            "on_chip": 20,
        }],
        metadata={"node_path": "root/leaf", "fewshot": "tile_shrink"},
        attempt_dir=attempt_dir,
        completed_session_index=0,
        next_session_index=1,
    ))

    assert "keep accepted tile reuse" in playbook
    assert '"verify_result": "PASS"' in captured_prompt
    assert '"score"' in captured_prompt
    call_dir = attempt_dir / "ace_curator" / "session_0_to_1"
    assert (call_dir / "user_prompt.txt").exists()
    assert (call_dir / "response.txt").exists()
    assert (call_dir / "updated_playbook.txt").read_text() == playbook


def test_ace_context_refresh_fn_repairs_curator_schema_error_once(tmp_path):
    from src.autotune2.ace_curator import build_ace_context_refresh_fn

    attempt_dir = tmp_path / "attempt"
    turn_dir = attempt_dir / "session_0" / "turn_0"
    turn_dir.mkdir(parents=True)
    (turn_dir / "status.txt").write_text("PARSE_FAIL")

    calls = []

    async def agent(conversation):
        calls.append([dict(m) for m in conversation])
        if len(calls) == 1:
            return '{"notes": "forgot playbook"}'
        assert "failed schema validation" in conversation[-1]["content"]
        return '{"playbook": "## STRATEGIES\\n[gen-00001] helpful=0 harmful=0 :: repaired"}'

    refresh_fn = build_ace_context_refresh_fn(agent)
    playbook = run(refresh_fn(
        playbook="",
        events=[{
            "node_path": "root/leaf",
            "status": "PARSE_FAIL",
            "session_index": 0,
            "session_turn": 0,
            "global_turn": 0,
        }],
        metadata={"node_path": "root/leaf"},
        attempt_dir=attempt_dir,
        completed_session_index=0,
        next_session_index=1,
    ))

    assert "repaired" in playbook
    assert len(calls) == 2
    assert (attempt_dir / "ace_curator" / "session_0_to_1_repair_1"
            / "response.txt").exists()


def test_ace_context_manager_passes_attempt_dir_to_refresh_fn(tmp_path):
    from src.autotune2.ace_context import AceContextConfig, AceContextManager

    seen_attempt_dir = None

    async def refresh_fn(
        *,
        playbook,
        events,
        metadata,
        attempt_dir,
        completed_session_index,
        next_session_index,
    ):
        nonlocal seen_attempt_dir
        seen_attempt_dir = attempt_dir
        return playbook + "\nupdated"

    manager = AceContextManager(
        AceContextConfig(enabled=True, refresh_interval_turns=1),
        refresh_fn=refresh_fn,
    )
    run(manager.refresh(
        tmp_path / "attempt",
        completed_session_index=0,
        next_session_index=1,
        events=[{"status": "PARSE_FAIL"}],
        metadata={"node_path": "root/leaf"},
    ))

    assert seen_attempt_dir == tmp_path / "attempt"
