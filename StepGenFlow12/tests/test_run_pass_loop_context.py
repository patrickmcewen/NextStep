import asyncio

from src.orchestrator import _run_pass_loop


def run(coro):
    return asyncio.run(coro)


def test_run_pass_loop_returns_failure_when_prompt_exhausts_context(tmp_path):
    class Agent:
        instructions = "tiny system prompt"
        __llm_config__ = {
            "context_window_tokens": 64,
            "output_token_margin": 20,
        }

    result = run(_run_pass_loop(
        Agent(),
        "refactor_final",
        "kernel",
        {},
        max_turns=1,
        ckpt_dir=tmp_path,
        executor="dsl",
        tensors={},
        log=lambda *_a, **_kw: None,
        check_order="correctness-first",
        prebuilt_user_prompt="large prompt " * 200,
    ))

    assert result["success"] is False
    assert result["code"] is None
    assert "prompt leaves no room" in result["last_messages"][-1]["content"]
    assert (
        tmp_path / "refactor_final" / "turn_0" / "status.txt"
    ).read_text().startswith("LLM_CONTEXT_EXHAUSTED:")
