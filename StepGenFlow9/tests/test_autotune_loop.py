"""Per-turn gate ladder for the DSL-form autotuner."""

from pathlib import Path

import pytest

from src import autotune as autotune_mod


# A correct, knob-free DSL source we can use as a baseline. tile sizes are
# kept tiny so the IR sim runs cheaply.
_VALID_DSL = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    b = offchip_load(tensors["B"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    c = binary_matmul(a, b)
    return offchip_store(c)
'''


@pytest.fixture
def stub_gates(monkeypatch):
    """Patch the four gate functions to controllable stubs."""
    state = {
        "dsl_correctness": "match=True",
        "translate_raises": False,
        "graph_correctness": "match=True",
        "measure_raises": False,
        "measure_result": (1234, "TIMING REPORT", "VERBOSE TIMING REPORT",
                           {"on_chip_per_node": {}, "off_chip_per_node": {},
                            "on_chip_bytes": 0, "off_chip_bytes": 0,
                            "pmu_buffer_bytes": None,
                            "pmu_utilization_pct": None}),
    }

    def fake_dsl(code, kernel_name, dims, tensors):
        if isinstance(state["dsl_correctness"], Exception):
            raise state["dsl_correctness"]
        return state["dsl_correctness"]

    def fake_translate(code):
        if state["translate_raises"]:
            raise RuntimeError("translate boom")
        return "TRANSLATED:" + code

    def fake_graph(code, kernel_name, dims, tensors):
        if isinstance(state["graph_correctness"], Exception):
            raise state["graph_correctness"]
        return state["graph_correctness"]

    def fake_measure(code, kernel_name, dims, tensors, hw_config, max_total_compute_bw):
        if state["measure_raises"]:
            raise RuntimeError("timing boom")
        return state["measure_result"]

    monkeypatch.setattr(autotune_mod, "_run_dsl_correctness", fake_dsl)
    monkeypatch.setattr(autotune_mod, "translate", fake_translate)
    monkeypatch.setattr(autotune_mod, "_run_graph_correctness", fake_graph)
    monkeypatch.setattr(autotune_mod, "_measure", fake_measure)
    return state


def _evaluate(**kw):
    return autotune_mod._evaluate_dsl_turn(
        dsl_code=_VALID_DSL,
        kernel_name="dummy",
        dims={},
        tensors={},
        hw_config={},
        max_total_compute_bw=128,
    )


def test_clean_pass_status_and_artifacts(stub_gates):
    out = _evaluate()
    assert out["status"] == "PASS"
    assert out["dsl_correctness_text"] == "match=True"
    assert out["dsl_shape_trace"] == ""
    assert out["translated_code"].startswith("TRANSLATED:")
    assert out["graph_correctness_text"] == "match=True"
    assert out["new_cycles"] == 1234
    assert out["new_report"] == "TIMING REPORT"
    assert out["new_verbose_report"] == "VERBOSE TIMING REPORT"
    assert out["translate_error_text"] is None
    assert out["timing_error_text"] is None


def test_gate1_stdout_is_captured_into_shape_trace(stub_gates, monkeypatch, capsys):
    """Anything gate 1 prints (e.g. step_dsl shape lines under STEP_DSL_TRACE=1)
    must end up in `dsl_shape_trace` and NOT leak to the autotuner's stdout."""
    def noisy_dsl(code, kernel_name, dims, tensors):
        print("[step_dsl] binary_matmul input shape(s): a=stream(64,)×tile(1,1024)")
        print("[step_dsl] binary_matmul output shape(s): stream(64,)×tile(1,1024)")
        return "match=True"

    monkeypatch.setattr(autotune_mod, "_run_dsl_correctness", noisy_dsl)

    out = _evaluate()
    assert out["status"] == "PASS"
    assert "[step_dsl] binary_matmul input shape(s)" in out["dsl_shape_trace"]
    assert "[step_dsl] binary_matmul output shape(s)" in out["dsl_shape_trace"]
    captured = capsys.readouterr()
    assert "[step_dsl]" not in captured.out


def test_dsl_fail_captures_shape_trace_for_llm_feedback(stub_gates, monkeypatch):
    """On gate 1 failure, the captured trace must be present in the result so
    the controller can surface it to the LLM alongside the mismatch."""
    def noisy_failing_dsl(code, kernel_name, dims, tensors):
        print("[step_dsl] binary_matmul input shape(s): a=stream(64,)×tile(1,1024)")
        return "match=False mismatch=...sample..."

    monkeypatch.setattr(autotune_mod, "_run_dsl_correctness", noisy_failing_dsl)

    out = _evaluate()
    assert out["status"] == "DSL_FAIL"
    assert out["dsl_shape_trace"].startswith("[step_dsl] binary_matmul")


def test_format_shape_trace_block_empty_returns_empty_string():
    assert autotune_mod._format_shape_trace_block("") == ""


def test_format_shape_trace_block_tail_trims_long_traces():
    """Traces longer than _SHAPE_TRACE_MAX_LINES keep the *tail* (lines just
    before the failing op are most informative)."""
    cap = autotune_mod._SHAPE_TRACE_MAX_LINES
    over = cap + 50
    lines = [f"line_{i}" for i in range(over)]
    block = autotune_mod._format_shape_trace_block("\n".join(lines))

    assert "earlier lines elided" in block
    assert "50 earlier lines elided" in block
    # First kept line is line_50 (over - cap = 50 lines were elided).
    assert "line_50" in block
    assert f"line_{over - 1}" in block
    # Lines from before the cut should be gone.
    assert "line_0\n" not in block


def test_dsl_fail_short_circuits_at_gate_1(stub_gates):
    stub_gates["dsl_correctness"] = "match=False mismatch=...sample..."
    out = _evaluate()
    assert out["status"] == "DSL_FAIL"
    assert out["dsl_correctness_text"].startswith("match=False")
    # Later gates did not run.
    assert out["translated_code"] is None
    assert out["graph_correctness_text"] is None
    assert out["new_cycles"] is None


def test_dsl_exec_raise_short_circuits_at_gate_1(stub_gates):
    stub_gates["dsl_correctness"] = RuntimeError("boom")
    out = _evaluate()
    assert out["status"] == "DSL_FAIL"
    assert "RuntimeError: boom" in out["dsl_correctness_text"]
    assert out["translated_code"] is None


def test_translate_error_short_circuits_at_gate_2(stub_gates):
    stub_gates["translate_raises"] = True
    out = _evaluate()
    assert out["status"] == "TRANSLATE_ERROR"
    assert out["dsl_correctness_text"] == "match=True"
    assert "translate boom" in out["translate_error_text"]
    assert out["translated_code"] is None
    assert out["graph_correctness_text"] is None


def test_ir_fail_short_circuits_at_gate_3(stub_gates):
    stub_gates["graph_correctness"] = "match=False sim_output=..."
    out = _evaluate()
    assert out["status"] == "IR_FAIL"
    assert out["dsl_correctness_text"] == "match=True"
    assert out["translated_code"].startswith("TRANSLATED:")
    assert out["graph_correctness_text"].startswith("match=False")
    assert out["new_cycles"] is None


def test_timing_error_short_circuits_at_step_4(stub_gates):
    stub_gates["measure_raises"] = True
    out = _evaluate()
    assert out["status"] == "TIMING_ERROR"
    assert "timing boom" in out["timing_error_text"]
    assert out["new_cycles"] is None


def test_run_autotune_reads_dsl_code_from_outer_dir(tmp_path):
    """The autotune entry point's resume path resolves to outer_<N>/dsl_code.py."""
    outer_dir = tmp_path / "outer_0"
    outer_dir.mkdir()
    (outer_dir / "dsl_code.py").write_text("# placeholder dsl source\n")

    code, src = autotune_mod._resolve_resume_dsl_with_source(
        str(outer_dir), kernel_name="dummy",
    )
    assert code == "# placeholder dsl source\n"
    assert src == outer_dir / "dsl_code.py"


def test_run_autotune_rejects_path_with_no_dsl_code(tmp_path):
    """Pointing at a directory with no dsl_code.py fails loudly."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises((AssertionError, FileNotFoundError)):
        autotune_mod._resolve_resume_dsl_with_source(
            str(empty_dir), kernel_name="dummy",
        )


# ---------- feasibility helper ----------

def test_check_feasibility_returns_true_when_no_constraints():
    mem_info = {"on_chip_bytes": 999999, "off_chip_bytes": 0,
                "pmu_buffer_bytes": None, "pmu_utilization_pct": None}
    assert autotune_mod._check_feasibility(mem_info, None) is True
    assert autotune_mod._check_feasibility(mem_info, {}) is True


def test_check_feasibility_true_when_all_under_bound():
    mem_info = {"on_chip_bytes": 100, "off_chip_bytes": 50,
                "pmu_buffer_bytes": None, "pmu_utilization_pct": None}
    assert autotune_mod._check_feasibility(
        mem_info, {"on_chip_bytes": 200}) is True


def test_check_feasibility_false_when_any_over_bound():
    mem_info = {"on_chip_bytes": 300, "off_chip_bytes": 50,
                "pmu_buffer_bytes": None, "pmu_utilization_pct": None}
    assert autotune_mod._check_feasibility(
        mem_info, {"on_chip_bytes": 200}) is False


def test_check_feasibility_at_bound_is_feasible():
    mem_info = {"on_chip_bytes": 200, "off_chip_bytes": 50,
                "pmu_buffer_bytes": None, "pmu_utilization_pct": None}
    assert autotune_mod._check_feasibility(
        mem_info, {"on_chip_bytes": 200}) is True


def test_check_feasibility_unknown_metric_asserts():
    with pytest.raises(AssertionError):
        autotune_mod._check_feasibility(
            {"on_chip_bytes": 0, "off_chip_bytes": 0,
             "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
            {"cycles": 1000})


def test_infeasibility_score_zero_when_no_constraints():
    mem_info = {"on_chip_bytes": 999, "off_chip_bytes": 999,
                "pmu_buffer_bytes": None, "pmu_utilization_pct": None}
    assert autotune_mod._infeasibility_score(mem_info, None) == 0
    assert autotune_mod._infeasibility_score(mem_info, {}) == 0


def test_infeasibility_score_zero_when_under_bound():
    mem_info = {"on_chip_bytes": 100, "off_chip_bytes": 50,
                "pmu_buffer_bytes": None, "pmu_utilization_pct": None}
    assert autotune_mod._infeasibility_score(
        mem_info, {"on_chip_bytes": 200, "off_chip_bytes": 100}) == 0


def test_infeasibility_score_sums_overshoot_across_metrics():
    mem_info = {"on_chip_bytes": 300, "off_chip_bytes": 80,
                "pmu_buffer_bytes": None, "pmu_utilization_pct": None}
    # on_chip overshoot=100, off_chip overshoot=30 -> 130
    assert autotune_mod._infeasibility_score(
        mem_info, {"on_chip_bytes": 200, "off_chip_bytes": 50}) == 130


# ---------- run_autotune feasibility-aware best ----------

def test_run_autotune_prefers_feasible_over_lower_cycles_infeasible(
    tmp_path, monkeypatch
):
    """When feasibility is set, a feasible-but-slower turn beats an
    infeasible-but-faster turn for the 'best' designation."""
    from pathlib import Path
    import asyncio

    # Stub StepDB config + tensor precompute (avoid loading real kernel data).
    monkeypatch.setattr(autotune_mod, "_load_stepdb_config", lambda: {
        "dummy": {"presets": {"small": {"M": 8, "N": 8}}}})
    monkeypatch.setattr(autotune_mod, "precompute_tensors",
                        lambda kernel, dims: {"A": None, "B": None})

    # Stub resume resolution to return a fixed DSL string.
    monkeypatch.setattr(autotune_mod, "_resolve_resume_dsl_with_source",
                        lambda r, k: ("BASELINE_DSL", Path(r)))

    # Per-call results, threaded through one shared iterator: the baseline
    # check pulls entry 0 via _evaluate_dsl_turn; per-turn calls pull
    # entries 1 and 2 via _evaluate_translate_and_timing (gate 1 is stubbed
    # to always PASS so it doesn't draw from the iterator).
    #   baseline:   infeasible @ 1000
    #   turn 0:     infeasible @  800 (faster but still over cap)
    #   turn 1:     feasible   @  900 (slower but under cap → must become best)
    seq = iter([
        {"status": "PASS", "translated_code": "T0", "new_cycles": 1000,
         "new_report": "R0", "new_verbose_report": "V0",
         "mem_info": {"on_chip_bytes": 400, "off_chip_bytes": 0,
                      "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
         "dsl_correctness_text": "match=True", "dsl_shape_trace": "",
         "translate_error_text": None, "graph_correctness_text": "match=True",
         "timing_error_text": None},
        {"status": "PASS", "translated_code": "T1", "new_cycles": 800,
         "new_report": "R1", "new_verbose_report": "V1",
         "mem_info": {"on_chip_bytes": 350, "off_chip_bytes": 0,
                      "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
         "translate_error_text": None, "graph_correctness_text": "match=True",
         "timing_error_text": None},
        {"status": "PASS", "translated_code": "T2", "new_cycles": 900,
         "new_report": "R2", "new_verbose_report": "V2",
         "mem_info": {"on_chip_bytes": 200, "off_chip_bytes": 0,
                      "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
         "translate_error_text": None, "graph_correctness_text": "match=True",
         "timing_error_text": None},
    ])
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_turn",
                        lambda *a, **kw: next(seq))
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_correctness",
                        lambda *a, **kw: {"status": "PASS",
                                          "dsl_correctness_text": "match=True",
                                          "dsl_shape_trace": ""})
    monkeypatch.setattr(autotune_mod, "_evaluate_translate_and_timing",
                        lambda *a, **kw: next(seq))

    # Stub agent runner — the loop just needs proposal text from each turn.
    class _Stub:
        # Includes offchip_load/offchip_store so the refactor_final
        # compliance check passes — this test exercises promote logic,
        # not compliance.
        final_output = (
            "```python\n"
            "def tiled_reference(dims, tensors):\n"
            "    x = offchip_load(tensors['A'])\n"
            "    return offchip_store(x)\n"
            "```"
        )
        new_items = ()
    async def fake_run(agent, conv):
        return _Stub()
    monkeypatch.setattr(autotune_mod.Runner, "run", fake_run)
    monkeypatch.setattr(autotune_mod, "make_autotune_agent",
                        lambda *a, **kw: object())

    result = asyncio.run(autotune_mod.run_autotune(
        kernel_name="dummy", preset="small",
        llm_config={"model": "x"},
        autotune_config={"hw_config": {}, "constraints": {"max_total_compute_bw": 1}, "max_turns": 2},
        resume_from=str(tmp_path / "fake.py"),
        max_turns=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        agent_variant="general",
        feasibility={"on_chip_bytes": 256},
        enable_judge=False,
    ))

    assert result["feasible"] is True
    assert result["best_cycles"] == 900   # feasible turn won, not the 800 infeasible turn
    assert result["best_on_chip_bytes"] == 200


def test_run_autotune_promotes_lower_overshoot_even_when_slower(
    tmp_path, monkeypatch
):
    """When both baseline and a turn are infeasible, the turn with lower
    overshoot becomes best even if its cycles regressed. A later faster
    turn that has *higher* overshoot must NOT displace it."""
    from pathlib import Path
    import asyncio

    monkeypatch.setattr(autotune_mod, "_load_stepdb_config", lambda: {
        "dummy": {"presets": {"small": {"M": 8, "N": 8}}}})
    monkeypatch.setattr(autotune_mod, "precompute_tensors",
                        lambda kernel, dims: {"A": None, "B": None})
    monkeypatch.setattr(autotune_mod, "_resolve_resume_dsl_with_source",
                        lambda r, k: ("BASELINE_DSL", Path(r)))

    # cap=256:
    #   baseline    on_chip=400 (over=144) cycles=1000
    #   turn 0      on_chip=300 (over= 44) cycles=1100  <- promote (lower overshoot)
    #   turn 1      on_chip=350 (over= 94) cycles= 900  <- reject (higher overshoot)
    # Shared iterator across _evaluate_dsl_turn (baseline) and
    # _evaluate_translate_and_timing (per-turn gates 2-4); gate 1 is stubbed
    # to always PASS so it doesn't draw from the iterator.
    seq = iter([
        {"status": "PASS", "translated_code": "T0", "new_cycles": 1000,
         "new_report": "R0", "new_verbose_report": "V0",
         "mem_info": {"on_chip_bytes": 400, "off_chip_bytes": 0,
                      "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
         "dsl_correctness_text": "match=True", "dsl_shape_trace": "",
         "translate_error_text": None, "graph_correctness_text": "match=True",
         "timing_error_text": None},
        {"status": "PASS", "translated_code": "T1", "new_cycles": 1100,
         "new_report": "R1", "new_verbose_report": "V1",
         "mem_info": {"on_chip_bytes": 300, "off_chip_bytes": 0,
                      "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
         "translate_error_text": None, "graph_correctness_text": "match=True",
         "timing_error_text": None},
        {"status": "PASS", "translated_code": "T2", "new_cycles": 900,
         "new_report": "R2", "new_verbose_report": "V2",
         "mem_info": {"on_chip_bytes": 350, "off_chip_bytes": 0,
                      "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
         "translate_error_text": None, "graph_correctness_text": "match=True",
         "timing_error_text": None},
    ])
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_turn",
                        lambda *a, **kw: next(seq))
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_correctness",
                        lambda *a, **kw: {"status": "PASS",
                                          "dsl_correctness_text": "match=True",
                                          "dsl_shape_trace": ""})
    monkeypatch.setattr(autotune_mod, "_evaluate_translate_and_timing",
                        lambda *a, **kw: next(seq))

    class _Stub:
        # Includes offchip_load/offchip_store so the refactor_final
        # compliance check passes — this test exercises promote logic,
        # not compliance.
        final_output = (
            "```python\n"
            "def tiled_reference(dims, tensors):\n"
            "    x = offchip_load(tensors['A'])\n"
            "    return offchip_store(x)\n"
            "```"
        )
        new_items = ()
    async def fake_run(agent, conv):
        return _Stub()
    monkeypatch.setattr(autotune_mod.Runner, "run", fake_run)
    monkeypatch.setattr(autotune_mod, "make_autotune_agent",
                        lambda *a, **kw: object())

    result = asyncio.run(autotune_mod.run_autotune(
        kernel_name="dummy", preset="small",
        llm_config={"model": "x"},
        autotune_config={"hw_config": {}, "constraints": {"max_total_compute_bw": 1}, "max_turns": 2},
        resume_from=str(tmp_path / "fake.py"),
        max_turns=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        agent_variant="general",
        feasibility={"on_chip_bytes": 256},
        enable_judge=False,
    ))

    assert result["feasible"] is False
    assert result["best_cycles"] == 1100      # turn 0 stuck (lower overshoot)
    assert result["best_on_chip_bytes"] == 300


def test_run_autotune_rejects_noncompliant_proposal(tmp_path, monkeypatch):
    """A proposal that passes all gates but uses a banned op (e.g. .unsqueeze)
    must NOT be promoted; per-turn status is NONCOMPLIANT and best stays at
    baseline cycles."""
    from pathlib import Path
    import asyncio

    monkeypatch.setattr(autotune_mod, "_load_stepdb_config", lambda: {
        "dummy": {"presets": {"small": {"M": 8, "N": 8}}}})
    monkeypatch.setattr(autotune_mod, "precompute_tensors",
                        lambda kernel, dims: {"A": None, "B": None})
    monkeypatch.setattr(autotune_mod, "_resolve_resume_dsl_with_source",
                        lambda r, k: ("BASELINE_DSL", Path(r)))

    # Baseline passes all gates. Turn 0's DSL passes gate 1 (DSL exec) but
    # uses .unsqueeze(), which refactor_final compliance bans — the loop
    # rejects it before reaching translate, so _evaluate_translate_and_timing
    # must never be invoked for turn 0.
    baseline_result = {
        "status": "PASS", "translated_code": "T0", "new_cycles": 1000,
        "new_report": "R0", "new_verbose_report": "V0",
        "mem_info": {"on_chip_bytes": 0, "off_chip_bytes": 0,
                     "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
        "dsl_correctness_text": "match=True", "dsl_shape_trace": "",
        "translate_error_text": None, "graph_correctness_text": "match=True",
        "timing_error_text": None,
    }
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_turn",
                        lambda *a, **kw: baseline_result)
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_correctness",
                        lambda *a, **kw: {"status": "PASS",
                                          "dsl_correctness_text": "match=True",
                                          "dsl_shape_trace": ""})

    def _must_not_translate(*a, **kw):
        raise AssertionError(
            "translate-and-timing should be skipped for noncompliant proposals")
    monkeypatch.setattr(autotune_mod, "_evaluate_translate_and_timing",
                        _must_not_translate)

    class _Stub:
        final_output = (
            "```python\n"
            "def tiled_reference(dims, tensors):\n"
            "    return tensors['A'].unsqueeze(0)\n"
            "```"
        )
        new_items = ()
    async def fake_run(agent, conv):
        return _Stub()
    monkeypatch.setattr(autotune_mod.Runner, "run", fake_run)
    monkeypatch.setattr(autotune_mod, "make_autotune_agent",
                        lambda *a, **kw: object())

    result = asyncio.run(autotune_mod.run_autotune(
        kernel_name="dummy", preset="small",
        llm_config={"model": "x"},
        autotune_config={"hw_config": {}, "constraints": {"max_total_compute_bw": 1}, "max_turns": 1},
        resume_from=str(tmp_path / "fake.py"),
        max_turns=1,
        checkpoint_dir=str(tmp_path / "ckpt"),
        agent_variant="general",
        enable_judge=False,
    ))

    # Faster proposal had banned ops -> NOT promoted, best stays at baseline.
    assert result["best_cycles"] == 1000
    status = (tmp_path / "ckpt" / "dummy" / "turn_0" / "status.txt").read_text()
    assert status == "NONCOMPLIANT"


def test_run_autotune_log_prefix_tags_stdout(tmp_path, monkeypatch, capsys):
    """`log_prefix` must be prepended to every stdout line from run_autotune so
    concurrent outer iterations stay distinguishable."""
    from pathlib import Path
    import asyncio

    monkeypatch.setattr(autotune_mod, "_load_stepdb_config", lambda: {
        "dummy": {"presets": {"small": {"M": 8, "N": 8}}}})
    monkeypatch.setattr(autotune_mod, "precompute_tensors",
                        lambda kernel, dims: {"A": None})
    monkeypatch.setattr(autotune_mod, "_resolve_resume_dsl_with_source",
                        lambda r, k: ("BASELINE_DSL", Path(r)))
    pass_result = {
        "status": "PASS", "translated_code": "T", "new_cycles": 1000,
        "new_report": "R", "new_verbose_report": "V",
        "mem_info": {"on_chip_bytes": 0, "off_chip_bytes": 0,
                     "pmu_buffer_bytes": None, "pmu_utilization_pct": None},
        "dsl_correctness_text": "match=True", "dsl_shape_trace": "",
        "translate_error_text": None, "graph_correctness_text": "match=True",
        "timing_error_text": None,
    }
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_turn",
                        lambda *a, **kw: pass_result)
    monkeypatch.setattr(autotune_mod, "_evaluate_dsl_correctness",
                        lambda *a, **kw: {"status": "PASS",
                                          "dsl_correctness_text": "match=True",
                                          "dsl_shape_trace": ""})
    monkeypatch.setattr(autotune_mod, "_evaluate_translate_and_timing",
                        lambda *a, **kw: pass_result)

    class _Stub:
        final_output = (
            "```python\n"
            "def tiled_reference(dims, tensors):\n"
            "    x = offchip_load(tensors['A'])\n"
            "    return offchip_store(x)\n"
            "```"
        )
        new_items = ()
    async def fake_run(agent, conv):
        return _Stub()
    monkeypatch.setattr(autotune_mod.Runner, "run", fake_run)
    monkeypatch.setattr(autotune_mod, "make_autotune_agent",
                        lambda *a, **kw: object())

    asyncio.run(autotune_mod.run_autotune(
        kernel_name="dummy", preset="small",
        llm_config={"model": "x"},
        autotune_config={"hw_config": {}, "constraints": {"max_total_compute_bw": 1}, "max_turns": 1},
        resume_from=str(tmp_path / "fake.py"),
        max_turns=1,
        checkpoint_dir=str(tmp_path / "ckpt"),
        agent_variant="general",
        enable_judge=False,
        log_prefix="[outer_7] ",
    ))

    out_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # Every non-blank line emitted by run_autotune carries the prefix.
    assert out_lines, "expected at least one stdout line from run_autotune"
    for ln in out_lines:
        assert ln.startswith("[outer_7] "), f"unprefixed line leaked: {ln!r}"
    # Sanity: the per-turn header and the baseline marker are present.
    joined = "\n".join(out_lines)
    assert "[autotune] Turn 1/1" in joined
    assert "Baseline total_cycles" in joined
