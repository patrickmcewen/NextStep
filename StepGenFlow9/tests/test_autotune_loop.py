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
        "measure_result": (1234, "TIMING REPORT"),
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
    assert out["translated_code"].startswith("TRANSLATED:")
    assert out["graph_correctness_text"] == "match=True"
    assert out["new_cycles"] == 1234
    assert out["new_report"] == "TIMING REPORT"
    assert out["translate_error_text"] is None
    assert out["timing_error_text"] is None


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
