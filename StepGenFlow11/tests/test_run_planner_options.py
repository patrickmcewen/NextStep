import subprocess


def test_run_help_shows_plan_flags():
    out = subprocess.run(
        ["/workspace/miniconda3/bin/conda", "run", "-n", "testenv",
         "python", "/workspace/DEIOpt/StepGenFlow11/run.py", "--help"],
        capture_output=True, text=True, cwd="/workspace/DEIOpt/StepGenFlow11",
    )
    assert "--no-plan" in out.stdout
    assert "--max-replans" in out.stdout


def test_run_regression_help_shows_plan_flags():
    out = subprocess.run(
        ["/workspace/miniconda3/bin/conda", "run", "-n", "testenv",
         "python", "/workspace/DEIOpt/StepGenFlow11/run_regression.py", "--help"],
        capture_output=True, text=True, cwd="/workspace/DEIOpt/StepGenFlow11",
    )
    assert "--no-plan" in out.stdout
    assert "--max-replans" in out.stdout
