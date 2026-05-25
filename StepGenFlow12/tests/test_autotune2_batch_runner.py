from pathlib import Path

from run_autotune2_batch import (
    build_autotune2_command,
    flatten_cli_args,
    load_batch_config,
)


def test_flatten_cli_args_omits_nulls_and_emits_only_explicit_values():
    args = {
        "model": "gpt-oss-120b",
        "sim_mode": "agent",
        "include_sources": True,
        "ace_context": False,
        "sim_calibration_seed_path": "",
        "top_k": 1,
        "unused_default": None,
    }

    tokens = flatten_cli_args(args)

    assert tokens == [
        "--model", "gpt-oss-120b",
        "--sim-mode", "agent",
        "--include-sources",
        "--no-ace-context",
        "--sim-calibration-seed-path", "",
        "--top-k", "1",
    ]
    assert "--unused-default" not in tokens


def test_build_autotune2_command_keeps_outer_dir_positional_first():
    cmd = build_autotune2_command(
        outer_dir=Path("/workspace/checkpoints/example/kernel/outer_0"),
        python_exe="/usr/bin/python",
        run_autotune2=Path("/repo/run_autotune2.py"),
        autotune2_args={"model": "gpt-oss-120b", "include_sources": True},
    )

    assert cmd == [
        "/usr/bin/python",
        "/repo/run_autotune2.py",
        "/workspace/checkpoints/example/kernel/outer_0",
        "--model",
        "gpt-oss-120b",
        "--include-sources",
    ]


def test_load_batch_config_reads_outer_dirs_and_runtime_args(tmp_path: Path):
    config_path = tmp_path / "autotune2_batch.yaml"
    outer_a = tmp_path / "checkpoints" / "a" / "kernel" / "outer_0"
    outer_b = tmp_path / "checkpoints" / "b" / "kernel" / "outer_0"
    outer_a.mkdir(parents=True)
    outer_b.mkdir(parents=True)
    config_path.write_text(f"""
max_parallel: 2
results_root: /workspace/autotune2_batch_results
run_autotune2: /repo/run_autotune2.py
autotune2_args:
  model: gpt-oss-120b
  sim_mode: agent
outer_dirs:
  - {outer_a}
  - {outer_b}
""")

    config = load_batch_config(config_path)

    assert config.max_parallel == 2
    assert config.results_root == Path("/workspace/autotune2_batch_results")
    assert config.run_autotune2 == Path("/repo/run_autotune2.py")
    assert config.autotune2_args == {
        "model": "gpt-oss-120b",
        "sim_mode": "agent",
    }
    assert [job.outer_dir for job in config.jobs] == [
        outer_a,
        outer_b,
    ]
    assert [job.label for job in config.jobs] == [
        "a__kernel__outer_0",
        "b__kernel__outer_0",
    ]
