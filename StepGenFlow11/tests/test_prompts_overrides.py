from src.prompts import build_pass_user_prompt


_NODE_REF = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x * 2

def get_inputs(dims):
    torch.manual_seed(101)
    return (torch.randn(dims["M"]),)

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""


def test_build_pass_user_prompt_uses_reference_override_when_provided():
    import torch
    tensors = {"x": torch.randn(4)}
    prompt = build_pass_user_prompt(
        pass_name="refactor_final",
        kernel_name="__synth_node__",
        dims={"M": 4},
        tensors=tensors,
        reference_code_override=_NODE_REF,
        precompute_source_override="def get_inputs(dims): ...  # synthetic",
    )
    assert "class Model" in prompt
    assert "def forward(self, x):" in prompt
    assert "def get_inputs(dims): ...  # synthetic" in prompt


def test_build_pass_user_prompt_falls_back_to_bench_config_when_no_override():
    from src.prompts import _load_stepdb_config
    config = _load_stepdb_config()
    sample_kernel = next(iter(config))
    sample_dims = next(iter(config[sample_kernel]["presets"].values()))
    prompt = build_pass_user_prompt(
        pass_name="refactor_final",
        kernel_name=sample_kernel,
        dims=sample_dims,
    )
    assert sample_kernel in prompt
