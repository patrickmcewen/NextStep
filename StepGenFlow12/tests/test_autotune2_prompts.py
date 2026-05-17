"""Unit tests for autotune2 prompt assembly + response parsing."""

import pytest
import yaml

from src.autotune2.contracts import TensorContract, vanilla_contract_for
from src.autotune2.contracts import DesignEntry
from src.autotune2.prompts import (
    AutotuneResponse,
    VariantSummary,
    build_autotune2_system_prompt,
    build_autotune2_user_prompt,
    contract_to_yaml_dict,
    parse_autotune2_response,
    render_accepted_summary,
    render_contract_human,
    render_variant_block,
)


# --- render_contract_human ---------------------------------------------------


def test_render_contract_human_identity_says_vanilla():
    c = vanilla_contract_for((4, 6, 8))
    assert render_contract_human(c, (4, 6, 8)) == "vanilla"


def test_render_contract_human_non_identity_shows_reshape_permute():
    c = TensorContract(reshape=(4, 6, 8), permutation=(2, 0, 1))
    assert render_contract_human(c, (4, 6, 8)) == "vanilla.reshape((4,6,8)).permute(2,0,1)"


def test_render_contract_human_only_reshape_changed_no_permute():
    # Same reshape (no factor change), nontrivial permutation
    c = TensorContract(reshape=(2, 2, 6, 8), permutation=(2, 0, 3, 1))
    s = render_contract_human(c, (4, 6, 8))
    assert "reshape((2,2,6,8))" in s
    assert "permute(2,0,3,1)" in s


# --- render_variant_block ----------------------------------------------------


def test_render_variant_block_baseline_plus_one_alt():
    block = render_variant_block(
        child_name="attention_block",
        arg_vanilla_shapes={"x": (64, 512), "w": (512, 512)},
        arg_is_raw={"x": True, "w": True},
        output_vanilla_shapes={"out_0": (64, 512)},
        variants=[
            VariantSummary(
                variant_index=0,
                input_contracts={},  # all RAW => no input contracts
                output_contracts={"out_0": vanilla_contract_for((64, 512))},
                cycles=12450, on_chip=81920,
            ),
            VariantSummary(
                variant_index=3,
                input_contracts={},
                output_contracts={
                    "out_0": TensorContract(
                        reshape=(16, 4, 512), permutation=(1, 0, 2)),
                },
                cycles=8200, on_chip=65536,
            ),
        ],
    )
    assert "Child: attention_block" in block
    assert "x (vanilla (64, 512)): RAW" in block
    assert "[0]" in block and "[3]" in block
    assert "cycles=12450" in block and "on_chip=81920" in block
    assert "cycles=8200" in block and "on_chip=65536" in block
    # variant 0 outputs are vanilla
    assert "out_0: vanilla" in block
    # variant 3 output is reshaped+permuted
    assert "reshape((16,4,512)).permute(1,0,2)" in block


def test_render_variant_block_rejects_contract_on_raw_arg():
    with pytest.raises(AssertionError, match="RAW arg"):
        render_variant_block(
            child_name="bad",
            arg_vanilla_shapes={"x": (4, 6)},
            arg_is_raw={"x": True},
            output_vanilla_shapes={"out_0": (4, 6)},
            variants=[
                VariantSummary(
                    variant_index=0,
                    input_contracts={"x": vanilla_contract_for((4, 6))},
                    output_contracts={},
                    cycles=0, on_chip=0,
                ),
            ],
        )


def test_render_variant_block_rejects_unknown_arg():
    with pytest.raises(AssertionError, match="not in"):
        render_variant_block(
            child_name="bad",
            arg_vanilla_shapes={"x": (4, 6)},
            arg_is_raw={"x": False},
            output_vanilla_shapes={"out_0": (4, 6)},
            variants=[
                VariantSummary(
                    variant_index=0,
                    input_contracts={"y": vanilla_contract_for((4, 6))},  # typo
                    output_contracts={},
                    cycles=0, on_chip=0,
                ),
            ],
        )


# --- build_autotune2_system_prompt -------------------------------------------


def test_system_prompt_embeds_step_dsl_code():
    out = build_autotune2_system_prompt(
        is_leaf=True, dsl_code="DSL_CODE_SENTINEL",
    )
    assert "DSL_CODE_SENTINEL" in out
    # Variant-generation framing is present
    assert "variant" in out.lower()


def test_system_prompt_distinguishes_leaf_and_parent():
    leaf = build_autotune2_system_prompt(is_leaf=True, dsl_code="X")
    parent = build_autotune2_system_prompt(is_leaf=False, dsl_code="X")
    # Parent protocol mentions child_picks; leaf protocol does not.
    assert "child_picks" in parent
    assert "child_picks" not in leaf


def test_system_prompt_requires_dsl_code():
    with pytest.raises(AssertionError, match="dsl_code"):
        build_autotune2_system_prompt(is_leaf=True, dsl_code="")


def test_system_prompt_embeds_tile_shrink_fewshot():
    # Both leaf and parent prompts must carry the tile-shrink recipe and
    # at least one of the worked examples, so the LLM can spot the
    # canonical rewrite (matmul → map_accum, rowwise_sum → accum_add, …)
    # without us having to spell it out in the user message.
    for is_leaf in (True, False):
        out = build_autotune2_system_prompt(is_leaf=is_leaf, dsl_code="X")
        assert "Shrinking tile sizes" in out, (
            f"is_leaf={is_leaf}: fewshot top header missing"
        )
        assert "binary_map_accum" in out, (
            f"is_leaf={is_leaf}: fewshot rule table missing"
        )
        assert "rms_norm" in out, (
            f"is_leaf={is_leaf}: Example 1 (rms_norm) missing"
        )
        # Ordering: fewshot must appear before the output protocol so the
        # LLM reads the recipe in the system-prompt narrative order.
        protocol_marker = (
            "Output protocol (leaf node)" if is_leaf
            else "Output protocol (parent node)"
        )
        assert out.index("Shrinking tile sizes") < out.index(protocol_marker), (
            f"is_leaf={is_leaf}: fewshot must precede the output protocol"
        )


# --- build_autotune2_user_prompt ---------------------------------------------


_PROMPT_KWARGS = dict(
    node_name="attention_softmax",
    function_signature="def attention_softmax(x, *, out_shapes):",
    pytorch_reference="class Model(nn.Module):\n    pass",
    pass1_dsl="def attention_softmax(x, *, out_shapes):\n    return x",
    dims_block="```json\n{\"B\": 2}\n```",
    tensors_block="x: (64, 512)",
)


def test_user_prompt_leaf_first_attempt_no_accepted_section():
    out = build_autotune2_user_prompt(
        is_leaf=True, accepted_summary="", **_PROMPT_KWARGS,
    )
    assert "attention_softmax" in out
    assert "class Model" in out
    assert "Already-accepted variants" not in out
    # Leaf prompts must not include child variant section
    assert "Child variant libraries" not in out
    # Per-node building blocks present
    assert "def attention_softmax(x, *, out_shapes):" in out
    assert "x: (64, 512)" in out


def test_user_prompt_leaf_with_accepted_summary_renders_block():
    out = build_autotune2_user_prompt(
        is_leaf=True,
        accepted_summary="  [0] cycles=1000, on_chip=2048",
        **_PROMPT_KWARGS,
    )
    assert "Already-accepted variants" in out
    assert "cycles=1000" in out


def test_user_prompt_parent_requires_child_variant_blocks():
    with pytest.raises(AssertionError, match="at least one child"):
        build_autotune2_user_prompt(
            is_leaf=False, child_variant_blocks={}, **_PROMPT_KWARGS,
        )


def test_user_prompt_leaf_rejects_child_variant_blocks():
    with pytest.raises(AssertionError, match="leaf prompts must not"):
        build_autotune2_user_prompt(
            is_leaf=True,
            child_variant_blocks={"foo": "<<>>"},
            **_PROMPT_KWARGS,
        )


def test_user_prompt_parent_embeds_variant_tables():
    out = build_autotune2_user_prompt(
        is_leaf=False,
        child_variant_blocks={
            "attention_block": "<<ATTN-TABLE>>",
            "moe_block": "<<MOE-TABLE>>",
        },
        **_PROMPT_KWARGS,
    )
    assert "<<ATTN-TABLE>>" in out
    assert "<<MOE-TABLE>>" in out
    assert "child_picks" in out


# --- render_accepted_summary -------------------------------------------------


def _design_entry(
    cycles, on_chip, *, input_contracts=None, output_contracts=None,
):
    return DesignEntry(
        dsl="def x(): pass",
        input_contracts=input_contracts or {},
        output_contracts=output_contracts or {},
        cycles=cycles, on_chip=on_chip,
        provenance="test",
    )


def test_accepted_summary_empty_returns_empty_string():
    out = render_accepted_summary(
        [], arg_vanilla_shapes={}, output_vanilla_shapes={},
    )
    assert out == ""


def test_accepted_summary_renders_cycles_and_contracts():
    out = render_accepted_summary(
        [
            _design_entry(
                1000, 2048,
                output_contracts={"out_0": vanilla_contract_for((64, 512))},
            ),
            _design_entry(
                500, 4096,
                output_contracts={
                    "out_0": TensorContract(
                        reshape=(16, 4, 512), permutation=(1, 0, 2),
                    ),
                },
            ),
        ],
        arg_vanilla_shapes={},
        output_vanilla_shapes={"out_0": (64, 512)},
    )
    assert "cycles=1000" in out and "on_chip=2048" in out
    assert "cycles=500" in out and "on_chip=4096" in out
    # First entry has vanilla output; second has reshape+permute.
    assert "vanilla" in out
    assert "reshape((16,4,512)).permute(1,0,2)" in out


# --- parse_autotune2_response ------------------------------------------------


_PARENT_RESPONSE_OK = """\
Some preamble from the agent...

```yaml
child_picks:
  attention_block: 3
  moe_block: 1
parent_input_contracts:
  Q: {reshape: [8, 8, 64], permutation: [1, 0, 2]}
parent_output_contracts:
  out_0: {reshape: [16, 4, 512], permutation: [1, 0, 2]}
```

```python
def my_parent(Q, *, out_shapes):
    return None
```
Trailing chatter.
"""


def test_parse_parent_response_happy_path():
    r = parse_autotune2_response(_PARENT_RESPONSE_OK, is_leaf=False)
    assert r.child_picks == {"attention_block": 3, "moe_block": 1}
    assert r.input_contracts["Q"].reshape == (8, 8, 64)
    assert r.input_contracts["Q"].permutation == (1, 0, 2)
    assert r.output_contracts["out_0"].reshape == (16, 4, 512)
    assert "def my_parent" in r.dsl


def test_parse_parent_response_validates_expected_children():
    with pytest.raises(AssertionError, match="must equal expected_child_names"):
        parse_autotune2_response(
            _PARENT_RESPONSE_OK, is_leaf=False,
            expected_child_names=("attention_block", "moe_block", "extra_one"),
        )


_LEAF_RESPONSE_OK = """\
```yaml
parent_input_contracts: {}
parent_output_contracts:
  out_0: {reshape: [64, 512], permutation: [0, 1]}
```

```python
def gqa_attention(Q, K, V, *, out_shapes):
    return Q
```
"""


def test_parse_leaf_response_happy_path():
    r = parse_autotune2_response(_LEAF_RESPONSE_OK, is_leaf=True)
    assert r.child_picks == {}
    assert r.input_contracts == {}
    assert r.output_contracts["out_0"] == TensorContract(
        reshape=(64, 512), permutation=(0, 1))
    assert "def gqa_attention" in r.dsl


def test_parse_leaf_response_rejects_child_picks_key():
    bad = """\
```yaml
child_picks:
  whatever: 0
parent_input_contracts: {}
parent_output_contracts: {}
```
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="unexpected yaml keys"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_response_rejects_missing_python_block():
    bad = """\
```yaml
parent_input_contracts: {}
parent_output_contracts: {}
```
"""
    with pytest.raises(AssertionError, match="```python"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_response_rejects_missing_yaml_block():
    bad = """\
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="```yaml"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_response_rejects_unknown_contract_keys():
    bad = """\
```yaml
parent_input_contracts: {}
parent_output_contracts:
  out_0: {reshape: [4, 4], permutation: [0, 1], extra: 99}
```
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="contract entry keys"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_response_rejects_non_int_reshape():
    bad = """\
```yaml
parent_input_contracts: {}
parent_output_contracts:
  out_0: {reshape: [4, "x"], permutation: [0, 1]}
```
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="'reshape' must be"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_response_invalid_permutation_caught_by_tensor_contract():
    """TensorContract.__post_init__ validates permutation; surfaces upstream."""
    bad = """\
```yaml
parent_input_contracts: {}
parent_output_contracts:
  out_0: {reshape: [4, 4], permutation: [0, 0]}
```
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="permutation must be"):
        parse_autotune2_response(bad, is_leaf=True)


# --- contract_to_yaml_dict (emission helper) ---------------------------------


def test_contract_to_yaml_dict_round_trip_via_yaml():
    c = TensorContract(reshape=(8, 8, 64), permutation=(1, 0, 2))
    y = contract_to_yaml_dict(c)
    text = yaml.safe_dump(y, default_flow_style=True).strip()
    parsed = yaml.safe_load(text)
    assert parsed == {"reshape": [8, 8, 64], "permutation": [1, 0, 2]}
