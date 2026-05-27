"""Unit tests for autotune2 prompt assembly + response parsing."""

import json
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_autotune_prompt_files_are_grouped_by_agent():
    prompt_root = PROJECT_ROOT / "prompts"
    expected = {
        "autotune/tile_shrink/autotune2_system.txt",
        "autotune/tile_shrink/autotune2_user_tile_shrink.txt",
        "autotune/tile_shrink/autotune_tile_shrink_fewshot.txt",
        "autotune/parallel/autotune2_system_parallel.txt",
        "autotune/parallel/autotune2_user_parallel.txt",
        "autotune/parallel/autotune_parallel_fewshot.txt",
        "autotune/parallel/autotune_parallel_system.txt",
        "autotune/general/autotune2_system_general.txt",
        "autotune/general/autotune2_user_general.txt",
        "autotune/general/autotune_general_fewshot.txt",
        "autotune/curation/system.txt",
        "autotune/curation/user.txt",
        "autotune/sim_manager/system.txt",
        "autotune/sim_manager/user.txt",
        "autotune/shared/autotune2_output_protocol_leaf.txt",
        "autotune/shared/autotune2_output_protocol_parent.txt",
        "autotune/shared/dsl_memory_notes.txt",
    }
    deprecated = {
        "autotune_system.txt",
        "autotune_memory_system.txt",
    }

    for rel in expected:
        assert (prompt_root / rel).exists(), rel
        assert not (prompt_root / Path(rel).name).exists(), rel
    for rel in deprecated:
        assert (prompt_root / rel).exists(), rel


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
    # Child selection is deterministic/system-side; neither prompt should ask
    # the LLM to pick child variants.
    assert "child_picks" not in parent
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


def test_general_system_prompt_merges_tile_shrink_and_parallel_guidance():
    out = build_autotune2_system_prompt(
        is_leaf=True, dsl_code="X", fewshot="general",
    )
    flat = " ".join(out.lower().split())
    assert "optimization recipes" in flat
    assert "Shrinking tile sizes" in out
    assert "Sources of parallelism" in out
    assert "binary_map_accum" in out
    assert "static_reassemble" in out
    assert "reducing tile sizes" in flat
    assert "headroom" in flat
    assert "before applying parallelism" in flat
    assert "Output protocol (leaf node)" in out


def test_general_fewshot_file_imports_specialized_fewshots():
    general_fewshot = (
        PROJECT_ROOT
        / "prompts"
        / "autotune"
        / "general"
        / "autotune_general_fewshot.txt"
    ).read_text()
    assert "{tile_shrink_fewshot}" in general_fewshot
    assert "{parallel_fewshot}" in general_fewshot
    assert len(general_fewshot.splitlines()) < 60

    rendered = build_autotune2_system_prompt(
        is_leaf=True, dsl_code="X", fewshot="general",
    )
    assert "{tile_shrink_fewshot}" not in rendered
    assert "{parallel_fewshot}" not in rendered
    assert "rms_norm" in rendered
    assert "Worked example: GEMM, independent M-axis parallelism" in rendered


# --- build_autotune2_user_prompt ---------------------------------------------


_PROMPT_KWARGS = dict(
    node_name="attention_softmax",
    function_signature="def attention_softmax(x, *, out_shapes):",
    pytorch_reference="class Model(nn.Module):\n    pass",
    baseline_dsl="def attention_softmax(x, *, out_shapes):\n    return x",
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
    # Direction is to minimize cycles while staying within budget.
    flat = " ".join(out.lower().split())
    assert "minimize cycle latency" in flat
    assert "on-chip memory budget" in flat


def test_user_prompt_specializes_per_fewshot():
    """tile_shrink and parallel templates should mention their own
    recipe by name so the user-prompt framing matches the system-prompt
    agent baked into the conversation."""
    shrink = build_autotune2_user_prompt(
        is_leaf=True, fewshot="tile_shrink", **_PROMPT_KWARGS,
    )
    parallel = build_autotune2_user_prompt(
        is_leaf=True, fewshot="parallel", **_PROMPT_KWARGS,
    )
    assert "tile-shrink" in shrink.lower()
    assert "tile-shrink" not in parallel.lower()
    assert "parallel" in parallel.lower()


def test_general_user_prompt_combines_memory_and_parallelism_strategy():
    out = build_autotune2_user_prompt(
        is_leaf=True, fewshot="general", **_PROMPT_KWARGS,
    )
    flat = " ".join(out.lower().split())
    assert "tile sizes" in flat
    assert "parallelism" in flat
    assert "headroom" in flat
    assert "before applying parallelism" in flat
    assert "minimize cycle latency" in flat


def test_user_prompt_rejects_unknown_fewshot():
    with pytest.raises(AssertionError, match="fewshot must be one of"):
        build_autotune2_user_prompt(
            is_leaf=True, fewshot="not_a_real_agent", **_PROMPT_KWARGS,
        )


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
    assert "child_picks" not in out
    assert "Pick exactly one" not in out


def test_user_prompt_parent_accepts_child_design_examples_without_variant_tables():
    out = build_autotune2_user_prompt(
        is_leaf=False,
        child_design_examples="### Child attention_block best design\n```python\n# child code\n```",
        **_PROMPT_KWARGS,
    )
    assert "Child design examples" in out
    assert "# child code" in out
    assert "Pick exactly one" not in out


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
parent_input_contracts:
  Q: {reshape: [8, 8, 64], permutation: [1, 0, 2]}
```

```python
def my_parent(Q, *, out_shapes):
    return None
```
Trailing chatter.
"""


def test_parse_parent_response_happy_path():
    r = parse_autotune2_response(_PARENT_RESPONSE_OK, is_leaf=False)
    assert r.input_contracts["Q"].reshape == (8, 8, 64)
    assert r.input_contracts["Q"].permutation == (1, 0, 2)
    assert "def my_parent" in r.dsl


def test_parse_parent_response_rejects_child_picks():
    body = """\
```yaml
child_picks:
  attention_block: 3
parent_input_contracts:
  Q: {reshape: [8, 8, 64], permutation: [1, 0, 2]}
```

```python
def my_parent(Q, *, out_shapes):
    return None
```
"""
    with pytest.raises(AssertionError, match="unexpected yaml keys"):
        parse_autotune2_response(body, is_leaf=False)


def test_parse_parent_response_ignores_expected_children_when_no_child_picks():
    r = parse_autotune2_response(
        _PARENT_RESPONSE_OK, is_leaf=False,
        expected_child_names=("attention_block", "moe_block", "extra_one"),
    )
    assert r.input_contracts["Q"].reshape == (8, 8, 64)


_LEAF_RESPONSE_OK = """\
```yaml
parent_input_contracts:
  Q: {reshape: [8, 8, 64], permutation: [1, 0, 2]}
```

```python
def gqa_attention(Q, K, V, *, out_shapes):
    return Q
```
"""


def test_parse_leaf_response_happy_path():
    r = parse_autotune2_response(_LEAF_RESPONSE_OK, is_leaf=True)
    assert r.input_contracts["Q"] == TensorContract(
        reshape=(8, 8, 64), permutation=(1, 0, 2))
    assert "def gqa_attention" in r.dsl


def test_parse_leaf_response_rejects_child_picks_key():
    bad = """\
```yaml
child_picks:
  whatever: 0
parent_input_contracts: {}
```
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="unexpected yaml keys"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_leaf_response_rejects_parent_output_contracts_key():
    """Output contracts are derived, not declared — including
    ``parent_output_contracts`` is now a hard parse error so the LLM
    is forced to drop it (and the prompt's intent is enforced).
    """
    bad = """\
```yaml
parent_input_contracts: {}
parent_output_contracts:
  out_0: {reshape: [4, 4], permutation: [0, 1]}
```
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="parent_output_contracts"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_response_rejects_missing_python_block():
    bad = """\
```yaml
parent_input_contracts: {}
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


def test_parse_response_rejects_malformed_yaml_as_parse_failure():
    bad = """\
```yaml
parent_input_contracts:
  res_add_0:      {reshape: [64, 8],    permutation: [0, 1]}
  expert_weights:{reshape: [64, 2, 1, 1],    permutation: [0, 1, 2, 3]}
```
```python
def x():
    pass
```
"""
    with pytest.raises(AssertionError, match="invalid yaml block"):
        parse_autotune2_response(bad, is_leaf=True)


def test_parse_response_rejects_unknown_contract_keys():
    bad = """\
```yaml
parent_input_contracts:
  Q: {reshape: [4, 4], permutation: [0, 1], extra: 99}
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
parent_input_contracts:
  Q: {reshape: [4, "x"], permutation: [0, 1]}
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
parent_input_contracts:
  Q: {reshape: [4, 4], permutation: [0, 0]}
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


# --- CurationAgent / SimDecisionAgent prompts (PR4) --------------------------


def test_curation_and_sim_manager_prompts_live_in_prompt_files():
    prompt_root = PROJECT_ROOT / "prompts" / "autotune"
    files = {
        "curation/system.txt": "You rank past simulator-calibration records",
        "curation/user.txt": "## Target composed source",
        "sim_manager/system.txt": "You decide, per variant",
        "sim_manager/user.txt": "## Variant context",
    }
    for rel, marker in files.items():
        text = (prompt_root / rel).read_text()
        assert marker in text, rel


def _curation_candidates(n: int):
    from src.autotune2.prompts import CurationCandidate
    return [
        CurationCandidate(
            record_id=f"rec{i:02x}",
            composed_source=f"# composed {i}\n",
            analytical_cycles=10 + i,
            rust_cycles=20 + i * 2,
            kernel="kern", preset="pre",
        )
        for i in range(n)
    ]


def test_build_curation_user_prompt_renders_target_and_candidates():
    from src.autotune2.prompts import build_curation_user_prompt
    cands = _curation_candidates(3)
    prompt = build_curation_user_prompt(
        target_source="def target(): return 1\n",
        candidates=cands, k=2,
    )
    # Target source block.
    assert "## Target composed source" in prompt
    assert "def target(): return 1" in prompt
    # Per-candidate header + cycles line.
    for c in cands:
        assert f"### Record {c.record_id}" in prompt
        assert f"analytical_cycles={c.analytical_cycles}" in prompt
        assert f"rust_cycles={c.rust_cycles}" in prompt
    # K is rendered.
    assert "Pick K=2 records" in prompt


def test_build_curation_user_prompt_rejects_empty_candidates():
    from src.autotune2.prompts import build_curation_user_prompt
    with pytest.raises(AssertionError, match="candidates must be non-empty"):
        build_curation_user_prompt(
            target_source="x", candidates=[], k=1,
        )


def test_build_curation_user_prompt_rejects_k_out_of_range():
    from src.autotune2.prompts import build_curation_user_prompt
    cands = _curation_candidates(2)
    with pytest.raises(AssertionError, match="k must be"):
        build_curation_user_prompt(target_source="x", candidates=cands, k=0)
    with pytest.raises(AssertionError, match="k must be"):
        build_curation_user_prompt(target_source="x", candidates=cands, k=5)


def test_build_curation_user_prompt_rejects_duplicate_record_ids():
    from src.autotune2.prompts import CurationCandidate, build_curation_user_prompt
    dup = [
        CurationCandidate(record_id="same", composed_source="a",
                          analytical_cycles=1, rust_cycles=1,
                          kernel="k", preset="p"),
        CurationCandidate(record_id="same", composed_source="b",
                          analytical_cycles=2, rust_cycles=2,
                          kernel="k", preset="p"),
    ]
    with pytest.raises(AssertionError, match="duplicate record_id"):
        build_curation_user_prompt(target_source="x", candidates=dup, k=1)


def test_parse_curation_response_picks_listed_ids():
    from src.autotune2.prompts import parse_curation_response
    body = (
        "Sure, here are the picks.\n\n"
        '```json\n{"record_ids": ["rec01", "rec03"]}\n```\n'
    )
    out = parse_curation_response(
        body, candidate_ids=["rec00", "rec01", "rec02", "rec03"], k=2,
    )
    assert out == ["rec01", "rec03"]


def test_parse_curation_response_accepts_raw_json_object():
    from src.autotune2.prompts import parse_curation_response
    out = parse_curation_response(
        '{"record_ids": ["rec01", "rec03"]}',
        candidate_ids=["rec00", "rec01", "rec02", "rec03"],
        k=2,
    )
    assert out == ["rec01", "rec03"]


def test_parse_curation_response_rejects_missing_fence():
    from src.autotune2.prompts import parse_curation_response
    with pytest.raises(json.JSONDecodeError):
        parse_curation_response(
            'not json',
            candidate_ids=["a"], k=1,
        )


def test_parse_curation_response_rejects_wrong_count():
    from src.autotune2.prompts import parse_curation_response
    body = '```json\n{"record_ids": ["a"]}\n```'
    with pytest.raises(AssertionError, match="exactly 2 record_ids"):
        parse_curation_response(body, candidate_ids=["a", "b"], k=2)


def test_parse_curation_response_rejects_unknown_id():
    from src.autotune2.prompts import parse_curation_response
    body = '```json\n{"record_ids": ["ghost"]}\n```'
    with pytest.raises(AssertionError, match="not in the candidate set"):
        parse_curation_response(body, candidate_ids=["a", "b"], k=1)


def test_parse_curation_response_rejects_duplicate_ids():
    from src.autotune2.prompts import parse_curation_response
    body = '```json\n{"record_ids": ["a", "a"]}\n```'
    with pytest.raises(AssertionError, match="duplicate record_id"):
        parse_curation_response(body, candidate_ids=["a", "b"], k=2)


def test_build_sim_decision_user_prompt_includes_budget_and_context():
    from src.autotune2.prompts import build_sim_decision_user_prompt
    cands = _curation_candidates(2)
    prompt = build_sim_decision_user_prompt(
        node_path="root/leaf", is_root=False, variant_kind="variant",
        attempt_index=2, turn_index=4,
        composed_source="def leaf(): return 1\n",
        analytical_cycles=42, analytical_on_chip=128,
        remaining_seconds=120.0, recent_rust_avg_sec=15.5,
        consumed_seconds=8.25,
        curated=cands,
    )
    assert "node_path=root/leaf" in prompt
    assert "variant_kind=variant" in prompt
    assert "cycles=42" in prompt
    assert "on_chip=128 bytes" in prompt
    assert "remaining_seconds=120.0" in prompt
    assert "recent_rust_avg_sec=15.50" in prompt
    assert "K=2" in prompt
    for c in cands:
        assert c.record_id in prompt


def test_build_sim_decision_user_prompt_renders_inf_budget():
    """An unlimited budget surfaces as 'inf' so the decision agent reads
    it as a no-cap signal."""
    from src.autotune2.prompts import build_sim_decision_user_prompt
    prompt = build_sim_decision_user_prompt(
        node_path="root", is_root=True, variant_kind="baseline",
        attempt_index=-1, turn_index=-1,
        composed_source="x", analytical_cycles=1, analytical_on_chip=1,
        remaining_seconds=float("inf"), recent_rust_avg_sec=0.0,
        consumed_seconds=0.0,
        curated=[],
    )
    assert "remaining_seconds=inf" in prompt


def test_build_sim_decision_user_prompt_handles_empty_curated():
    from src.autotune2.prompts import build_sim_decision_user_prompt
    prompt = build_sim_decision_user_prompt(
        node_path="r", is_root=True, variant_kind="baseline",
        attempt_index=-1, turn_index=-1,
        composed_source="x", analytical_cycles=1, analytical_on_chip=1,
        remaining_seconds=60.0, recent_rust_avg_sec=0.0,
        consumed_seconds=0.0,
        curated=[],
    )
    assert "K=0" in prompt
    assert "no curated records" in prompt


def test_build_sim_decision_user_prompt_rejects_unknown_variant_kind():
    from src.autotune2.prompts import build_sim_decision_user_prompt
    with pytest.raises(AssertionError, match="variant_kind"):
        build_sim_decision_user_prompt(
            node_path="r", is_root=True, variant_kind="bogus",
            attempt_index=0, turn_index=0,
            composed_source="x", analytical_cycles=1, analytical_on_chip=1,
            remaining_seconds=60.0, recent_rust_avg_sec=0.0,
            consumed_seconds=0.0, curated=[],
        )


def test_parse_sim_decision_response_accepts_both_decisions():
    from src.autotune2.prompts import parse_sim_decision_response
    for d in ("rust", "analytical"):
        body = f'```json\n{{"decision": "{d}", "reason": "ok"}}\n```'
        decision, reason = parse_sim_decision_response(body)
        assert decision == d
        assert reason == "ok"


def test_parse_sim_decision_response_accepts_raw_json_object():
    from src.autotune2.prompts import parse_sim_decision_response
    decision, reason = parse_sim_decision_response(
        '{"decision": "rust", "reason": "provider omitted fence"}'
    )
    assert decision == "rust"
    assert reason == "provider omitted fence"


def test_parse_sim_decision_response_rejects_unknown_decision():
    from src.autotune2.prompts import parse_sim_decision_response
    body = '```json\n{"decision": "maybe", "reason": "hmm"}\n```'
    with pytest.raises(AssertionError, match="must be 'rust' or 'analytical'"):
        parse_sim_decision_response(body)


def test_parse_sim_decision_response_rejects_missing_keys():
    from src.autotune2.prompts import parse_sim_decision_response
    body = '```json\n{"decision": "rust"}\n```'
    with pytest.raises(AssertionError, match="'decision' and 'reason'"):
        parse_sim_decision_response(body)


def test_build_ace_context_curator_prompts_render_from_autotune_prompt_dir():
    from src.autotune2.prompts import (
        build_ace_context_curator_system_prompt,
        build_ace_context_curator_user_prompt,
    )

    system = build_ace_context_curator_system_prompt()
    assert "autotune2 context curator" in system.lower()
    assert "{current_playbook}" not in system

    user = build_ace_context_curator_user_prompt(
        current_playbook=(
            "## STRATEGIES\n"
            "[gen-00001] helpful=0 harmful=0 :: keep tile reuse local"
        ),
        events=[{
            "node_path": "root/leaf",
            "status": "ACCEPTED",
            "session_index": 1,
            "session_turn": 0,
            "global_turn": 4,
            "cycles": 10,
            "on_chip": 20,
        }],
        metadata={"node_path": "root/leaf", "fewshot": "tile_shrink"},
        turn_summaries=[{
            "session_index": 1,
            "session_turn": 0,
            "status": "ACCEPTED",
            "status_text": "ACCEPTED",
            "score": {"entries": [{"cycles": 10, "on_chip": 20}]},
        }],
    )
    assert "root/leaf" in user
    assert "keep tile reuse local" in user
    assert '"global_turn": 4' in user
    assert '"score"' in user
    assert "{events_json}" not in user


def test_parse_ace_context_curator_response_extracts_playbook():
    from src.autotune2.prompts import parse_ace_context_curator_response

    response = (
        "```json\n"
        "{\"playbook\": \"## STRATEGIES\\n"
        "[gen-00001] helpful=0 harmful=0 :: prefer fused loads\", "
        "\"notes\": \"added one bullet\"}\n"
        "```"
    )
    playbook = parse_ace_context_curator_response(response)
    assert "prefer fused loads" in playbook


def test_parse_ace_context_curator_response_accepts_raw_json_object():
    from src.autotune2.prompts import parse_ace_context_curator_response

    response = (
        "{\"playbook\": \"## STRATEGIES\\n"
        "[gen-00001] helpful=0 harmful=0 :: accept raw JSON\", "
        "\"notes\": \"provider omitted fence\"}"
    )
    playbook = parse_ace_context_curator_response(response)
    assert "accept raw JSON" in playbook


def test_parse_ace_context_curator_response_rejects_empty_playbook():
    from src.autotune2.prompts import parse_ace_context_curator_response

    with pytest.raises(AssertionError, match="non-empty"):
        parse_ace_context_curator_response("```json\n{\"playbook\": \"\"}\n```")


def test_curation_and_decision_system_prompts_are_static_and_nonempty():
    from src.autotune2.prompts import (
        build_curation_system_prompt,
        build_sim_decision_system_prompt,
    )
    # Two calls return byte-identical strings (static, no inputs).
    assert build_curation_system_prompt() == build_curation_system_prompt()
    assert build_sim_decision_system_prompt() == build_sim_decision_system_prompt()
    # Substantive content (avoids trivially-empty stub).
    cur = build_curation_system_prompt().lower()
    assert "calibration" in cur and "record" in cur, (
        f"curation system prompt missing core domain terms; first 300 chars: "
        f"{cur[:300]!r}"
    )
    dec = build_sim_decision_system_prompt().lower()
    assert "decision" in dec or "decide" in dec, (
        f"sim-decision system prompt missing core domain terms; first 300 "
        f"chars: {dec[:300]!r}"
    )


# --- FinalPickAgent prompts (PR5) --------------------------------------------


def _final_pick_candidates(n: int, *, curated_per_cand: int = 0):
    """Build N FinalPickCandidates, optionally each carrying
    ``curated_per_cand`` curation rows for the user-prompt curated block."""
    from src.autotune2.prompts import CurationCandidate, FinalPickCandidate
    out = []
    for i in range(n):
        curated = [
            CurationCandidate(
                record_id=f"rec{i}_{j}",
                composed_source=f"# rec source {i}-{j}\n",
                analytical_cycles=100 + j,
                rust_cycles=120 + j,
                kernel="k", preset="p",
            )
            for j in range(curated_per_cand)
        ]
        out.append(FinalPickCandidate(
            variant_index=i,
            cycles=10 * (i + 1),
            on_chip=100 - 10 * i,
            cycle_source="analytical" if i % 2 == 0 else "rust",
            composed_source=f"# variant {i}\n",
            curated=curated,
        ))
    return out


def test_build_final_pick_system_prompt_is_static_and_nonempty():
    from src.autotune2.prompts import build_final_pick_system_prompt
    assert build_final_pick_system_prompt() == build_final_pick_system_prompt()
    body = build_final_pick_system_prompt().lower()
    assert "pareto" in body
    assert "variant_index" in body


def test_build_final_pick_user_prompt_renders_all_candidates_and_curated_blocks():
    from src.autotune2.prompts import build_final_pick_user_prompt
    cands = _final_pick_candidates(3, curated_per_cand=2)
    prompt = build_final_pick_user_prompt(
        root_path="root/path", kernel="kern", preset="pre",
        candidates=cands,
    )
    assert "root_path=root/path" in prompt
    assert "kernel=kern" in prompt
    assert "N=3" in prompt
    for c in cands:
        assert f"### Candidate {c.variant_index}" in prompt
        assert f"cycles={c.cycles} ({c.cycle_source})" in prompt
        assert f"on_chip={c.on_chip} bytes" in prompt
        for r in c.curated:
            assert r.record_id in prompt
            assert f"analytical_cycles={r.analytical_cycles}" in prompt
    assert "variant_index in [0, 2]" in prompt


def test_build_final_pick_user_prompt_renders_empty_curated_for_cold_start():
    from src.autotune2.prompts import build_final_pick_user_prompt
    cands = _final_pick_candidates(2, curated_per_cand=0)
    prompt = build_final_pick_user_prompt(
        root_path="r", kernel="k", preset="p", candidates=cands,
    )
    assert "K=0" in prompt
    assert "no curated records" in prompt


def test_build_final_pick_user_prompt_rejects_empty_candidates():
    from src.autotune2.prompts import build_final_pick_user_prompt
    with pytest.raises(AssertionError, match="candidates must be non-empty"):
        build_final_pick_user_prompt(
            root_path="r", kernel="k", preset="p", candidates=[],
        )


def test_build_final_pick_user_prompt_rejects_misindexed_candidate():
    """``variant_index`` must equal the candidate's list position so the
    parser's ``[0, N-1]`` range matches what the LLM is told to pick."""
    from src.autotune2.prompts import FinalPickCandidate, build_final_pick_user_prompt
    misindexed = [
        FinalPickCandidate(
            variant_index=0, cycles=1, on_chip=1, cycle_source="analytical",
            composed_source="a", curated=[],
        ),
        FinalPickCandidate(
            variant_index=3,  # should be 1
            cycles=1, on_chip=1, cycle_source="analytical",
            composed_source="b", curated=[],
        ),
    ]
    with pytest.raises(AssertionError, match="variant_index"):
        build_final_pick_user_prompt(
            root_path="r", kernel="k", preset="p", candidates=misindexed,
        )


def test_build_final_pick_user_prompt_rejects_unknown_cycle_source():
    from src.autotune2.prompts import FinalPickCandidate, build_final_pick_user_prompt
    bad = [FinalPickCandidate(
        variant_index=0, cycles=1, on_chip=1, cycle_source="estimate",
        composed_source="x", curated=[],
    )]
    with pytest.raises(AssertionError, match="cycle_source"):
        build_final_pick_user_prompt(
            root_path="r", kernel="k", preset="p", candidates=bad,
        )


def test_parse_final_pick_response_returns_index_and_reason():
    from src.autotune2.prompts import parse_final_pick_response
    body = (
        "Sure, I pick number two.\n\n"
        '```json\n{"variant_index": 2, "reason": "lowest rust cycles"}\n```'
    )
    idx, reason = parse_final_pick_response(body, num_candidates=4)
    assert idx == 2
    assert reason == "lowest rust cycles"


def test_parse_final_pick_response_rejects_missing_fence():
    from src.autotune2.prompts import parse_final_pick_response
    with pytest.raises(AssertionError, match="fenced ```json"):
        parse_final_pick_response(
            '{"variant_index": 0, "reason": "x"}', num_candidates=2,
        )


def test_parse_final_pick_response_rejects_out_of_range_index():
    from src.autotune2.prompts import parse_final_pick_response
    body = '```json\n{"variant_index": 5, "reason": "x"}\n```'
    with pytest.raises(AssertionError, match=r"variant_index.*in \[0,"):
        parse_final_pick_response(body, num_candidates=3)


def test_parse_final_pick_response_rejects_non_int_index():
    from src.autotune2.prompts import parse_final_pick_response
    body = '```json\n{"variant_index": "1", "reason": "x"}\n```'
    with pytest.raises(AssertionError, match="must be an int"):
        parse_final_pick_response(body, num_candidates=3)


def test_parse_final_pick_response_rejects_boolean_index():
    """``True``/``False`` are ``int`` subclasses in Python so we explicitly
    reject them — a JSON ``true`` should not pass as variant_index 1."""
    from src.autotune2.prompts import parse_final_pick_response
    body = '```json\n{"variant_index": true, "reason": "x"}\n```'
    with pytest.raises(AssertionError, match="must be an int"):
        parse_final_pick_response(body, num_candidates=3)


def test_parse_final_pick_response_rejects_missing_keys():
    from src.autotune2.prompts import parse_final_pick_response
    body = '```json\n{"variant_index": 0}\n```'
    with pytest.raises(AssertionError, match="'variant_index' and 'reason'"):
        parse_final_pick_response(body, num_candidates=1)
