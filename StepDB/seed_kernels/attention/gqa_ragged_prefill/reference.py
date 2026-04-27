"""PyTorch reference: Batched GQA prefill with a ragged (variable-length) layout.

Ported verbatim from the flashinfer_trace definition
`gqa_ragged_prefill_causal_h32_kv16_d128.json` (Gemma 3 27B, TP=1).

The `run` function below is the definition's reference implementation unchanged.
This module wraps it with the StepDB `get_inputs` / `compute_gold` API:
uniform-length batches are constructed (every sequence has `q_len` queries
and `kv_len` keys) so that the ragged indptrs are deterministic.

Source JSON:
  flashinfer-bench/flashinfer_trace/definitions/gqa_ragged/
    gqa_ragged_prefill_causal_h32_kv16_d128.json
"""
import math

import torch
import torch.nn as nn

SEED = 42

NUM_QO_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 16
    assert head_dim == 128

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        q_batch = q_f32[q_start:q_end]
        k_batch = k_f32[kv_start:kv_end]
        v_batch = v_f32[kv_start:kv_end]

        num_q_tokens = q_batch.shape[0]
        num_kv_tokens = k_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens

        k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)

        logits = torch.einsum('qhd,khd->qhk', q_batch, k_expanded) * sm_scale

        q_positions = torch.arange(num_q_tokens, device=device)
        kv_positions = torch.arange(num_kv_tokens, device=device)
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], float('-inf'))

        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)
        output_batch = torch.einsum('qhk,khd->qhd', attn_weights, v_expanded)
        output[q_start:q_end] = output_batch.to(torch.bfloat16)

    return output, lse


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        output, _lse = run(q, k, v, qo_indptr, kv_indptr, sm_scale)
        return output


def get_inputs(dims):
    torch.manual_seed(SEED)
    batch_size = dims["batch_size"]
    q_len = dims["q_len"]
    kv_len = dims["kv_len"]
    assert kv_len >= q_len, "kv_len must be >= q_len for the causal mask"

    total_q = batch_size * q_len
    total_kv = batch_size * kv_len
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * q_len
    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * kv_len
    return q, k, v, qo_indptr, kv_indptr, sm_scale


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    return model(*get_inputs(dims))
