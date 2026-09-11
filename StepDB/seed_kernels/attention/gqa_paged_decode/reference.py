"""PyTorch reference: Batched GQA decode with a paged KV cache (page_size=1).

Ported verbatim from the flashinfer_trace definition
`gqa_paged_decode_h32_kv16_d128_ps1.json` (Gemma 3 27B, TP=1).

The `run` function below is the definition's reference implementation unchanged.
Inputs come from StepDB/precompute.py via the `tensors` arg.

Source JSON:
  flashinfer-bench/flashinfer_trace/definitions/gqa_paged/
    gqa_paged_decode_h32_kv16_d128_ps1.json
"""
import math

import torch


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 16
    assert head_dim == 128
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    k_cache_flat = k_cache.squeeze(1).to(
        torch.float32
    )  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(
        torch.float32
    )  # [num_pages, num_kv_heads, head_dim]

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_start >= page_end:
            output[b].zero_()
            continue

        token_indices = kv_indices[page_start:page_end].to(torch.long)
        num_tokens = token_indices.shape[0]

        if num_tokens == 0:
            output[b].zero_()
            continue

        k_batch = k_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        v_batch = v_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]

        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio

            q_head = q_batch[h]  # [head_dim]
            k_head = k_batch[:, kv_head]  # [num_tokens, head_dim]
            v_head = v_batch[:, kv_head]  # [num_tokens, head_dim]

            logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
            logits_scaled = logits * sm_scale

            lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
            out_head = torch.matmul(attn, v_head)  # [head_dim]
            output[b, h] = out_head.to(torch.bfloat16)

    return output, lse


def compute_gold(dims, tensors):
    output, _lse = run(
        tensors["q"], tensors["k_cache"], tensors["v_cache"],
        tensors["kv_indptr"], tensors["kv_indices"], tensors["sm_scale"],
    )
    return output
