"""PyTorch reference: Batched MLA decode with a paged KV cache (page_size=1).

Ported verbatim from the flashinfer_trace definition
`mla_paged_decode_h16_ckv512_kpe64_ps1.json` (DeepSeek-V3 / R1, TP=8:
  num_qo_heads=128/8=16, head_dim_ckv=512, head_dim_kpe=64).

The `run` function below is the definition's reference implementation unchanged.
Inputs come from StepDB/precompute.py via the `tensors` arg.

Source JSON:
  flashinfer-bench/flashinfer_trace/definitions/mla_paged/
    mla_paged_decode_h16_ckv512_kpe64_ps1.json
"""
import math

import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            # No KV cache for this batch element
            output[b].zero_()
            continue

        pages = kv_indices[page_beg:page_end]
        # Derive kv_len from kv_indptr (for page_size=1, num_pages == num_tokens)
        L_tokens = page_end - page_beg

        if L_tokens <= 0 or pages.numel() == 0:
            output[b].zero_()
            continue

        # Pages are token indices for page_size=1
        tok_idx = pages[:L_tokens].to(torch.long)

        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return output, lse


def compute_gold(dims, tensors):
    output, _lse = run(
        tensors["q_nope"], tensors["q_pe"],
        tensors["ckv_cache"], tensors["kpe_cache"],
        tensors["kv_indptr"], tensors["kv_indices"], tensors["sm_scale"],
    )
    return output
