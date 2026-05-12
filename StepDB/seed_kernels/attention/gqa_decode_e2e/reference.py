"""PyTorch reference: ragged-batched GQA decode + O-projection.

Extracted from ``seed_kernels/transformer_layer/end_to_end/reference.py:81-100``
(steps [6] GQA attention and [7] O-projection).

Per-batch: read the first ``seq_lens[b]`` entries of the per-batch KV cache,
compute a numerically-stable softmax (max-subtraction, no ``1/sqrt(d)``
scaling), then project the flattened attention output through
``o_proj_weight`` to ``hidden_dim``. Variable per-batch sequence lengths
(decode "ragged" pattern) are the distinguishing trait vs. the paged/tiled
GQA decode kernels.

Inputs come from StepDB/precompute.py via the ``tensors`` arg.
"""
import torch


def compute_gold(dims, tensors):
    Q = tensors["Q"]                          # [batch, num_heads, head_dim]
    k_cache = tensors["k_cache"]              # [batch, max_seq_len, num_kv_heads, head_dim]
    v_cache = tensors["v_cache"]              # [batch, max_seq_len, num_kv_heads, head_dim]
    seq_lens = tensors["seq_lens"]            # list[int] of length batch
    o_proj_weight = tensors["o_proj_weight"]  # [num_heads * head_dim, hidden_dim]

    batch, num_heads, head_dim = Q.shape
    num_kv_heads = k_cache.shape[2]
    assert num_heads % num_kv_heads == 0, (
        f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
    )
    query_per_kvhead = num_heads // num_kv_heads

    # Vectorize across kv-heads: view Q as [Hkv, qpkv, D] and permute the
    # per-batch K/V slice to [Hkv, S, D] so the matmul broadcasts the qpkv
    # query group across each kv-head.
    attn_output = torch.zeros(batch, num_heads, head_dim)
    Q_grouped = Q.view(batch, num_kv_heads, query_per_kvhead, head_dim)
    for i in range(batch):
        seq_len = seq_lens[i]
        q_i = Q_grouped[i]                                # [Hkv, qpkv, D]
        k_i = k_cache[i, :seq_len].permute(1, 0, 2)       # [Hkv, S, D]
        v_i = v_cache[i, :seq_len].permute(1, 0, 2)       # [Hkv, S, D]

        scores = q_i @ k_i.transpose(-1, -2)              # [Hkv, qpkv, S]
        row_max = scores.amax(dim=-1, keepdim=True)
        exp_scores = torch.exp(scores - row_max)
        context = exp_scores @ v_i                        # [Hkv, qpkv, D]
        attn_output[i] = (context / exp_scores.sum(dim=-1, keepdim=True)) \
            .reshape(num_heads, head_dim)

    attn_flat = attn_output.view(batch, num_heads * head_dim)
    return attn_flat @ o_proj_weight
