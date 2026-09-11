"""PyTorch reference: KernelBench Level 4 / Problem 7 — GPT-2 forward pass.

Ported from /workspace/KernelBench/KernelBench/level4/7_gpt2_bs32_seq256.py.

Important deviations from the KernelBench source:

1.  The KernelBench source uses ``AutoModelForCausalLM.from_pretrained("gpt2")``
    — i.e. it loads HuggingFace's actual pretrained weights from disk.
    Pretrained weights are not reproducible from a torch seed, so this port
    swaps them for a ``GPT2LMHeadModel(config)`` that is constructed under
    ``torch.manual_seed(SEED)``. Same architecture, random init.
2.  Dropout (resid/attn/embd) is hard-set to 0 in the config so the forward
    is deterministic regardless of train/eval mode.
3.  GPT-2's per-block weights are stacked along a leading layer axis (shape
    ``(L, *param_shape)``) so the entire model fits in 17 tensor args
    instead of ~150 — see the forward signature.

``Model.forward`` is hand-rolled and matches
``GPT2LMHeadModel(config)(input_ids).logits`` to within float32 rounding
(verified offline: max abs diff ~6e-8 vs. transformers' own forward, which
uses a slightly different op order inside attention). The hand-roll uses
the HuggingFace ``Conv1D`` convention (``x @ W + b`` with weight shape
``(in, out)``) so the extracted weights drop in unchanged. The matching
precompute lives at ``StepDB/precompute.py:_precompute_kernelbench_gpt2``.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


SEED = 42


def _build_gpt2_config(dims):
    return GPT2Config(
        vocab_size=dims["V"],
        n_positions=dims["P"],
        n_embd=dims["D"],
        n_layer=dims["L"],
        n_head=dims["H"],
        n_inner=dims["I"],
        activation_function="gelu_new",
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
    )


def _stack_block(sd, L, key_template):
    """Stack matrix-shaped per-layer params: each entry is already ≥2-D, so the
    stacked form is ≥3-D and a per-layer slice ``[i]`` is still ≥2-D."""
    return torch.stack(
        [sd[key_template.format(l=l)].detach().clone().contiguous() for l in range(L)],
        dim=0,
    ).contiguous()


def _stack_block_vec(sd, L, key_template):
    """Stack vector-shaped per-layer params (``(D,)``-style: LN affine, matmul
    biases) as ``(L, 1, D)`` so the per-layer slice ``[i]`` is ``(1, D)`` —
    2-D, satisfying StepGenFlow's on-chip ``rank >= 2`` invariant. Call sites
    that need a 1-D vector (``F.layer_norm`` weight/bias) ``.squeeze(0)`` at
    use; matmul ``+ bias`` broadcasts ``(1, D)`` over ``(B, T, D)`` unchanged."""
    return torch.stack(
        [
            sd[key_template.format(l=l)].detach().clone().contiguous().unsqueeze(0)
            for l in range(L)
        ],
        dim=0,
    ).contiguous()


def _gelu_new(x):
    # transformers' "gelu_new" / NewGELU activation
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Model(nn.Module):
    def __init__(self, n_head):
        # n_head is captured at __init__ time as a Python int so the forward
        # body can use it for compile-time reshape / sqrt without coercing
        # from a tensor inside the DSL leaf (which has no clean .item() path).
        super().__init__()
        self.n_head = n_head

    def forward(
        self,
        input_ids,
        causal_mask,
        wte,
        wpe,
        block_ln1_w,
        block_ln1_b,
        block_c_attn_w_q,
        block_c_attn_w_k,
        block_c_attn_w_v,
        block_c_attn_b_q,
        block_c_attn_b_k,
        block_c_attn_b_v,
        block_c_proj_w,
        block_c_proj_b,
        block_ln2_w,
        block_ln2_b,
        block_mlp_fc_w,
        block_mlp_fc_b,
        block_mlp_proj_w,
        block_mlp_proj_b,
        ln_f_w,
        ln_f_b,
    ):
        n_head = self.n_head
        B, T = input_ids.shape
        # block_ln1_w has shape (L, 1, D) — the unit dim makes each per-layer
        # slice [i] a (1, D) tensor (≥2-D, required by the STeP flow).
        L, _, D = block_ln1_w.shape
        assert D % n_head == 0, f"n_embd {D} must be divisible by n_head {n_head}"
        D_head = D // n_head

        positions = torch.arange(T, device=input_ids.device)
        x = wte[input_ids] + wpe[positions]  # (B, T, D)

        for layer in range(L):
            # --- attention sub-block (pre-norm + residual) ---
            # block_ln*_w/b are stored as (L, 1, D) so per-layer slices stay
            # ≥2-D for the STeP flow; F.layer_norm wants 1-D affine params,
            # so squeeze only at the call site.
            h = F.layer_norm(
                x, (D,), block_ln1_w[layer].squeeze(0), block_ln1_b[layer].squeeze(0)
            )
            # Conv1D convention: x @ W + b, weight stored as (in, out).
            # QKV weights/biases are pre-split column-wise in precompute so the
            # leaf can do three independent (D, D) linears instead of a fused
            # (D, 3D) matmul followed by an in-DSL split. Numerically identical.
            q = h @ block_c_attn_w_q[layer] + block_c_attn_b_q[layer]  # (B, T, D)
            k = h @ block_c_attn_w_k[layer] + block_c_attn_b_k[layer]
            v = h @ block_c_attn_w_v[layer] + block_c_attn_b_v[layer]
            q = q.view(B, T, n_head, D_head).transpose(1, 2)
            k = k.view(B, T, n_head, D_head).transpose(1, 2)
            v = v.view(B, T, n_head, D_head).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_head)
            # Additive causal mask: 0 on/below diag, -inf above. Broadcasts
            # over (B, H, *, *). Equivalent to masked_fill(bool, -inf) after
            # softmax, but expressed as a precomputed input + elementwise add
            # so it does not require Model.forward to fabricate a tensor.
            scores = scores + causal_mask
            attn = F.softmax(scores, dim=-1)
            ctx = attn @ v  # (B, H, T, D_head)
            ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
            ctx = ctx @ block_c_proj_w[layer] + block_c_proj_b[layer]
            x = x + ctx

            # --- MLP sub-block (pre-norm + residual) ---
            h = F.layer_norm(
                x, (D,), block_ln2_w[layer].squeeze(0), block_ln2_b[layer].squeeze(0)
            )
            h = h @ block_mlp_fc_w[layer] + block_mlp_fc_b[layer]
            h = _gelu_new(h)
            h = h @ block_mlp_proj_w[layer] + block_mlp_proj_b[layer]
            x = x + h

        # ln_f weights are stored as (1, D) so they are >=2-D and loadable by
        # StepDB's offchip_load. F.layer_norm wants 1-D affine params, so
        # squeeze just at the call site.
        x = F.layer_norm(x, (D,), ln_f_w.squeeze(0), ln_f_b.squeeze(0))
        logits = x @ wte.T  # tied LM head — wte is (V, D)
        return logits


def get_init_inputs(dims):
    # n_head is a compile-time integer captured on the Model instance — it
    # only ever drives integer arithmetic / view() in forward, so passing it
    # through the per-tensor input pipeline forced an unusable .item() hop
    # inside the DSL leaf.
    return [dims["H"]]


def _split_qkv(stacked, D):
    """Split a stacked (L, *, 3D) Conv1D-QKV tensor column-wise into three
    (L, *, D) tensors. Conv1D's fused (D, 3D) weight is column-stacked
    Q | K | V, so this is bit-identical to the runtime split."""
    q, k, v = stacked.split(D, dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def get_inputs(dims):
    torch.manual_seed(SEED)
    cfg = _build_gpt2_config(dims)

    # GPT2LMHeadModel(config) runs the full init under the seed. We extract
    # its parameters and stack the per-block ones along a new layer axis.
    model = GPT2LMHeadModel(cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    L = dims["L"]
    D = dims["D"]

    wte = sd["transformer.wte.weight"].clone().contiguous()
    wpe = sd["transformer.wpe.weight"].clone().contiguous()
    # LN affine + matmul biases get the (L, 1, D)-style layout so per-layer
    # slices stay 2-D. Matrix weights (already 2-D per layer) just stack.
    block_ln1_w = _stack_block_vec(sd, L, "transformer.h.{l}.ln_1.weight")
    block_ln1_b = _stack_block_vec(sd, L, "transformer.h.{l}.ln_1.bias")
    # Fused QKV weight (D, 3D) and bias (1, 3D) are pre-split column-wise
    # into three (D, D) / (1, D) tensors so the DSL leaf can do three
    # independent linears instead of an in-DSL split + parallelize chain.
    block_c_attn_w_stacked = _stack_block(sd, L, "transformer.h.{l}.attn.c_attn.weight")
    block_c_attn_b_stacked = _stack_block_vec(sd, L, "transformer.h.{l}.attn.c_attn.bias")
    block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v = _split_qkv(block_c_attn_w_stacked, D)
    block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v = _split_qkv(block_c_attn_b_stacked, D)
    block_c_proj_w = _stack_block(sd, L, "transformer.h.{l}.attn.c_proj.weight")
    block_c_proj_b = _stack_block_vec(sd, L, "transformer.h.{l}.attn.c_proj.bias")
    block_ln2_w = _stack_block_vec(sd, L, "transformer.h.{l}.ln_2.weight")
    block_ln2_b = _stack_block_vec(sd, L, "transformer.h.{l}.ln_2.bias")
    block_mlp_fc_w = _stack_block(sd, L, "transformer.h.{l}.mlp.c_fc.weight")
    block_mlp_fc_b = _stack_block_vec(sd, L, "transformer.h.{l}.mlp.c_fc.bias")
    block_mlp_proj_w = _stack_block(sd, L, "transformer.h.{l}.mlp.c_proj.weight")
    block_mlp_proj_b = _stack_block_vec(sd, L, "transformer.h.{l}.mlp.c_proj.bias")
    # Final LN params: stored as (1, D) so offchip_load (which requires the
    # underlying tensor to be >=2-D) can consume them. Model.forward squeezes
    # back to 1-D at the F.layer_norm call site.
    ln_f_w = sd["transformer.ln_f.weight"].clone().contiguous().unsqueeze(0)
    ln_f_b = sd["transformer.ln_f.bias"].clone().contiguous().unsqueeze(0)

    # Input token ids draw last — must be the final RNG-consuming call so
    # the precompute can mirror it.
    input_ids = torch.randint(0, dims["V"], (dims["B"], dims["T"]))

    # Causal mask (additive form): 0 on and below the diagonal, -inf above.
    # Shape (T, T). Deterministic / RNG-free, but lives in the input list so
    # the STeP impl can read it from offchip rather than fabricate it.
    T = dims["T"]
    causal_mask = torch.zeros(T, T, dtype=torch.float32)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )

    return [
        input_ids,
        causal_mask,
        wte,
        wpe,
        block_ln1_w,
        block_ln1_b,
        block_c_attn_w_q,
        block_c_attn_w_k,
        block_c_attn_w_v,
        block_c_attn_b_q,
        block_c_attn_b_k,
        block_c_attn_b_v,
        block_c_proj_w,
        block_c_proj_b,
        block_ln2_w,
        block_ln2_b,
        block_mlp_fc_w,
        block_mlp_fc_b,
        block_mlp_proj_w,
        block_mlp_proj_b,
        ln_f_w,
        ln_f_b,
    ]


def compute_gold(dims):
    with torch.no_grad():
        return Model(*get_init_inputs(dims))(*get_inputs(dims))
