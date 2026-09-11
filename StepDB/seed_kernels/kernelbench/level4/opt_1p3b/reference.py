"""PyTorch reference: KernelBench Level 4 — facebook/opt-1.3b forward pass (causal-LM).

Ported from:

  * ``/workspace/KernelBench/KernelBench/level4/4_facebook-opt-1p3b_bs32_seq256.py``
  * ``/workspace/KernelBench/KernelBench/level4/2_facebook-opt-1p3b_bs1_seq2047.py``
  * ``/workspace/KernelBench/KernelBench/level4/8_facebook-opt-1p3b_bs512_seq32.py``

All three KernelBench files share the same model
(``AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")`` →
``OPTForCausalLM``) and differ only in ``(batch_size, sequence_length)``.
They are exposed as three presets here.

Important deviations from the KernelBench source:

1.  The KernelBench source calls ``from_pretrained`` — i.e. it loads HF's
    actual pretrained weights from disk. Pretrained weights are not
    reproducible from a torch seed, so this port swaps them for an
    ``OPTForCausalLM(config)`` that is constructed under
    ``torch.manual_seed(SEED)``. Same architecture, random init.
2.  Dropouts (residual / attention / activation) are zeroed in the config
    so the forward is deterministic regardless of train/eval mode.
3.  ``use_cache=False`` is set on the config because the upstream KernelBench
    invocation does a single fresh forward (no KV cache reuse).
4.  Per-block weights are stacked along a leading layer axis (shape
    ``(L, *param_shape)``) so the entire model fits in 17 tensor args
    instead of ~250 — see the forward signature.
5.  OPT-1.3b sits in the ``do_layer_norm_before=True`` branch
    (PRE-norm, applies to all OPT sizes from 125m up to 175B except 350m).
    The hand-roll therefore puts LayerNorm BEFORE each sub-block, in
    contrast to the BART port which is POST-norm.
6.  OPT-1.3b has ``word_embed_proj_dim == hidden_size == 2048``, so the
    decoder's optional ``project_in``/``project_out`` linears are ``None``
    and are omitted from the forward signature. Smaller OPT variants
    (e.g. 350m) DO need these projections and would require a different
    forward signature; that's why the directory and bench key are pinned
    to the 1.3b variant rather than a generic ``opt``.

``Model.forward`` is hand-rolled and matches
``OPTForCausalLM(config)(input_ids).logits`` to within float32 rounding
(verified offline: max abs diff ~1e-6 on the tiny preset against
transformers' own forward, which keeps ``q * scaling`` separate from the
``q @ k.T`` matmul). The hand-roll uses the standard ``nn.Linear``
convention (``x @ W.T + b`` with weight shape ``(out, in)``) so the
extracted weights drop in unchanged. The matching precompute lives at
``StepDB/precompute.py:_precompute_kernelbench_opt_1p3b``.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OPTConfig, OPTForCausalLM


SEED = 42


def _build_opt_config(dims):
    return OPTConfig(
        vocab_size=dims["V"],
        hidden_size=dims["D"],
        ffn_dim=dims["I"],
        num_hidden_layers=dims["L"],
        num_attention_heads=dims["H"],
        max_position_embeddings=dims["P"],
        word_embed_proj_dim=dims["D"],  # OPT-1.3b has no project_in/project_out.
        activation_function="relu",
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        tie_word_embeddings=True,
        use_cache=False,
        pad_token_id=1,
    )


def _stack_block(sd, L, key_template):
    """Stack matrix-shaped per-layer params: each entry is already >=2-D, so the
    stacked form is >=3-D and a per-layer slice ``[i]`` is still >=2-D."""
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


class Model(nn.Module):
    def __init__(self, n_head):
        # n_head is captured at __init__ time as a Python int so the forward
        # body can use it for compile-time reshape / sqrt without coercing
        # from a tensor inside the DSL leaf.
        super().__init__()
        self.n_head = n_head

    def forward(
        self,
        input_ids,
        causal_mask,
        wte,
        wpe,
        block_self_attn_ln_w,
        block_self_attn_ln_b,
        block_self_attn_w_q,
        block_self_attn_w_k,
        block_self_attn_w_v,
        block_self_attn_w_out,
        block_self_attn_b_q,
        block_self_attn_b_k,
        block_self_attn_b_v,
        block_self_attn_b_out,
        block_final_ln_w,
        block_final_ln_b,
        block_fc1_w,
        block_fc1_b,
        block_fc2_w,
        block_fc2_b,
        decoder_ln_w,
        decoder_ln_b,
    ):
        n_head = self.n_head
        B, T = input_ids.shape
        # block_self_attn_ln_w has shape (L, 1, D) — the unit dim makes each
        # per-layer slice [i] a (1, D) tensor (>=2-D, required by the STeP flow).
        L, _, D = block_self_attn_ln_w.shape
        assert D % n_head == 0, f"hidden_size {D} must be divisible by n_head {n_head}"
        D_head = D // n_head
        scaling = 1.0 / math.sqrt(D_head)

        # Embeddings + learned positional (OPT's +2 offset; the wpe table is
        # sized (max_position + 2, D) at precompute time accordingly).
        positions = torch.arange(T, device=input_ids.device) + 2  # OPT's +2 offset
        x = wte[input_ids] + wpe[positions]  # (B, T, D)

        for layer in range(L):
            # ----- self-attention sub-block (PRE-norm + residual) -----
            residual = x
            # PRE-norm: LayerNorm BEFORE attention (do_layer_norm_before=True).
            # F.layer_norm wants 1-D affine params, so squeeze the stored (1, D).
            h = F.layer_norm(
                x,
                (D,),
                block_self_attn_ln_w[layer].squeeze(0),
                block_self_attn_ln_b[layer].squeeze(0),
            )
            # nn.Linear convention: x @ W.T + b, weight stored as (out, in).
            # OPT scales the query by 1/sqrt(D_head) before the q @ k.T matmul
            # (the transformers source keeps this split from the matmul to
            # match the original fairseq numerics — we replicate that order).
            q = (h @ block_self_attn_w_q[layer].T + block_self_attn_b_q[layer]) * scaling
            k = h @ block_self_attn_w_k[layer].T + block_self_attn_b_k[layer]
            v = h @ block_self_attn_w_v[layer].T + block_self_attn_b_v[layer]
            q = q.view(B, T, n_head, D_head).transpose(1, 2)
            k = k.view(B, T, n_head, D_head).transpose(1, 2)
            v = v.view(B, T, n_head, D_head).transpose(1, 2)
            scores = q @ k.transpose(-2, -1)  # already scaled via q * scaling
            # Additive causal mask: 0 on/below diag, -inf above. Broadcasts
            # over (B, H, *, *).
            scores = scores + causal_mask
            attn = F.softmax(scores, dim=-1)
            ctx = attn @ v  # (B, H, T, D_head)
            ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
            ctx = ctx @ block_self_attn_w_out[layer].T + block_self_attn_b_out[layer]
            x = residual + ctx

            # ----- FFN sub-block (PRE-norm + residual) -----
            residual = x
            h = F.layer_norm(
                x,
                (D,),
                block_final_ln_w[layer].squeeze(0),
                block_final_ln_b[layer].squeeze(0),
            )
            h = h @ block_fc1_w[layer].T + block_fc1_b[layer]
            h = F.relu(h)  # OPT-1.3b activation_function='relu'
            h = h @ block_fc2_w[layer].T + block_fc2_b[layer]
            x = residual + h

        # Decoder-level final LayerNorm (separate from per-layer final_ln above),
        # present whenever do_layer_norm_before=True and not _remove_final_layer_norm.
        # Stored as (1, D); squeeze at the call site.
        x = F.layer_norm(x, (D,), decoder_ln_w.squeeze(0), decoder_ln_b.squeeze(0))

        # OPTForCausalLM ties lm_head.weight to embed_tokens.weight (no bias).
        logits = x @ wte.T
        return logits


def get_init_inputs(dims):
    # n_head is a compile-time integer captured on the Model instance — it
    # only ever drives integer arithmetic / view() in forward, mirroring the
    # gpt2 / bart_large ports.
    return [dims["H"]]


def get_inputs(dims):
    torch.manual_seed(SEED)
    cfg = _build_opt_config(dims)

    # OPTForCausalLM(config) runs the full init under the seed. This
    # includes the lm_head weight that is later overwritten by tying — that
    # init still consumes RNG, so we must NOT skip it. We extract the
    # parameters we actually need.
    model = OPTForCausalLM(cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    L = dims["L"]

    wte = sd["model.decoder.embed_tokens.weight"].clone().contiguous()
    # OPTLearnedPositionalEmbedding sizes its table as (max_position + 2, D)
    # because of the +2 offset baked into its forward.
    wpe = sd["model.decoder.embed_positions.weight"].clone().contiguous()

    # Per-layer self-attention PRE-norm (do_layer_norm_before=True).
    block_self_attn_ln_w = _stack_block_vec(
        sd, L, "model.decoder.layers.{l}.self_attn_layer_norm.weight"
    )
    block_self_attn_ln_b = _stack_block_vec(
        sd, L, "model.decoder.layers.{l}.self_attn_layer_norm.bias"
    )

    # OPT uses nn.Linear (weight shape (out, in)) and stores q/k/v/out
    # separately (no fused-QKV like GPT-2's Conv1D), so no column split needed.
    block_self_attn_w_q = _stack_block(sd, L, "model.decoder.layers.{l}.self_attn.q_proj.weight")
    block_self_attn_w_k = _stack_block(sd, L, "model.decoder.layers.{l}.self_attn.k_proj.weight")
    block_self_attn_w_v = _stack_block(sd, L, "model.decoder.layers.{l}.self_attn.v_proj.weight")
    block_self_attn_w_out = _stack_block(sd, L, "model.decoder.layers.{l}.self_attn.out_proj.weight")
    block_self_attn_b_q = _stack_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.q_proj.bias")
    block_self_attn_b_k = _stack_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.k_proj.bias")
    block_self_attn_b_v = _stack_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.v_proj.bias")
    block_self_attn_b_out = _stack_block_vec(sd, L, "model.decoder.layers.{l}.self_attn.out_proj.bias")

    # Per-layer FFN PRE-norm (this is the layer's ``final_layer_norm`` in OPT
    # parlance — it sits between the attention sub-block and the FFN). Distinct
    # from the decoder-level final LN we extract below.
    block_final_ln_w = _stack_block_vec(sd, L, "model.decoder.layers.{l}.final_layer_norm.weight")
    block_final_ln_b = _stack_block_vec(sd, L, "model.decoder.layers.{l}.final_layer_norm.bias")

    # FFN: fc1 (D -> I), fc2 (I -> D), both with biases.
    block_fc1_w = _stack_block(sd, L, "model.decoder.layers.{l}.fc1.weight")
    block_fc1_b = _stack_block_vec(sd, L, "model.decoder.layers.{l}.fc1.bias")
    block_fc2_w = _stack_block(sd, L, "model.decoder.layers.{l}.fc2.weight")
    block_fc2_b = _stack_block_vec(sd, L, "model.decoder.layers.{l}.fc2.bias")

    # Decoder-level final LayerNorm (applied after the last decoder layer when
    # do_layer_norm_before=True and not _remove_final_layer_norm). Stored as
    # (1, D) so offchip_load (>=2-D underlying) can consume it; forward
    # squeezes back to 1-D for F.layer_norm.
    decoder_ln_w = sd["model.decoder.final_layer_norm.weight"].clone().contiguous().unsqueeze(0)
    decoder_ln_b = sd["model.decoder.final_layer_norm.bias"].clone().contiguous().unsqueeze(0)

    # Input token ids — final RNG-consuming call, mirrored in precompute.
    input_ids = torch.randint(0, dims["V"], (dims["B"], dims["T"]))

    # Causal mask (additive form): 0 on/below the diagonal, -inf above.
    # Shape (T, T). Deterministic / RNG-free, but lives in the input list so
    # the STeP impl can offchip-load it rather than fabricate it.
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
        block_self_attn_ln_w,
        block_self_attn_ln_b,
        block_self_attn_w_q,
        block_self_attn_w_k,
        block_self_attn_w_v,
        block_self_attn_w_out,
        block_self_attn_b_q,
        block_self_attn_b_k,
        block_self_attn_b_v,
        block_self_attn_b_out,
        block_final_ln_w,
        block_final_ln_b,
        block_fc1_w,
        block_fc1_b,
        block_fc2_w,
        block_fc2_b,
        decoder_ln_w,
        decoder_ln_b,
    ]


def compute_gold(dims):
    with torch.no_grad():
        return Model(*get_init_inputs(dims))(*get_inputs(dims))
