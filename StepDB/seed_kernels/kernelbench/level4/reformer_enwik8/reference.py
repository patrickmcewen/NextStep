"""PyTorch reference: KernelBench Level 4 / Problem 13 — google/reformer-enwik8 forward pass.

Ported from /workspace/KernelBench/KernelBench/level4/13_google-reformer-enwik8_bs32_seq256.py.

Important deviations from the KernelBench source:

1.  The KernelBench source calls ``AutoModelForCausalLM.from_pretrained("google/reformer-enwik8")`` —
    i.e. it loads HuggingFace's actual pretrained weights from disk. Pretrained
    weights are not reproducible from a torch seed, so this port swaps them for
    a ``ReformerModelWithLMHead(config)`` constructed under
    ``torch.manual_seed(SEED)``. Same architecture, random init.
2.  Every dropout in the config is set to 0 so the forward is deterministic
    regardless of train/eval mode.
3.  ``use_cache=False``. Inference is a single fresh forward (no KV cache).
4.  ``hash_seed`` is set to a fixed integer (default ``None`` in the upstream
    config). Reformer's LSH attention draws random rotation matrices for
    bucket hashing inside ``forward``; without a hash seed those draws come
    from the global RNG state and produce non-deterministic outputs across
    successive forward calls. Setting ``hash_seed`` pins the rotation seed.
    In this port we additionally constrain ``T <= lsh_attn_chunk_length``, so
    LSH falls into HF's ``do_standard_self_attention`` path and the rotation
    machinery is never exercised — the seed is set for safety only.
5.  Per-layer Reformer weights have two slightly different shapes:
        * local layers carry ``query.weight, key.weight, value.weight``
        * LSH layers carry a single shared ``query_key.weight`` plus ``value.weight``
    To unify the per-layer signature into one stack of (L, D, D) tensors, the
    LSH layer's shared ``query_key.weight`` is duplicated into both ``q_w``
    and ``k_w`` slots. The forward dispatches on ``self.attn_layers[l]`` to
    pick the right math: local does ``k = K@x.T / sqrt(d_head)``; LSH does
    ``k = len_norm(query_key@x.T) / sqrt(d_head)`` (length normalisation, not
    scale only) with shared QK + self-masking.

Structural simplifications baked into the port (constrain presets accordingly):

  * ``T <= axial_pos_shape[1]`` — eval-mode axial position embedding for
    ``position_ids = arange(T)`` reduces to ``cat([w0[0,0,:], w1[0,p,:]])``,
    no axial-axis-0 traversal needed.
  * ``T <= lsh_attn_chunk_length`` — LSH stays in HF's
    ``do_standard_self_attention`` branch (no hashing / bucket sort).
  * ``T == n * local_attn_chunk_length`` for some n >= 1 — local-attn either
    runs un-chunked (``T == LOC_CL``) or chunks evenly. Avoids the HF
    auto-padding path entirely.

``Model.forward`` is hand-rolled and matches
``ReformerModelWithLMHead(config)(input_ids).logits`` bit-exactly at every
preset (verified — see the docstring of ``compute_gold`` below).  The matching
precompute lives at ``StepDB/precompute.py:_precompute_kernelbench_reformer_enwik8``.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ReformerConfig, ReformerModelWithLMHead


SEED = 42

# Reformer's own mask-value constants (see modeling_reformer.py: -1e9 for
# the general / causal mask, -1e5 for the LSH self-mask). The HF code stores
# them as registered buffers; we use them as raw scalars since they are
# arch-fixed.
_MASK_NEG = -1e9
_LSH_SELF_MASK_NEG = -1e5


def _build_reformer_config(dims):
    """Build a ReformerConfig from the dims dict.

    ``ATTN_LAYERS`` is a string of length ``L`` over the alphabet ``{'l','L'}``:
    ``l`` = local self-attention, ``L`` = LSH self-attention. This mirrors
    ``config.attn_layers`` (a list of "local"/"lsh" strings).
    """
    attn_layers = ["local" if c == "l" else "lsh" for c in dims["ATTN_LAYERS"]]
    assert len(attn_layers) == dims["L"], (
        f"ATTN_LAYERS length {len(attn_layers)} must equal L={dims['L']}"
    )
    assert dims["AE0"] + dims["AE1"] == dims["D"], (
        f"axial_pos_embds_dim sum {dims['AE0']}+{dims['AE1']} must equal hidden_size D={dims['D']}"
    )
    assert dims["AX0"] * dims["AX1"] == dims["MAX_POS"], (
        f"axial_pos_shape product {dims['AX0']}*{dims['AX1']} must equal MAX_POS={dims['MAX_POS']}"
    )
    return ReformerConfig(
        vocab_size=dims["V"],
        hidden_size=dims["D"],
        num_attention_heads=dims["H"],
        attention_head_size=dims["D_HEAD"],
        feed_forward_size=dims["FF"],
        num_hidden_layers=dims["L"],
        attn_layers=attn_layers,
        axial_pos_embds=True,
        axial_pos_shape=(dims["AX0"], dims["AX1"]),
        axial_pos_embds_dim=(dims["AE0"], dims["AE1"]),
        max_position_embeddings=dims["MAX_POS"],
        num_buckets=dims.get("NUM_BUCKETS", 8),
        num_hashes=1,  # unused in standard mode; kept at 1 for safety
        lsh_attn_chunk_length=dims["LSH_CL"],
        local_attn_chunk_length=dims["LOC_CL"],
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        chunk_size_lm_head=0,
        chunk_size_feed_forward=0,
        hash_seed=SEED,
        is_decoder=True,
        use_cache=False,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        lsh_attention_probs_dropout_prob=0.0,
        local_attention_probs_dropout_prob=0.0,
        hidden_act="relu",
        layer_norm_eps=1e-12,
        axial_norm_std=1.0,
        tie_word_embeddings=False,
    )


def _stack_block(sd, L, key_template):
    """Stack matrix-shaped per-layer params: each entry is already >=2-D, so the
    stacked form is >=3-D and a per-layer slice ``[i]`` is still >=2-D."""
    return torch.stack(
        [sd[key_template.format(l=l)].detach().clone().contiguous() for l in range(L)],
        dim=0,
    ).contiguous()


def _stack_block_vec(sd, L, key_template):
    """Stack vector-shaped per-layer params (``(D,)``-style: LN affine) as
    ``(L, 1, D)`` so the per-layer slice ``[i]`` is ``(1, D)`` — 2-D, as
    required by StepGenFlow's on-chip ``rank >= 2`` invariant.
    ``F.layer_norm`` callers ``.squeeze(0)`` at use."""
    return torch.stack(
        [
            sd[key_template.format(l=l)].detach().clone().contiguous().unsqueeze(0)
            for l in range(L)
        ],
        dim=0,
    ).contiguous()


def _stack_qkv_for_layer(sd, L, attn_types):
    """For each of the L layers, return (q_w, k_w, v_w) where:

      * local layers use distinct ``query.weight``, ``key.weight``, ``value.weight``
      * LSH layers use ``query_key.weight`` for BOTH q_w and k_w (shared QK),
        and ``value.weight`` for v_w

    Returned tensors are each (L, D, D) so the per-layer signature is uniform.
    """
    q_list, k_list, v_list = [], [], []
    for l in range(L):
        if attn_types[l] == "local":
            q = sd[f"reformer.encoder.layers.{l}.attention.self_attention.query.weight"]
            k = sd[f"reformer.encoder.layers.{l}.attention.self_attention.key.weight"]
            v = sd[f"reformer.encoder.layers.{l}.attention.self_attention.value.weight"]
        else:  # lsh
            qk = sd[f"reformer.encoder.layers.{l}.attention.self_attention.query_key.weight"]
            v = sd[f"reformer.encoder.layers.{l}.attention.self_attention.value.weight"]
            q = qk
            k = qk
        q_list.append(q.detach().clone().contiguous())
        k_list.append(k.detach().clone().contiguous())
        v_list.append(v.detach().clone().contiguous())
    return (
        torch.stack(q_list, dim=0).contiguous(),
        torch.stack(k_list, dim=0).contiguous(),
        torch.stack(v_list, dim=0).contiguous(),
    )


def _look_adjacent(x, n_before):
    """Concat each chunk with its ``n_before`` previous chunks (circular).

    Mirrors ``EfficientAttentionMixin._look_adjacent`` for ``n_chunks_after=0``.
    Input ``x`` shape: ``(B, H, nc, chunk_len, ...)``. Output keeps the same
    leading dims but the chunk_len axis becomes ``(1 + n_before) * chunk_len``.
    """
    if n_before == 0:
        return x
    slices = []
    for i in range(-n_before, 1):
        if i == 0:
            slices.append(x)
        else:
            slices.append(torch.cat([x[:, :, i:, ...], x[:, :, :i, ...]], dim=2))
    return torch.cat(slices, dim=3)


class Model(nn.Module):
    def __init__(self, n_head, d_head, attn_layers_str, ae0, ae1, loc_cl, lsh_cl):
        # Python-int / Python-str init args captured on the instance so the
        # forward body can drive integer arithmetic, view(), and per-layer
        # branching without coercing from tensors. Mirrors the gpt2 port's
        # rationale ("n_head is a compile-time integer").
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.attn_layers = tuple(attn_layers_str)  # tuple of 'l'/'L'
        self.ae0 = ae0
        self.ae1 = ae1
        self.loc_cl = loc_cl
        self.lsh_cl = lsh_cl

    def forward(
        self,
        input_ids,         # (B, T) long
        # Axial positional embedding — sliced + reshaped in precompute so the
        # underlying tensors are >=2-D (StepGenFlow requirement) and the
        # forward body does no axial-axis-0 indexing for T <= AX1.
        axial_pos_0,       # (1, AE0)   constant prefix for every position (row 0 of HF's weights[0])
        axial_pos_1,       # (AX1, AE1) axis-1 column embedding (HF's weights[1] with the size-1 dim squeezed)
        wte,               # (V, D)
        # Per-layer (L, ...).
        block_attn_ln_w,   # (L, 1, D)
        block_attn_ln_b,   # (L, 1, D)
        block_attn_q_w,    # (L, D, D)  — for LSH layers, == block_attn_k_w
        block_attn_k_w,    # (L, D, D)
        block_attn_v_w,    # (L, D, D)
        block_attn_out_w,  # (L, D, D)  — no bias (ReformerSelfOutput.dense has bias=False)
        block_ff_ln_w,     # (L, 1, D)
        block_ff_ln_b,     # (L, 1, D)
        block_ff_dense_w,  # (L, FF, D)
        block_ff_dense_b,  # (L, 1, FF)
        block_ff_out_w,    # (L, D, FF)
        block_ff_out_b,    # (L, 1, D)
        # Final encoder LN — runs over 2*D since the reversible-net concatenates
        # the two streams. Stored as (1, 2D) for the >=2-D underlying invariant;
        # F.layer_norm squeezes back to 1-D at the call site.
        enc_ln_w,          # (1, 2D)
        enc_ln_b,          # (1, 2D)
        lm_head_w,         # (V, 2D)   — ReformerOnlyLMHead.decoder, bias=False
    ):
        n_head = self.n_head
        d_head = self.d_head
        B, T = input_ids.shape
        L, _, D = block_attn_ln_w.shape
        assert D == n_head * d_head, f"D={D} must equal n_head*d_head={n_head*d_head}"
        eps = 1e-12  # matches config.layer_norm_eps

        # ---- Embeddings ----
        # Axial pos: pos_emb[b, p, :AE0] = axial_pos_0[0]  (constant across all positions)
        #            pos_emb[b, p, AE0:] = axial_pos_1[p]  (varies with position)
        # Only valid for T <= axial_pos_1.shape[0] (== AX1) — see module docstring.
        pos_part0 = axial_pos_0.unsqueeze(0).expand(B, T, -1)         # (B, T, AE0)
        pos_part1 = axial_pos_1[:T].unsqueeze(0).expand(B, -1, -1)    # (B, T, AE1)
        pos_emb = torch.cat([pos_part0, pos_part1], dim=-1)           # (B, T, D)
        x = wte[input_ids] + pos_emb                                  # (B, T, D)

        # ---- Reversible-net split: both streams start as the embedding. ----
        attn_output = x
        hidden_states = x

        for layer in range(L):
            attn_type = self.attn_layers[layer]

            # ---- Attention sub-block ----
            # Pre-norm on the X_2 stream.
            h_ln = F.layer_norm(
                hidden_states, (D,),
                block_attn_ln_w[layer].squeeze(0),
                block_attn_ln_b[layer].squeeze(0),
                eps,
            )

            # Q/K/V/out — Reformer uses bias=False linears (nn.Linear convention,
            # weight shape (out, in)). Local has separate q/k/v; LSH has shared
            # QK (precompute duplicates query_key into both q_w[l] and k_w[l]).
            q = h_ln @ block_attn_q_w[layer].T
            k = h_ln @ block_attn_k_w[layer].T
            v = h_ln @ block_attn_v_w[layer].T
            q = q.view(B, T, n_head, d_head).transpose(1, 2)  # (B, H, T, d_head)
            k = k.view(B, T, n_head, d_head).transpose(1, 2)
            v = v.view(B, T, n_head, d_head).transpose(1, 2)

            if attn_type == "l":  # local self-attention
                # K is just scale-normalised; Q is unchanged.
                k = k / math.sqrt(d_head)
                if T <= self.loc_cl:
                    # Standard (un-chunked) causal self-attention.
                    scores = q @ k.transpose(-1, -2)                  # (B, H, T, T)
                    t_idx = torch.arange(T, device=scores.device)
                    causal = t_idx.unsqueeze(-1) >= t_idx.unsqueeze(-2)
                    scores = torch.where(causal, scores, scores.new_tensor(_MASK_NEG))
                    # Reformer's attention norm uses logsumexp then exp(scores - lse).
                    # Equivalent to softmax — kept as the HF op sequence for bit-exactness.
                    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                    ctx = torch.exp(scores - lse) @ v                 # (B, H, T, d_head)
                else:
                    # Chunked local attention: chunk into nc chunks of chunk_len
                    # tokens, each chunk attends to itself plus 1 chunk before.
                    nc = T // self.loc_cl
                    q_c = q.view(B, n_head, nc, self.loc_cl, d_head)
                    k_c = _look_adjacent(k.view(B, n_head, nc, self.loc_cl, d_head), 1)
                    v_c = _look_adjacent(v.view(B, n_head, nc, self.loc_cl, d_head), 1)
                    idx = torch.arange(T, device=q.device).view(1, 1, nc, self.loc_cl).expand(
                        B, n_head, nc, self.loc_cl
                    )
                    q_idx = idx
                    k_idx = _look_adjacent(idx, 1)
                    scores = q_c @ k_c.transpose(-1, -2)              # (B, H, nc, cl, (1+nb)*cl)
                    causal = q_idx.unsqueeze(-1) >= k_idx.unsqueeze(-2)
                    scores = torch.where(causal, scores, scores.new_tensor(_MASK_NEG))
                    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                    ctx_c = torch.exp(scores - lse) @ v_c             # (B, H, nc, cl, d_head)
                    ctx = ctx_c.reshape(B, n_head, T, d_head)
            else:  # 'L' — LSH self-attention (standard-mode degenerate form)
                # K is length-normalised (over d_head) AND scaled. This is the
                # ``_len_and_dim_norm`` op from HF's LSHSelfAttention.
                variance = torch.mean(k ** 2, dim=-1, keepdim=True)
                k = (k * torch.rsqrt(variance + 1e-6)) / math.sqrt(d_head)
                # T <= lsh_cl by construction (see module docstring); LSH falls
                # into HF's do_standard_self_attention branch. No hashing/sort.
                scores = q @ k.transpose(-1, -2)                      # (B, H, T, T)
                t_idx = torch.arange(T, device=scores.device)
                causal = t_idx.unsqueeze(-1) >= t_idx.unsqueeze(-2)
                self_mask = t_idx.unsqueeze(-1) != t_idx.unsqueeze(-2)
                # Apply causal mask first (-1e9), then self-mask (-1e5). HF
                # applies them in this order; doing the reverse would make the
                # diagonal -1e9 instead of -1e5 and change the first-token
                # behaviour (the only token whose row has no other valid keys).
                scores = torch.where(causal, scores, scores.new_tensor(_MASK_NEG))
                scores = torch.where(self_mask, scores, scores.new_tensor(_LSH_SELF_MASK_NEG))
                lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                ctx = torch.exp(scores - lse) @ v                     # (B, H, T, d_head)

            ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
            ctx = ctx @ block_attn_out_w[layer].T                     # (B, T, D), no bias
            # RevNet step 1:  Y_1 = X_1 + f(X_2)
            attn_output = attn_output + ctx

            # ---- Feed-forward sub-block ----
            # Pre-norm on the (new) attn_output stream.
            h_ff = F.layer_norm(
                attn_output, (D,),
                block_ff_ln_w[layer].squeeze(0),
                block_ff_ln_b[layer].squeeze(0),
                eps,
            )
            h_ff = h_ff @ block_ff_dense_w[layer].T + block_ff_dense_b[layer]  # (B, T, FF)
            # config.hidden_act = "relu". Local dropouts are 0 so ReLU is the
            # only non-linear step.
            h_ff = F.relu(h_ff)
            h_ff = h_ff @ block_ff_out_w[layer].T + block_ff_out_b[layer]      # (B, T, D)
            # RevNet step 2:  Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + h_ff

        # ---- Final encoder LN over the concatenated 2*D stream ----
        h = torch.cat([attn_output, hidden_states], dim=-1)           # (B, T, 2D)
        h = F.layer_norm(h, (2 * D,), enc_ln_w.squeeze(0), enc_ln_b.squeeze(0), eps)

        # ---- LM head: tied-free linear (bias=False, lm_head.bias is allocated
        # by HF but never used in forward — see ReformerOnlyLMHead.forward_chunk). ----
        logits = h @ lm_head_w.T                                      # (B, T, V)
        return logits


def get_init_inputs(dims):
    # Python-int / -str args captured on the Model instance — they drive
    # integer arithmetic / view() / per-layer branching. The string
    # ``ATTN_LAYERS`` becomes a tuple of 'l'/'L' chars in __init__.
    return [
        dims["H"],
        dims["D_HEAD"],
        dims["ATTN_LAYERS"],
        dims["AE0"],
        dims["AE1"],
        dims["LOC_CL"],
        dims["LSH_CL"],
    ]


def get_inputs(dims):
    torch.manual_seed(SEED)
    cfg = _build_reformer_config(dims)

    # ReformerModelWithLMHead(config) runs the full init under the seed,
    # consuming all weight RNG (including the unused lm_head.bias zero-fill,
    # which doesn't consume RNG, and the axial pos embeddings that get
    # re-init'd to normal(0, axial_norm_std=1.0) in post_init).
    model = ReformerModelWithLMHead(cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    L = dims["L"]

    # Axial position embeddings. HF stores them as:
    #   weights[0] : (AX0, 1,   AE0)
    #   weights[1] : (1,   AX1, AE1)
    # In eval mode + T <= AX1 + position_ids=arange(T), the contribution is:
    #   pos[b, p, :AE0] = weights[0][0, 0, :]   (constant across positions)
    #   pos[b, p, AE0:] = weights[1][0, p, :]   (varies with p)
    # so we precompute compact slices that drop the size-1 axes:
    axial_pos_0 = (
        sd["reformer.embeddings.position_embeddings.weights.0"][0, 0]
        .clone().contiguous().unsqueeze(0)  # (1, AE0)
    )
    axial_pos_1 = (
        sd["reformer.embeddings.position_embeddings.weights.1"][0]
        .clone().contiguous()  # (AX1, AE1)
    )

    wte = sd["reformer.embeddings.word_embeddings.weight"].clone().contiguous()

    # Per-layer attn LN affine.
    block_attn_ln_w = _stack_block_vec(sd, L, "reformer.encoder.layers.{l}.attention.layer_norm.weight")
    block_attn_ln_b = _stack_block_vec(sd, L, "reformer.encoder.layers.{l}.attention.layer_norm.bias")

    # Per-layer Q/K/V/out. Local layers have distinct q/k/v; LSH layers have
    # a shared ``query_key`` linear (duplicated into both q_w and k_w slots).
    attn_types = ["local" if c == "l" else "lsh" for c in dims["ATTN_LAYERS"]]
    block_attn_q_w, block_attn_k_w, block_attn_v_w = _stack_qkv_for_layer(sd, L, attn_types)
    block_attn_out_w = _stack_block(sd, L, "reformer.encoder.layers.{l}.attention.output.dense.weight")

    # Per-layer FFN.
    block_ff_ln_w = _stack_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.layer_norm.weight")
    block_ff_ln_b = _stack_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.layer_norm.bias")
    block_ff_dense_w = _stack_block(sd, L, "reformer.encoder.layers.{l}.feed_forward.dense.dense.weight")
    block_ff_dense_b = _stack_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.dense.dense.bias")
    block_ff_out_w = _stack_block(sd, L, "reformer.encoder.layers.{l}.feed_forward.output.dense.weight")
    block_ff_out_b = _stack_block_vec(sd, L, "reformer.encoder.layers.{l}.feed_forward.output.dense.bias")

    # Final encoder LN over 2*D, stored as (1, 2D) for the >=2-D invariant.
    enc_ln_w = sd["reformer.encoder.layer_norm.weight"].clone().contiguous().unsqueeze(0)
    enc_ln_b = sd["reformer.encoder.layer_norm.bias"].clone().contiguous().unsqueeze(0)

    # LM head decoder (bias=False).
    lm_head_w = sd["lm_head.decoder.weight"].clone().contiguous()

    # Input token ids — final RNG-consuming call, mirrored in precompute.
    input_ids = torch.randint(0, dims["V"], (dims["B"], dims["T"]))

    return [
        input_ids,
        axial_pos_0,
        axial_pos_1,
        wte,
        block_attn_ln_w,
        block_attn_ln_b,
        block_attn_q_w,
        block_attn_k_w,
        block_attn_v_w,
        block_attn_out_w,
        block_ff_ln_w,
        block_ff_ln_b,
        block_ff_dense_w,
        block_ff_dense_b,
        block_ff_out_w,
        block_ff_out_b,
        enc_ln_w,
        enc_ln_b,
        lm_head_w,
    ]


def compute_gold(dims):
    """Forward through the hand-rolled Model; bit-exact vs.
    ``ReformerModelWithLMHead(config)(input_ids).logits`` at every preset
    (verified during port: max abs diff = 0.0)."""
    with torch.no_grad():
        return Model(*get_init_inputs(dims))(*get_inputs(dims))
