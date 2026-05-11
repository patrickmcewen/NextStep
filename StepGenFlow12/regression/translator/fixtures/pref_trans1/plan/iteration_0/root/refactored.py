import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper needed by the parent (post‑attention RMSNorm)
def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # child implements the whole attention pipeline
        self.attention_block = AttentionBlockModel()

    def forward(
        self,
        input_tensor, q_proj, k_proj, v_proj,
        cos, sin, o_proj_weight,
        w_gate, w_up, w_down,
        expert_weights, expert_onehot,
    ):
        """
        Overall transformer layer:
        1. Attention block (delegated to child)
        2. Post‑attention RMSNorm
        3. MoE top‑k routing & expert computation
        4. Final residual addition
        """
        # ----------- 1️⃣ Attention block (child) --------------
        res_add_0 = self.attention_block(
            input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight
        )  # shape [S, hidden_dim]

        # ----------- 2️⃣ Post‑attention RMSNorm --------------
        normed_2 = _rms_norm(res_add_0)   # shape [S, hidden_dim]

        # ----------- 3️⃣ MoE block ---------------------------
        seq_len, dim = normed_2.shape
        n_routed_experts, dim_check, _ = w_gate.shape
        assert dim == dim_check, "MoE weight dim mismatch"

        moe_output = torch.zeros(seq_len, dim, dtype=torch.float32)

        for e_idx in range(n_routed_experts):
            # Find tokens routed to this expert (one‑hot format)
            tok, top_pos = torch.where(expert_onehot[:, :, e_idx] == 1)
            if len(tok) == 0:
                continue

            # Gating & up‑projection
            gate_out = normed_2[tok] @ w_gate[e_idx]          # [T, inter_dim]
            up_out   = normed_2[tok] @ w_up[e_idx]            # [T, inter_dim]

            # Non‑linear activation (SiLU) and down‑projection
            hidden   = F.silu(gate_out) * up_out               # [T, inter_dim]
            down_out = hidden @ w_down[e_idx]                  # [T, dim]

            # Weight the expert contribution and accumulate
            moe_output[tok] += down_out * expert_weights[tok, top_pos, None]

        # ----------- 4️⃣ Final residual add -----------------
        return moe_output + res_add_0


def get_inputs(dims):
    """Return the full argument list expected by the original reference."""
    from precompute import precompute_tensors  # StepDB/precompute.py
    t = precompute_tensors("prefill_transformer_simple", dims)
    return [
        t["input_tensor"], t["q_proj"], t["k_proj"], t["v_proj"],
        t["cos"], t["sin"], t["o_proj_weight"],
        t["w_gate"], t["w_up"], t["w_down"],
        t["expert_weights"], t["expert_onehot"],
    ]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
